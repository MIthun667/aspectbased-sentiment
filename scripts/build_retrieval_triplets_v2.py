from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


class PairDataset(Dataset):
    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=128,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        encoded = self.tokenizer(
            str(row["context"]),
            str(row["term"]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
        )

        sequence_ids = encoded.sequence_ids()

        sentence_mask = [
            1 if x == 0 else 0
            for x in sequence_ids
        ]

        target_mask = [
            1 if x == 1 else 0
            for x in sequence_ids
        ]

        return {
            "input_ids": torch.tensor(
                encoded["input_ids"],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                encoded["attention_mask"],
                dtype=torch.long,
            ),
            "sentence_mask": torch.tensor(
                sentence_mask,
                dtype=torch.long,
            ),
            "target_mask": torch.tensor(
                target_mask,
                dtype=torch.long,
            ),
        }


def masked_mean(states, mask):
    mask = mask.unsqueeze(-1).to(states.dtype)

    return (
        (states * mask).sum(dim=1)
        /
        mask.sum(dim=1).clamp_min(1.0)
    )


@torch.no_grad()
def compute_embeddings(
    dataframe,
    tokenizer,
    model,
    device,
    batch_size,
    max_length,
):
    dataset = PairDataset(
        dataframe,
        tokenizer,
        max_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    representations = []

    model.eval()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sentence_mask = batch["sentence_mask"].to(device)
        target_mask = batch["target_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        hidden = outputs.last_hidden_state

        cls_state = hidden[:, 0]

        sentence_state = masked_mean(
            hidden,
            sentence_mask,
        )

        target_state = masked_mean(
            hidden,
            target_mask,
        )

        interaction = (
            cls_state
            * target_state
        )

        representation = torch.cat(
            [
                cls_state,
                sentence_state,
                target_state,
                interaction,
            ],
            dim=-1,
        )

        representations.append(
            representation.float().cpu()
        )

    embeddings = torch.cat(
        representations,
        dim=0,
    )

    # -----------------------------------------------------
    # Remove the dominant common direction.
    # Raw transformer embeddings are highly anisotropic.
    # -----------------------------------------------------
    embeddings = (
        embeddings
        - embeddings.mean(
            dim=0,
            keepdim=True,
        )
    )

    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=-1,
    )

    return embeddings


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--domain",
        default="Laptop",
        choices=[
            "Laptop",
            "Restaurants",
            "Twitter",
        ],
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            Path.home()
            / "TPHCA"
            / "unFixedData"
        ),
    )

    parser.add_argument(
        "--model",
        default="roberta-base",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    path = (
        args.data_root
        / args.domain
        / "train.csv"
    )

    df = pd.read_csv(
        path
    ).reset_index(drop=True)

    normalized_contexts = (
        df["context"]
        .map(normalize_text)
        .tolist()
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    model = AutoModel.from_pretrained(
        args.model
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    print("=" * 80)
    print("TARGET-AWARE RETRIEVAL V2")
    print("Domain:", args.domain)
    print("N:", len(df))
    print("Model:", args.model)
    print("Same-context retrieval: EXCLUDED")
    print("=" * 80)

    embeddings = compute_embeddings(
        dataframe=df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print(
        "Retrieval embedding shape:",
        tuple(embeddings.shape),
    )

    similarity = (
        embeddings
        @ embeddings.T
    )

    labels = torch.tensor(
        df["polarity"]
        .astype(int)
        .values,
        dtype=torch.long,
    )

    records = []

    top1_pos = []
    top1_neg = []

    for i in range(len(df)):
        same_class = (
            labels == labels[i]
        )

        different_class = (
            labels != labels[i]
        )

        same_context = torch.tensor(
            [
                c == normalized_contexts[i]
                for c in normalized_contexts
            ],
            dtype=torch.bool,
        )

        # Exclude anchor itself AND every row originating
        # from the exact same sentence.
        valid_context = ~same_context

        positive_mask = (
            same_class
            & valid_context
        )

        negative_mask = (
            different_class
            & valid_context
        )

        scores = similarity[i]

        pos_scores = scores.masked_fill(
            ~positive_mask,
            -1e9,
        )

        neg_scores = scores.masked_fill(
            ~negative_mask,
            -1e9,
        )

        k_pos = min(
            args.top_k,
            int(positive_mask.sum()),
        )

        k_neg = min(
            args.top_k,
            int(negative_mask.sum()),
        )

        if k_pos == 0 or k_neg == 0:
            raise RuntimeError(
                f"No candidates for index {i}"
            )

        pv, pi = torch.topk(
            pos_scores,
            k=k_pos,
        )

        nv, ni = torch.topk(
            neg_scores,
            k=k_neg,
        )

        records.append(
            {
                "anchor_index": i,
                "label": int(labels[i]),
                "positive_indices": pi.tolist(),
                "positive_similarities": pv.tolist(),
                "negative_indices": ni.tolist(),
                "negative_similarities": nv.tolist(),
            }
        )

        top1_pos.append(
            float(pv[0])
        )

        top1_neg.append(
            float(nv[0])
        )

    output_dir = (
        Path("data")
        / "retrieval"
        / args.domain.lower()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"target_aware_v2_top{args.top_k}.json"
    )

    payload = {
        "domain": args.domain,
        "retrieval": "target_aware_v2",
        "model": args.model,
        "same_context_excluded": True,
        "top_k": args.top_k,
        "train_size": len(df),
        "mean_top1_positive_similarity": (
            float(np.mean(top1_pos))
        ),
        "mean_top1_negative_similarity": (
            float(np.mean(top1_neg))
        ),
        "negative_ge_positive": int(
            np.sum(
                np.array(top1_neg)
                >=
                np.array(top1_pos)
            )
        ),
        "records": records,
    }

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Top-1 positive mean:",
        np.mean(top1_pos),
    )

    print(
        "Top-1 negative mean:",
        np.mean(top1_neg),
    )

    print(
        "Negative >= positive:",
        payload["negative_ge_positive"],
        "/",
        len(df),
    )

    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
