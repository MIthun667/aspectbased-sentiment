from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


class PairDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int = 128,
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
            return_tensors="pt",
        )

        return {
            "index": idx,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


@torch.no_grad()
def compute_embeddings(
    df,
    tokenizer,
    model,
    device,
    batch_size,
    max_length,
):
    dataset = PairDataset(
        df,
        tokenizer,
        max_length=max_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()

    all_embeddings = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # RoBERTa <s> representation.
        cls = outputs.last_hidden_state[:, 0]

        cls = F.normalize(
            cls.float(),
            p=2,
            dim=-1,
        )

        all_embeddings.append(
            cls.cpu()
        )

    return torch.cat(
        all_embeddings,
        dim=0,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--domain",
        default="Laptop",
        choices=["Laptop", "Restaurants", "Twitter"],
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "TPHCA" / "unFixedData",
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

    parser.add_argument(
        "--seed",
        type=int,
        default=25,
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_path = (
        args.data_root
        / args.domain
        / "train.csv"
    )

    df = pd.read_csv(train_path).reset_index(drop=True)

    print("=" * 80)
    print("RETRIEVAL TRIPLET BUILDER")
    print("Domain:", args.domain)
    print("Train:", train_path)
    print("N:", len(df))
    print("Model:", args.model)
    print("Top-k:", args.top_k)
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    model = AutoModel.from_pretrained(
        args.model
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    embeddings = compute_embeddings(
        df=df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print(
        "Embeddings:",
        tuple(embeddings.shape),
    )

    # N is only ~2k for Laptop, so full cosine
    # similarity is perfectly manageable.
    similarity = (
        embeddings
        @ embeddings.T
    )

    labels = torch.tensor(
        df["polarity"].astype(int).values
    )

    retrieval = []

    positive_sims = []
    negative_sims = []

    for i in range(len(df)):
        label_i = labels[i]

        same_label = labels == label_i
        different_label = labels != label_i

        # Never retrieve itself.
        same_label[i] = False

        # Avoid exact duplicate target-context rows.
        same_pair = (
            (
                df["context"].astype(str)
                == str(df.iloc[i]["context"])
            )
            &
            (
                df["term"].astype(str)
                == str(df.iloc[i]["term"])
            )
        ).to_numpy()

        same_pair = torch.tensor(
            same_pair,
            dtype=torch.bool,
        )

        same_label = (
            same_label
            & ~same_pair
        )

        different_label = (
            different_label
            & ~same_pair
        )

        sim_i = similarity[i]

        pos_scores = sim_i.clone()
        pos_scores[~same_label] = -1e9

        neg_scores = sim_i.clone()
        neg_scores[~different_label] = -1e9

        k_pos = min(
            args.top_k,
            int(same_label.sum()),
        )

        k_neg = min(
            args.top_k,
            int(different_label.sum()),
        )

        if k_pos == 0 or k_neg == 0:
            raise RuntimeError(
                f"No valid retrieval candidates for row {i}"
            )

        pos_values, pos_indices = torch.topk(
            pos_scores,
            k=k_pos,
        )

        neg_values, neg_indices = torch.topk(
            neg_scores,
            k=k_neg,
        )

        record = {
            "anchor_index": i,
            "label": int(label_i),
            "positive_indices": [
                int(x)
                for x in pos_indices.tolist()
            ],
            "positive_similarities": [
                float(x)
                for x in pos_values.tolist()
            ],
            "negative_indices": [
                int(x)
                for x in neg_indices.tolist()
            ],
            "negative_similarities": [
                float(x)
                for x in neg_values.tolist()
            ],
        }

        retrieval.append(record)

        positive_sims.append(
            float(pos_values[0])
        )

        negative_sims.append(
            float(neg_values[0])
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
        / f"roberta_pair_top{args.top_k}.json"
    )

    payload = {
        "domain": args.domain,
        "model": args.model,
        "top_k": args.top_k,
        "seed": args.seed,
        "train_size": len(df),
        "mean_top1_positive_similarity": float(
            np.mean(positive_sims)
        ),
        "mean_top1_negative_similarity": float(
            np.mean(negative_sims)
        ),
        "records": retrieval,
    }

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Mean top-1 positive similarity:",
          np.mean(positive_sims))

    print("Mean top-1 negative similarity:",
          np.mean(negative_sims))

    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
