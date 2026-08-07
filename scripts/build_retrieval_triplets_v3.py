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
            1 if sid == 0 else 0
            for sid in sequence_ids
        ]

        target_mask = [
            1 if sid == 1 else 0
            for sid in sequence_ids
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


def center_and_normalize(x):
    x = x - x.mean(
        dim=0,
        keepdim=True,
    )

    return F.normalize(
        x,
        p=2,
        dim=-1,
    )


@torch.no_grad()
def compute_retrieval_spaces(
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
        max_length=max_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()

    target_vectors = []
    context_vectors = []

    for batch in loader:
        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        sentence_mask = batch[
            "sentence_mask"
        ].to(device)

        target_mask = batch[
            "target_mask"
        ].to(device)

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

        # ------------------------------------------------
        # Target retrieval space
        # ------------------------------------------------
        target_vector = target_state

        # ------------------------------------------------
        # Context retrieval space
        #
        # Keep contextual global information separate
        # from target identity.
        # ------------------------------------------------
        context_vector = torch.cat(
            [
                cls_state,
                sentence_state,
            ],
            dim=-1,
        )

        target_vectors.append(
            target_vector.float().cpu()
        )

        context_vectors.append(
            context_vector.float().cpu()
        )

    target_vectors = torch.cat(
        target_vectors,
        dim=0,
    )

    context_vectors = torch.cat(
        context_vectors,
        dim=0,
    )

    target_vectors = center_and_normalize(
        target_vectors
    )

    context_vectors = center_and_normalize(
        context_vectors
    )

    return (
        target_vectors,
        context_vectors,
    )


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

    parser.add_argument(
        "--target-weight",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--exact-target-bonus",
        type=float,
        default=0.20,
    )

    args = parser.parse_args()

    if not 0.0 <= args.target_weight <= 1.0:
        raise ValueError(
            "--target-weight must be in [0, 1]"
        )

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

    normalized_targets = (
        df["term"]
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
    print("TARGET-FIRST RETRIEVAL V3")
    print("Domain:", args.domain)
    print("N:", len(df))
    print("Model:", args.model)
    print(
        "Target weight:",
        args.target_weight,
    )
    print(
        "Context weight:",
        1.0 - args.target_weight,
    )
    print(
        "Exact-target bonus:",
        args.exact_target_bonus,
    )
    print(
        "Same-context retrieval: EXCLUDED"
    )
    print("=" * 80)

    (
        target_vectors,
        context_vectors,
    ) = compute_retrieval_spaces(
        dataframe=df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print(
        "Target embedding shape:",
        tuple(target_vectors.shape),
    )

    print(
        "Context embedding shape:",
        tuple(context_vectors.shape),
    )

    target_similarity = (
        target_vectors
        @ target_vectors.T
    )

    context_similarity = (
        context_vectors
        @ context_vectors.T
    )

    alpha = args.target_weight

    combined_similarity = (
        alpha
        * target_similarity
        +
        (1.0 - alpha)
        * context_similarity
    )

    # ----------------------------------------------------
    # Exact-target bonus
    # ----------------------------------------------------
    n = len(df)

    exact_target_matrix = torch.zeros(
        (n, n),
        dtype=combined_similarity.dtype,
    )

    target_to_indices = {}

    for i, target in enumerate(
        normalized_targets
    ):
        target_to_indices.setdefault(
            target,
            [],
        ).append(i)

    for indices in target_to_indices.values():
        if len(indices) < 2:
            continue

        idx = torch.tensor(
            indices,
            dtype=torch.long,
        )

        exact_target_matrix[
            idx.unsqueeze(1),
            idx.unsqueeze(0),
        ] = args.exact_target_bonus

    combined_similarity = (
        combined_similarity
        + exact_target_matrix
    )

    labels = torch.tensor(
        df["polarity"]
        .astype(int)
        .values,
        dtype=torch.long,
    )

    records = []

    top1_positive_scores = []
    top1_negative_scores = []

    top1_positive_target_scores = []
    top1_negative_target_scores = []

    top1_positive_context_scores = []
    top1_negative_context_scores = []

    exact_positive_count = 0
    exact_negative_count = 0

    for i in range(n):
        same_class = (
            labels == labels[i]
        )

        different_class = (
            labels != labels[i]
        )

        same_context = torch.tensor(
            [
                ctx == normalized_contexts[i]
                for ctx in normalized_contexts
            ],
            dtype=torch.bool,
        )

        valid_context = ~same_context

        positive_mask = (
            same_class
            & valid_context
        )

        negative_mask = (
            different_class
            & valid_context
        )

        pos_scores = (
            combined_similarity[i]
            .masked_fill(
                ~positive_mask,
                -1e9,
            )
        )

        neg_scores = (
            combined_similarity[i]
            .masked_fill(
                ~negative_mask,
                -1e9,
            )
        )

        k_pos = min(
            args.top_k,
            int(
                positive_mask
                .sum()
            ),
        )

        k_neg = min(
            args.top_k,
            int(
                negative_mask
                .sum()
            ),
        )

        if (
            k_pos == 0
            or k_neg == 0
        ):
            raise RuntimeError(
                f"No candidates for row {i}"
            )

        pos_values, pos_indices = (
            torch.topk(
                pos_scores,
                k=k_pos,
            )
        )

        neg_values, neg_indices = (
            torch.topk(
                neg_scores,
                k=k_neg,
            )
        )

        p0 = int(
            pos_indices[0]
        )

        n0 = int(
            neg_indices[0]
        )

        if (
            normalized_targets[p0]
            ==
            normalized_targets[i]
        ):
            exact_positive_count += 1

        if (
            normalized_targets[n0]
            ==
            normalized_targets[i]
        ):
            exact_negative_count += 1

        record = {
            "anchor_index": i,
            "label": int(
                labels[i]
            ),

            "positive_indices": [
                int(x)
                for x in pos_indices.tolist()
            ],

            "negative_indices": [
                int(x)
                for x in neg_indices.tolist()
            ],

            "positive_scores": [
                float(x)
                for x in pos_values.tolist()
            ],

            "negative_scores": [
                float(x)
                for x in neg_values.tolist()
            ],

            "positive_target_similarities": [
                float(
                    target_similarity[
                        i,
                        int(x),
                    ]
                )
                for x
                in pos_indices.tolist()
            ],

            "negative_target_similarities": [
                float(
                    target_similarity[
                        i,
                        int(x),
                    ]
                )
                for x
                in neg_indices.tolist()
            ],

            "positive_context_similarities": [
                float(
                    context_similarity[
                        i,
                        int(x),
                    ]
                )
                for x
                in pos_indices.tolist()
            ],

            "negative_context_similarities": [
                float(
                    context_similarity[
                        i,
                        int(x),
                    ]
                )
                for x
                in neg_indices.tolist()
            ],

            "positive_exact_target": [
                (
                    normalized_targets[
                        int(x)
                    ]
                    ==
                    normalized_targets[i]
                )
                for x
                in pos_indices.tolist()
            ],

            "negative_exact_target": [
                (
                    normalized_targets[
                        int(x)
                    ]
                    ==
                    normalized_targets[i]
                )
                for x
                in neg_indices.tolist()
            ],
        }

        records.append(
            record
        )

        top1_positive_scores.append(
            float(pos_values[0])
        )

        top1_negative_scores.append(
            float(neg_values[0])
        )

        top1_positive_target_scores.append(
            float(
                target_similarity[
                    i,
                    p0,
                ]
            )
        )

        top1_negative_target_scores.append(
            float(
                target_similarity[
                    i,
                    n0,
                ]
            )
        )

        top1_positive_context_scores.append(
            float(
                context_similarity[
                    i,
                    p0,
                ]
            )
        )

        top1_negative_context_scores.append(
            float(
                context_similarity[
                    i,
                    n0,
                ]
            )
        )

    pos_arr = np.asarray(
        top1_positive_scores
    )

    neg_arr = np.asarray(
        top1_negative_scores
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
        / (
            f"target_first_v3_"
            f"top{args.top_k}.json"
        )
    )

    payload = {
        "domain": args.domain,
        "retrieval": (
            "target_first_v3"
        ),
        "model": args.model,
        "train_size": n,
        "top_k": args.top_k,
        "target_weight": (
            args.target_weight
        ),
        "context_weight": (
            1.0
            - args.target_weight
        ),
        "exact_target_bonus": (
            args.exact_target_bonus
        ),
        "same_context_excluded": True,

        "mean_top1_positive_score": float(
            pos_arr.mean()
        ),

        "mean_top1_negative_score": float(
            neg_arr.mean()
        ),

        "negative_ge_positive": int(
            (neg_arr >= pos_arr).sum()
        ),

        "mean_top1_positive_target_similarity": float(
            np.mean(
                top1_positive_target_scores
            )
        ),

        "mean_top1_negative_target_similarity": float(
            np.mean(
                top1_negative_target_scores
            )
        ),

        "mean_top1_positive_context_similarity": float(
            np.mean(
                top1_positive_context_scores
            )
        ),

        "mean_top1_negative_context_similarity": float(
            np.mean(
                top1_negative_context_scores
            )
        ),

        "exact_target_top1_positive_count": (
            exact_positive_count
        ),

        "exact_target_top1_negative_count": (
            exact_negative_count
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
        "Mean top-1 positive score:",
        pos_arr.mean(),
    )

    print(
        "Mean top-1 negative score:",
        neg_arr.mean(),
    )

    print(
        "Negative >= positive:",
        int(
            (neg_arr >= pos_arr).sum()
        ),
        "/",
        n,
    )

    print()
    print(
        "Mean positive TARGET sim:",
        np.mean(
            top1_positive_target_scores
        ),
    )

    print(
        "Mean negative TARGET sim:",
        np.mean(
            top1_negative_target_scores
        ),
    )

    print(
        "Mean positive CONTEXT sim:",
        np.mean(
            top1_positive_context_scores
        ),
    )

    print(
        "Mean negative CONTEXT sim:",
        np.mean(
            top1_negative_context_scores
        ),
    )

    print()
    print(
        "Top-1 exact-target positives:",
        exact_positive_count,
        "/",
        n,
    )

    print(
        "Top-1 exact-target negatives:",
        exact_negative_count,
        "/",
        n,
    )

    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
