from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from transformers import AutoTokenizer


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.models.target_evidence_trrl2 import (
    TargetEvidenceTRRL2,
)


def normalize_text(x: str) -> str:
    x = str(x).lower().strip()
    x = re.sub(r"\s+", " ", x)
    return x


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RelationBaseDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=128,
    ):
        self.df = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.df["target_norm"] = (
            self.df["term"]
            .map(normalize_text)
        )

        self.df["context_norm"] = (
            self.df["context"]
            .map(normalize_text)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(
        self,
        idx,
    ):
        row = self.df.iloc[idx]

        encoded = self.tokenizer(
            str(row["context"]),
            str(row["term"]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
        )

        sequence_ids = (
            encoded.sequence_ids()
        )

        sentence_mask = [
            1 if sid == 0 else 0
            for sid in sequence_ids
        ]

        target_mask = [
            1 if sid == 1 else 0
            for sid in sequence_ids
        ]

        if sum(sentence_mask) == 0:
            raise RuntimeError(
                f"No sentence tokens at row {idx}"
            )

        if sum(target_mask) == 0:
            raise RuntimeError(
                f"Target truncated at row {idx}"
            )

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

            "polarity": torch.tensor(
                int(row["polarity"]),
                dtype=torch.long,
            ),

            "row_index": torch.tensor(
                idx,
                dtype=torch.long,
            ),
        }


def build_triplet_candidates(
    dataframe,
):
    df = (
        dataframe
        .reset_index(drop=True)
        .copy()
    )

    df["target_norm"] = (
        df["term"]
        .map(normalize_text)
    )

    df["context_norm"] = (
        df["context"]
        .map(normalize_text)
    )

    groups = defaultdict(list)

    for idx, row in df.iterrows():
        groups[
            row["target_norm"]
        ].append(idx)

    records = []

    for i, row in df.iterrows():

        positives = []
        negatives = []

        for j in groups[
            row["target_norm"]
        ]:
            if i == j:
                continue

            candidate = df.iloc[j]

            if (
                candidate["context_norm"]
                ==
                row["context_norm"]
            ):
                continue

            if (
                int(candidate["polarity"])
                ==
                int(row["polarity"])
            ):
                positives.append(j)

            else:
                negatives.append(j)

        if positives and negatives:
            records.append(
                {
                    "anchor": i,
                    "positives": positives,
                    "negatives": negatives,
                }
            )

    return records


class TripletDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        triplets,
    ):
        self.base = base_dataset
        self.triplets = triplets

    def __len__(self):
        return len(
            self.triplets
        )

    def __getitem__(
        self,
        idx,
    ):
        record = self.triplets[
            idx
        ]

        pos_idx = random.choice(
            record["positives"]
        )

        neg_idx = random.choice(
            record["negatives"]
        )

        anchor = self.base[
            record["anchor"]
        ]

        positive = self.base[
            pos_idx
        ]

        negative = self.base[
            neg_idx
        ]

        output = {}

        for prefix, sample in [
            ("anchor", anchor),
            ("positive", positive),
            ("negative", negative),
        ]:

            for field in [
                "input_ids",
                "attention_mask",
                "sentence_mask",
                "target_mask",
            ]:
                output[
                    f"{prefix}_{field}"
                ] = sample[field]

        return output


def encode_side(
    model,
    batch,
    prefix,
    device,
):
    return model(
        input_ids=batch[
            f"{prefix}_input_ids"
        ].to(device),

        attention_mask=batch[
            f"{prefix}_attention_mask"
        ].to(device),

        sentence_mask=batch[
            f"{prefix}_sentence_mask"
        ].to(device),

        target_mask=batch[
            f"{prefix}_target_mask"
        ].to(device),
    )


@torch.no_grad()
def encode_full_dataset(
    model,
    dataset,
    device,
    batch_size=64,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()

    result = []

    for batch in loader:

        z = model(
            input_ids=batch[
                "input_ids"
            ].to(device),

            attention_mask=batch[
                "attention_mask"
            ].to(device),

            sentence_mask=batch[
                "sentence_mask"
            ].to(device),

            target_mask=batch[
                "target_mask"
            ].to(device),
        )

        result.append(
            z.cpu()
        )

    return torch.cat(
        result,
        dim=0,
    )


def relation_retrieval_metrics(
    embeddings,
    dataframe,
):
    df = (
        dataframe
        .reset_index(drop=True)
        .copy()
    )

    df["target_norm"] = (
        df["term"]
        .map(normalize_text)
    )

    df["context_norm"] = (
        df["context"]
        .map(normalize_text)
    )

    similarity = (
        embeddings
        @ embeddings.T
    )

    eligible = 0

    hit1 = 0
    hit3 = 0

    reciprocal_ranks = []

    same_best = []
    different_best = []

    for i, row in df.iterrows():

        same_target = np.array(
            df["target_norm"]
            ==
            row["target_norm"]
        )

        same_context = np.array(
            df["context_norm"]
            ==
            row["context_norm"]
        )

        same_polarity = np.array(
            df["polarity"]
            ==
            row["polarity"]
        )

        candidates = (
            same_target
            & ~same_context
        )

        positives = (
            candidates
            & same_polarity
        )

        negatives = (
            candidates
            & ~same_polarity
        )

        if (
            positives.sum() == 0
            or negatives.sum() == 0
        ):
            continue

        eligible += 1

        scores = (
            similarity[i]
            .clone()
        )

        valid = torch.tensor(
            candidates,
            dtype=torch.bool,
        )

        scores[
            ~valid
        ] = -1e9

        ranking = torch.argsort(
            scores,
            descending=True,
        ).tolist()

        positive_indices = set(
            np.where(
                positives
            )[0].tolist()
        )

        negative_indices = (
            np.where(
                negatives
            )[0].tolist()
        )

        if (
            ranking[0]
            in positive_indices
        ):
            hit1 += 1

        if any(
            idx in positive_indices
            for idx in ranking[:3]
        ):
            hit3 += 1

        rank = next(
            rank_idx + 1
            for rank_idx, candidate
            in enumerate(ranking)
            if candidate
            in positive_indices
        )

        reciprocal_ranks.append(
            1.0 / rank
        )

        same_scores = [
            float(
                similarity[
                    i,
                    j,
                ]
            )
            for j
            in positive_indices
        ]

        diff_scores = [
            float(
                similarity[
                    i,
                    j,
                ]
            )
            for j
            in negative_indices
        ]

        same_best.append(
            max(
                same_scores
            )
        )

        different_best.append(
            max(
                diff_scores
            )
        )

    pos = np.asarray(
        same_best
    )

    neg = np.asarray(
        different_best
    )

    return {
        "eligible_anchors": (
            eligible
        ),

        "same_polarity_at_1": (
            hit1 / eligible
            if eligible else 0.0
        ),

        "same_polarity_at_3": (
            hit3 / eligible
            if eligible else 0.0
        ),

        "mrr": (
            float(
                np.mean(
                    reciprocal_ranks
                )
            )
            if reciprocal_ranks
            else 0.0
        ),

        "mean_best_same_polarity_similarity": (
            float(
                pos.mean()
            )
            if len(pos)
            else 0.0
        ),

        "mean_best_different_polarity_similarity": (
            float(
                neg.mean()
            )
            if len(neg)
            else 0.0
        ),

        "different_ge_same_count": (
            int(
                (neg >= pos)
                .sum()
            )
            if len(pos)
            else 0
        ),
    }


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
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--relation-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--attention-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=25,
    )

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    path = (
        args.data_root
        / args.domain
        / "train.csv"
    )

    df = pd.read_csv(
        path
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.model,
            use_fast=True,
        )
    )

    base_dataset = (
        RelationBaseDataset(
            dataframe=df,
            tokenizer=tokenizer,
            max_length=(
                args.max_length
            ),
        )
    )

    triplets = (
        build_triplet_candidates(
            df
        )
    )

    triplet_dataset = (
        TripletDataset(
            base_dataset,
            triplets,
        )
    )

    loader = DataLoader(
        triplet_dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = TargetEvidenceTRRL2(
        backbone_name=(
            args.model
        ),
        relation_dim=(
            args.relation_dim
        ),
        attention_dim=(
            args.attention_dim
        ),
        dropout=0.1,
        freeze_backbone=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        (
            p
            for p in model.parameters()
            if p.requires_grad
        ),
        lr=args.lr,
        weight_decay=0.01,
    )

    print("=" * 80)
    print(
        "TRRL-2 — TARGET-CONDITIONED "
        "EVIDENCE RELATION LEARNING"
    )
    print("=" * 80)

    print(
        "Train rows:",
        len(df),
    )

    print(
        "Eligible anchors:",
        len(
            triplets
        ),
    )

    print(
        "Coverage:",
        f"{100 * len(triplets) / len(df):.2f}%"
    )

    print(
        "Backbone frozen:",
        True,
    )

    initial_embeddings = (
        encode_full_dataset(
            model,
            base_dataset,
            device,
        )
    )

    initial_metrics = (
        relation_retrieval_metrics(
            initial_embeddings,
            df,
        )
    )

    print(
        "\nINITIAL RELATION RETRIEVAL"
    )

    print(
        json.dumps(
            initial_metrics,
            indent=2,
        )
    )

    best_mrr = -1.0
    best_state = None

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        total_loss = 0.0
        total_pos = 0.0
        total_neg = 0.0

        for batch in loader:

            optimizer.zero_grad(
                set_to_none=True
            )

            anchor = encode_side(
                model,
                batch,
                "anchor",
                device,
            )

            positive = encode_side(
                model,
                batch,
                "positive",
                device,
            )

            negative = encode_side(
                model,
                batch,
                "negative",
                device,
            )

            pos_distance = (
                1.0
                -
                F.cosine_similarity(
                    anchor,
                    positive,
                    dim=-1,
                )
            )

            neg_distance = (
                1.0
                -
                F.cosine_similarity(
                    anchor,
                    negative,
                    dim=-1,
                )
            )

            loss = F.relu(
                pos_distance
                -
                neg_distance
                +
                args.margin
            ).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                (
                    p
                    for p
                    in model.parameters()
                    if p.requires_grad
                ),
                1.0,
            )

            optimizer.step()

            total_loss += (
                loss.item()
            )

            total_pos += float(
                pos_distance
                .mean()
                .detach()
                .cpu()
            )

            total_neg += float(
                neg_distance
                .mean()
                .detach()
                .cpu()
            )

        embeddings = (
            encode_full_dataset(
                model,
                base_dataset,
                device,
            )
        )

        metrics = (
            relation_retrieval_metrics(
                embeddings,
                df,
            )
        )

        steps = len(loader)

        print(
            f"\nEpoch {epoch}"
        )

        print(
            "triplet loss:",
            round(
                total_loss / steps,
                6,
            )
        )

        print(
            "positive distance:",
            round(
                total_pos / steps,
                4,
            )
        )

        print(
            "negative distance:",
            round(
                total_neg / steps,
                4,
            )
        )

        print(
            "same-polarity@1:",
            round(
                metrics[
                    "same_polarity_at_1"
                ],
                4,
            )
        )

        print(
            "same-polarity@3:",
            round(
                metrics[
                    "same_polarity_at_3"
                ],
                4,
            )
        )

        print(
            "MRR:",
            round(
                metrics[
                    "mrr"
                ],
                4,
            )
        )

        print(
            "different>=same:",
            metrics[
                "different_ge_same_count"
            ],
            "/",
            metrics[
                "eligible_anchors"
            ],
        )

        if (
            metrics["mrr"]
            >
            best_mrr
        ):
            best_mrr = (
                metrics["mrr"]
            )

            best_state = {
                key: value
                .detach()
                .cpu()
                .clone()

                for key, value
                in model
                .state_dict()
                .items()
            }

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    final_embeddings = (
        encode_full_dataset(
            model,
            base_dataset,
            device,
        )
    )

    final_metrics = (
        relation_retrieval_metrics(
            final_embeddings,
            df,
        )
    )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "trrl2"
        / args.domain.lower()
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        output_dir
        / "relation_encoder.pt",
    )

    torch.save(
        final_embeddings,
        output_dir
        / "relation_embeddings.pt",
    )

    metadata = {
        "experiment": (
            "TRRL-2"
        ),

        "representation": (
            "target_conditioned_evidence"
        ),

        "domain": (
            args.domain
        ),

        "backbone": (
            args.model
        ),

        "backbone_frozen": True,

        "relation_dim": (
            args.relation_dim
        ),

        "attention_dim": (
            args.attention_dim
        ),

        "margin": (
            args.margin
        ),

        "seed": (
            args.seed
        ),

        "eligible_anchors": (
            len(triplets)
        ),

        "initial_relation_retrieval": (
            initial_metrics
        ),

        "final_relation_retrieval": (
            final_metrics
        ),
    }

    (
        output_dir
        / "metrics.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL TRRL-2 RELATION RETRIEVAL"
    )

    print(
        json.dumps(
            final_metrics,
            indent=2,
        )
    )

    print(
        "Saved:",
        output_dir,
    )


if __name__ == "__main__":
    main()
