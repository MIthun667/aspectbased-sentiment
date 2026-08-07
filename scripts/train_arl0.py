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
    Sampler,
)

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.models.aspect_identity_arl0 import (
    AspectIdentityARL0,
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


class AspectIdentityDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=128,
    ):
        self.df = dataframe.reset_index(drop=True)

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

        target_names = sorted(
            self.df["target_norm"].unique()
        )

        self.target2id = {
            target: i
            for i, target
            in enumerate(target_names)
        }

        self.target_ids = [
            self.target2id[x]
            for x in self.df["target_norm"]
        ]

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

        target_mask = [
            1 if sid == 1 else 0
            for sid in sequence_ids
        ]

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

            "target_mask": torch.tensor(
                target_mask,
                dtype=torch.long,
            ),

            "target_id": torch.tensor(
                self.target_ids[idx],
                dtype=torch.long,
            ),

            "row_index": torch.tensor(
                idx,
                dtype=torch.long,
            ),
        }


class RepeatedAspectBatchSampler(Sampler):
    """
    Each batch contains:

        targets_per_batch distinct aspect identities

    and

        samples_per_target examples from each identity.

    Example default:

        8 targets * 2 samples = batch size 16
    """

    def __init__(
        self,
        dataset: AspectIdentityDataset,
        targets_per_batch=8,
        samples_per_target=2,
        batches_per_epoch=200,
        seed=25,
    ):
        self.dataset = dataset

        self.targets_per_batch = (
            targets_per_batch
        )

        self.samples_per_target = (
            samples_per_target
        )

        self.batches_per_epoch = (
            batches_per_epoch
        )

        self.seed = seed

        groups = defaultdict(list)

        for idx, target_id in enumerate(
            dataset.target_ids
        ):
            groups[target_id].append(idx)

        # Only targets with enough examples can
        # provide positive pairs.
        self.groups = {
            target_id: indices
            for target_id, indices
            in groups.items()
            if len(indices) >= samples_per_target
        }

        self.valid_target_ids = list(
            self.groups.keys()
        )

        if (
            len(self.valid_target_ids)
            < targets_per_batch
        ):
            raise RuntimeError(
                "Not enough repeated targets "
                "for requested batch construction."
            )

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        rng = random.Random(
            self.seed
        )

        for _ in range(
            self.batches_per_epoch
        ):
            selected_targets = rng.sample(
                self.valid_target_ids,
                self.targets_per_batch,
            )

            batch = []

            for target_id in selected_targets:
                candidates = (
                    self.groups[target_id]
                )

                selected = rng.sample(
                    candidates,
                    self.samples_per_target,
                )

                batch.extend(selected)

            rng.shuffle(batch)

            yield batch


def supervised_contrastive_loss(
    embeddings,
    labels,
    temperature=0.07,
):
    """
    Standard supervised contrastive objective.

    Every same-target item except self is a positive.
    Every different-target item is a negative.
    """

    embeddings = F.normalize(
        embeddings,
        dim=-1,
    )

    similarity = (
        embeddings
        @ embeddings.T
    ) / temperature

    batch_size = embeddings.size(0)

    identity = torch.eye(
        batch_size,
        dtype=torch.bool,
        device=embeddings.device,
    )

    same = (
        labels.unsqueeze(0)
        ==
        labels.unsqueeze(1)
    )

    positive_mask = (
        same
        & ~identity
    )

    valid_anchor = (
        positive_mask.any(dim=1)
    )

    # Remove self similarity from denominator.
    similarity = similarity.masked_fill(
        identity,
        -1e9,
    )

    log_prob = (
        similarity
        - torch.logsumexp(
            similarity,
            dim=1,
            keepdim=True,
        )
    )

    positive_count = (
        positive_mask
        .sum(dim=1)
        .clamp_min(1)
    )

    mean_positive_log_prob = (
        (
            log_prob
            * positive_mask
            .to(log_prob.dtype)
        ).sum(dim=1)
        /
        positive_count
    )

    loss = (
        -mean_positive_log_prob[
            valid_anchor
        ].mean()
    )

    return loss


@torch.no_grad()
def encode_dataset(
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

    embeddings = []

    for batch in loader:
        z = model(
            input_ids=batch[
                "input_ids"
            ].to(device),

            attention_mask=batch[
                "attention_mask"
            ].to(device),

            target_mask=batch[
                "target_mask"
            ].to(device),
        )

        embeddings.append(
            z.cpu()
        )

    return torch.cat(
        embeddings,
        dim=0,
    )


def retrieval_metrics(
    embeddings,
    dataframe,
):
    """
    Evaluate aspect identity retrieval.

    Same exact sentence is excluded.

    Metrics are computed only for anchors having
    at least one same-target example in another context.
    """

    df = dataframe.reset_index(
        drop=True
    ).copy()

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

    n = len(df)

    eligible = 0

    hit1 = 0
    hit5 = 0

    reciprocal_ranks = []

    for i in range(n):

        same_target = np.array(
            df["target_norm"]
            ==
            df.iloc[i][
                "target_norm"
            ]
        )

        same_context = np.array(
            df["context_norm"]
            ==
            df.iloc[i][
                "context_norm"
            ]
        )

        valid_positive = (
            same_target
            & ~same_context
        )

        valid_candidate = (
            ~same_context
        )

        if valid_positive.sum() == 0:
            continue

        eligible += 1

        scores = (
            similarity[i]
            .clone()
        )

        valid_tensor = torch.tensor(
            valid_candidate,
            dtype=torch.bool,
        )

        scores[
            ~valid_tensor
        ] = -1e9

        ranking = torch.argsort(
            scores,
            descending=True,
        ).tolist()

        positive_indices = set(
            np.where(
                valid_positive
            )[0].tolist()
        )

        if ranking[0] in positive_indices:
            hit1 += 1

        if any(
            idx in positive_indices
            for idx in ranking[:5]
        ):
            hit5 += 1

        rank = next(
            r + 1
            for r, idx
            in enumerate(ranking)
            if idx in positive_indices
        )

        reciprocal_ranks.append(
            1.0 / rank
        )

    return {
        "eligible_anchors": eligible,

        "same_target_at_1": (
            hit1 / eligible
            if eligible else 0.0
        ),

        "same_target_at_5": (
            hit5 / eligible
            if eligible else 0.0
        ),

        "mrr": float(
            np.mean(
                reciprocal_ranks
            )
        ) if reciprocal_ranks else 0.0,
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
        "--projection-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--targets-per-batch",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--samples-per-target",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
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

    train_path = (
        args.data_root
        / args.domain
        / "train.csv"
    )

    df = pd.read_csv(
        train_path
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.model,
            use_fast=True,
        )
    )

    dataset = AspectIdentityDataset(
        df,
        tokenizer,
        max_length=args.max_length,
    )

    sampler = RepeatedAspectBatchSampler(
        dataset=dataset,

        targets_per_batch=(
            args.targets_per_batch
        ),

        samples_per_target=(
            args.samples_per_target
        ),

        batches_per_epoch=(
            args.batches_per_epoch
        ),

        seed=args.seed,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = AspectIdentityARL0(
        backbone_name=args.model,
        projection_dim=(
            args.projection_dim
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
    print("ARL-0 — ASPECT IDENTITY LEARNING")
    print("Domain:", args.domain)
    print("Train N:", len(dataset))
    print(
        "Repeated target identities:",
        len(
            sampler.valid_target_ids
        ),
    )
    print(
        "Batch size:",
        args.targets_per_batch
        * args.samples_per_target,
    )
    print(
        "RoBERTa frozen:",
        True,
    )
    print("=" * 80)

    # -------------------------------------------------
    # Baseline:
    # random projection head before learning.
    # -------------------------------------------------
    initial_embeddings = (
        encode_dataset(
            model,
            dataset,
            device,
        )
    )

    initial_metrics = (
        retrieval_metrics(
            initial_embeddings,
            df,
        )
    )

    print(
        "\nINITIAL RETRIEVAL"
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

        running_loss = 0.0

        for step, batch in enumerate(
            loader,
            start=1,
        ):
            optimizer.zero_grad(
                set_to_none=True
            )

            z = model(
                input_ids=batch[
                    "input_ids"
                ].to(device),

                attention_mask=batch[
                    "attention_mask"
                ].to(device),

                target_mask=batch[
                    "target_mask"
                ].to(device),
            )

            labels = batch[
                "target_id"
            ].to(device)

            loss = (
                supervised_contrastive_loss(
                    embeddings=z,
                    labels=labels,
                    temperature=(
                        args.temperature
                    ),
                )
            )

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

            running_loss += (
                loss.item()
            )

        embeddings = encode_dataset(
            model,
            dataset,
            device,
        )

        metrics = retrieval_metrics(
            embeddings,
            df,
        )

        avg_loss = (
            running_loss
            / len(loader)
        )

        print(
            f"\nEpoch {epoch}"
        )

        print(
            "loss:",
            round(
                avg_loss,
                6,
            ),
        )

        print(
            "same-target@1:",
            round(
                metrics[
                    "same_target_at_1"
                ],
                4,
            ),
        )

        print(
            "same-target@5:",
            round(
                metrics[
                    "same_target_at_5"
                ],
                4,
            ),
        )

        print(
            "MRR:",
            round(
                metrics[
                    "mrr"
                ],
                4,
            ),
        )

        if metrics["mrr"] > best_mrr:
            best_mrr = metrics[
                "mrr"
            ]

            best_state = {
                k: v.detach().cpu().clone()
                for k, v
                in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "arl0"
        / args.domain.lower()
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_dir
        / "aspect_identity_projector.pt"
    )

    torch.save(
        model.state_dict(),
        model_path,
    )

    final_embeddings = encode_dataset(
        model,
        dataset,
        device,
    )

    final_metrics = retrieval_metrics(
        final_embeddings,
        df,
    )

    torch.save(
        final_embeddings,
        output_dir
        / "train_embeddings.pt",
    )

    metadata = {
        "experiment": "ARL-0",
        "domain": args.domain,
        "backbone": args.model,
        "backbone_frozen": True,
        "projection_dim": (
            args.projection_dim
        ),
        "temperature": (
            args.temperature
        ),
        "epochs": args.epochs,
        "seed": args.seed,
        "initial_retrieval": (
            initial_metrics
        ),
        "final_retrieval": (
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

    print("\n" + "=" * 80)
    print("FINAL ARL-0 RETRIEVAL")
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
