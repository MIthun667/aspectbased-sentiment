from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


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


from src.models.target_relational_m3 import (
    TargetRelationalM3,
)


def set_seed(
    seed: int,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


class LiteratureABSADataset(
    Dataset
):
    def __init__(
        self,
        csv_path: Path,
        tokenizer,
        max_length: int = 128,
    ):
        self.data = pd.read_csv(
            csv_path
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

        required = {
            "term",
            "context",
            "polarity",
        }

        missing = (
            required
            - set(
                self.data.columns
            )
        )

        if missing:
            raise ValueError(
                f"Missing columns: "
                f"{sorted(missing)}"
            )

    def __len__(
        self,
    ):
        return len(
            self.data
        )

    def __getitem__(
        self,
        idx,
    ):
        row = (
            self.data
            .iloc[idx]
        )

        sentence = str(
            row["context"]
        )

        aspect = str(
            row["term"]
        )

        label = int(
            row["polarity"]
        )

        encoded = (
            self.tokenizer(
                sentence,
                aspect,
                truncation=True,
                max_length=(
                    self.max_length
                ),
                padding=(
                    "max_length"
                ),
                return_attention_mask=True,
            )
        )

        sequence_ids = (
            encoded
            .sequence_ids()
        )

        sentence_mask = [
            1 if sid == 0 else 0
            for sid in sequence_ids
        ]

        target_mask = [
            1 if sid == 1 else 0
            for sid in sequence_ids
        ]

        if (
            sum(
                sentence_mask
            )
            == 0
        ):
            raise RuntimeError(
                f"No sentence tokens "
                f"for sample {idx}"
            )

        if (
            sum(
                target_mask
            )
            == 0
        ):
            raise RuntimeError(
                f"Target truncated: "
                f"{aspect!r}"
            )

        return {
            "input_ids": (
                torch.tensor(
                    encoded[
                        "input_ids"
                    ],
                    dtype=torch.long,
                )
            ),

            "attention_mask": (
                torch.tensor(
                    encoded[
                        "attention_mask"
                    ],
                    dtype=torch.long,
                )
            ),

            "sentence_mask": (
                torch.tensor(
                    sentence_mask,
                    dtype=torch.long,
                )
            ),

            "target_mask": (
                torch.tensor(
                    target_mask,
                    dtype=torch.long,
                )
            ),

            "labels": (
                torch.tensor(
                    label,
                    dtype=torch.long,
                )
            ),
        }


def compute_metrics(
    eval_pred,
):
    logits, labels = (
        eval_pred
    )

    # Defensive handling for HuggingFace models that may
    # expose additional outputs.
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    return {
        "accuracy": (
            accuracy_score(
                labels,
                predictions,
            )
        ),

        "macro_f1": (
            f1_score(
                labels,
                predictions,
                average="macro",
            )
        ),
    }


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
):
    """
    Batch-hard triplet loss over target-relational embeddings.

    embeddings:
        [B, D], L2-normalized

    hardest positive:
        same class with largest cosine distance

    hardest negative:
        different class with smallest cosine distance

    Anchors without a valid positive or negative are ignored.
    """

    batch_size = (
        embeddings.size(0)
    )

    if batch_size < 3:
        zero = (
            embeddings.sum()
            * 0.0
        )

        return (
            zero,
            {
                "valid_anchors": 0,
                "mean_pos_distance": 0.0,
                "mean_neg_distance": 0.0,
            },
        )

    similarity = (
        embeddings
        @ embeddings.T
    )

    distance = (
        1.0
        - similarity
    )

    labels_col = (
        labels
        .view(-1, 1)
    )

    same_label = (
        labels_col
        == labels_col.T
    )

    different_label = (
        ~same_label
    )

    # Exclude self-comparison.
    identity = torch.eye(
        batch_size,
        dtype=torch.bool,
        device=(
            embeddings.device
        ),
    )

    positive_mask = (
        same_label
        & ~identity
    )

    negative_mask = (
        different_label
    )

    has_positive = (
        positive_mask.any(
            dim=1
        )
    )

    has_negative = (
        negative_mask.any(
            dim=1
        )
    )

    valid = (
        has_positive
        & has_negative
    )

    if not valid.any():
        zero = (
            embeddings.sum()
            * 0.0
        )

        return (
            zero,
            {
                "valid_anchors": 0,
                "mean_pos_distance": 0.0,
                "mean_neg_distance": 0.0,
            },
        )

    # Hardest positive =
    # largest same-class distance.
    positive_distances = (
        distance
        .masked_fill(
            ~positive_mask,
            -1e9,
        )
    )

    hardest_positive = (
        positive_distances
        .max(dim=1)
        .values
    )

    # Hardest negative =
    # smallest different-class distance.
    negative_distances = (
        distance
        .masked_fill(
            ~negative_mask,
            1e9,
        )
    )

    hardest_negative = (
        negative_distances
        .min(dim=1)
        .values
    )

    hp = (
        hardest_positive[
            valid
        ]
    )

    hn = (
        hardest_negative[
            valid
        ]
    )

    losses = F.relu(
        hp
        - hn
        + margin
    )

    loss = (
        losses.mean()
    )

    stats = {
        "valid_anchors": (
            int(
                valid
                .sum()
                .detach()
                .cpu()
            )
        ),

        "mean_pos_distance": (
            float(
                hp
                .mean()
                .detach()
                .cpu()
            )
        ),

        "mean_neg_distance": (
            float(
                hn
                .mean()
                .detach()
                .cpu()
            )
        ),
    }

    return (
        loss,
        stats,
    )


class RelationalContrastiveTrainer(
    Trainer
):
    def __init__(
        self,
        *args,
        contrast_weight=0.3,
        contrast_margin=0.2,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.contrast_weight = (
            contrast_weight
        )

        self.contrast_margin = (
            contrast_margin
        )

        self._contrast_steps = 0
        self._valid_anchor_sum = 0

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        # ---------------------------------------------------------
        # Evaluation
        # ---------------------------------------------------------
        # Contrastive representations are a TRAINING-ONLY signal.
        #
        # During evaluation we return the ordinary classification
        # output so HuggingFace Trainer receives only classifier
        # logits as predictions.
        # ---------------------------------------------------------
        if not model.training:
            outputs = model(
                **inputs,
                return_representation=False,
            )

            loss = outputs.loss

            if return_outputs:
                return loss, outputs

            return loss

        # ---------------------------------------------------------
        # Training
        # ---------------------------------------------------------
        labels = inputs["labels"]

        outputs = model(
            **inputs,
            return_representation=True,
        )

        ce_loss = outputs.loss
        embedding = outputs.contrast_embedding

        if embedding is None:
            raise RuntimeError(
                "Training requested contrastive learning, "
                "but contrast_embedding is None."
            )

        (
            contrast_loss,
            contrast_stats,
        ) = batch_hard_triplet_loss(
            embeddings=embedding,
            labels=labels,
            margin=self.contrast_margin,
        )

        loss = (
            ce_loss
            + self.contrast_weight
            * contrast_loss
        )

        self._contrast_steps += 1

        self._valid_anchor_sum += (
            contrast_stats["valid_anchors"]
        )

        if (
            self.state.global_step > 0
            and self.state.global_step % 100 == 0
        ):
            self.log(
                {
                    "ce_loss_aux": float(
                        ce_loss.detach().cpu()
                    ),
                    "contrast_loss": float(
                        contrast_loss.detach().cpu()
                    ),
                    "hard_pos_dist": (
                        contrast_stats[
                            "mean_pos_distance"
                        ]
                    ),
                    "hard_neg_dist": (
                        contrast_stats[
                            "mean_neg_distance"
                        ]
                    ),
                    "valid_triplet_anchors": (
                        contrast_stats[
                            "valid_anchors"
                        ]
                    ),
                }
            )

        if return_outputs:
            return loss, outputs

        return loss


def main():
    parser = (
        argparse
        .ArgumentParser()
    )

    parser.add_argument(
        "--domain",
        choices=[
            "Laptop",
            "Restaurants",
            "Twitter",
        ],
        default="Laptop",
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
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--relation-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--contrast-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--contrast-weight",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--contrast-margin",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = (
        parser
        .parse_args()
    )

    set_seed(
        args.seed
    )

    train_path = (
        args.data_root
        / args.domain
        / "train.csv"
    )

    test_path = (
        args.data_root
        / args.domain
        / "test.csv"
    )

    print(
        "=" * 80
    )

    print(
        "TARGET-RELATIONAL M3-A"
    )

    print(
        "Architecture: M1"
    )

    print(
        "Learning: CE + "
        "batch-hard relational contrast"
    )

    print(
        "Domain:",
        args.domain,
    )

    print(
        "Train:",
        train_path,
    )

    print(
        "Test:",
        test_path,
    )

    print(
        "Seed:",
        args.seed,
    )

    print(
        "Contrast weight:",
        args.contrast_weight,
    )

    print(
        "Contrast margin:",
        args.contrast_margin,
    )

    print(
        "Protocol: literature-style "
        "test selection"
    )

    print(
        "=" * 80
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.model,
            use_fast=True,
        )
    )

    train_dataset = (
        LiteratureABSADataset(
            train_path,
            tokenizer,
            args.max_length,
        )
    )

    test_dataset = (
        LiteratureABSADataset(
            test_path,
            tokenizer,
            args.max_length,
        )
    )

    print(
        "Train instances:",
        len(train_dataset),
    )

    print(
        "Test instances:",
        len(test_dataset),
    )

    sample = (
        train_dataset[0]
    )

    print(
        "\nSANITY CHECK"
    )

    print(
        "sequence length:",
        len(
            sample[
                "input_ids"
            ]
        ),
    )

    print(
        "sentence subwords:",
        int(
            sample[
                "sentence_mask"
            ].sum()
        ),
    )

    print(
        "target subwords:",
        int(
            sample[
                "target_mask"
            ].sum()
        ),
    )

    if args.smoke:
        train_dataset.data = (
            train_dataset
            .data
            .iloc[:128]
            .reset_index(
                drop=True
            )
        )

        test_dataset.data = (
            test_dataset
            .data
            .iloc[:128]
            .reset_index(
                drop=True
            )
        )

        args.epochs = 1

        print(
            "\nSMOKE MODE"
        )

        print(
            "Train:",
            len(
                train_dataset
            ),
        )

        print(
            "Test:",
            len(
                test_dataset
            ),
        )

    model = (
        TargetRelationalM3(
            backbone_name=(
                args.model
            ),
            num_labels=3,
            relation_dim=(
                args.relation_dim
            ),
            contrast_dim=(
                args.contrast_dim
            ),
            dropout=(
                args.dropout
            ),
        )
    )

    run_name = (
        f"{args.domain.lower()}"
        f"_seed{args.seed}"
        f"_cw{args.contrast_weight}"
        f"_cm{args.contrast_margin}"
    )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "target_relational_m3"
        / run_name
    )

    training_args = (
        TrainingArguments(
            output_dir=str(
                output_dir
            ),

            learning_rate=(
                args.lr
            ),

            per_device_train_batch_size=(
                args.batch_size
            ),

            per_device_eval_batch_size=(
                args.batch_size
                * 2
            ),

            num_train_epochs=(
                args.epochs
            ),

            weight_decay=0.01,

            eval_strategy="epoch",
            save_strategy="epoch",

            load_best_model_at_end=True,

            metric_for_best_model=(
                "macro_f1"
            ),

            greater_is_better=True,

            save_total_limit=2,

            logging_steps=25,

            report_to="none",

            seed=args.seed,
            data_seed=args.seed,

            fp16=(
                torch.cuda
                .is_available()
            ),

            remove_unused_columns=False,
        )
    )

    trainer = (
        RelationalContrastiveTrainer(
            model=model,
            args=training_args,

            train_dataset=(
                train_dataset
            ),

            eval_dataset=(
                test_dataset
            ),

            processing_class=(
                tokenizer
            ),

            compute_metrics=(
                compute_metrics
            ),

            contrast_weight=(
                args.contrast_weight
            ),

            contrast_margin=(
                args.contrast_margin
            ),
        )
    )

    trainer.train()

    print(
        "\nBEST TEST-SELECTED "
        "CHECKPOINT"
    )

    print(
        trainer
        .state
        .best_model_checkpoint
    )

    print(
        "Best metric:",
        trainer
        .state
        .best_metric,
    )

    test_results = (
        trainer.evaluate(
            test_dataset
        )
    )

    print(
        "\nFINAL TEST EVALUATION"
    )

    print(
        test_results
    )

    metrics_dir = (
        PROJECT_ROOT
        / "outputs"
        / "metrics"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        metrics_dir
        / (
            f"target_relational_m3_"
            f"{run_name}.json"
        )
    )

    payload = {
        "framework": (
            "target_relational_m3_a"
        ),

        "architecture": (
            "m1_target_relational"
        ),

        "training_objective": (
            "cross_entropy_plus_"
            "batch_hard_relational_triplet"
        ),

        "domain": (
            args.domain
        ),

        "model": (
            args.model
        ),

        "seed": (
            args.seed
        ),

        "relation_dim": (
            args.relation_dim
        ),

        "contrast_dim": (
            args.contrast_dim
        ),

        "contrast_weight": (
            args.contrast_weight
        ),

        "contrast_margin": (
            args.contrast_margin
        ),

        "epochs": (
            args.epochs
        ),

        "batch_size": (
            args.batch_size
        ),

        "learning_rate": (
            args.lr
        ),

        "best_checkpoint": (
            trainer
            .state
            .best_model_checkpoint
        ),

        "best_macro_f1": (
            trainer
            .state
            .best_metric
        ),

        "test": (
            test_results
        ),
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    print(
        "\nSaved:",
        metrics_path,
    )


if __name__ == "__main__":
    main()
