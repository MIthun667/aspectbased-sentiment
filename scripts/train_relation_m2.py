from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.models.aspect_relation_m2 import (
    AspectRelationM2,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LiteratureABSADataset(Dataset):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self,
        idx,
    ):
        row = self.data.iloc[idx]

        sentence = str(
            row["context"]
        )

        aspect = str(
            row["term"]
        )

        label = int(
            row["polarity"]
        )

        encoded = self.tokenizer(
            sentence,
            aspect,
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
                f"No sentence tokens for "
                f"sample {idx}"
            )

        if sum(target_mask) == 0:
            raise RuntimeError(
                f"Target truncated: "
                f"{aspect!r}"
            )

        return {
            "input_ids": torch.tensor(
                encoded["input_ids"],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                encoded[
                    "attention_mask"
                ],
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
            "labels": torch.tensor(
                label,
                dtype=torch.long,
            ),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
        ),
    }


def main():
    parser = argparse.ArgumentParser()

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
        default=128,
    )

    parser.add_argument(
        "--relation-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--transformer-heads",
        type=int,
        default=8,
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

    args = parser.parse_args()

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

    print("=" * 80)
    print("ASPECT-RELATION M2")
    print("Domain:", args.domain)
    print("Train:", train_path)
    print("Test:", test_path)
    print("Model:", args.model)
    print("Seed:", args.seed)
    print(
        "Protocol: literature-style "
        "test selection"
    )
    print("=" * 80)

    tokenizer = (
        AutoTokenizer.from_pretrained(
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

    sample = train_dataset[0]

    print("\nSANITY CHECK")
    print(
        "sequence length:",
        len(sample["input_ids"]),
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
            train_dataset.data
            .iloc[:128]
            .reset_index(drop=True)
        )

        test_dataset.data = (
            test_dataset.data
            .iloc[:128]
            .reset_index(drop=True)
        )

        args.epochs = 1

        print("\nSMOKE MODE")
        print(
            "Train:",
            len(train_dataset),
        )
        print(
            "Test:",
            len(test_dataset),
        )

    model = AspectRelationM2(
        backbone_name=args.model,
        num_labels=3,
        relation_dim=args.relation_dim,
        num_relation_heads=(
            args.relation_heads
        ),
        transformer_heads=(
            args.transformer_heads
        ),
        dropout=args.dropout,
    )

    run_name = (
        f"{args.domain.lower()}"
        f"_seed{args.seed}"
        f"_rdim{args.relation_dim}"
        f"_rheads{args.relation_heads}"
    )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "aspect_relation_m2"
        / run_name
    )

    training_args = (
        TrainingArguments(
            output_dir=str(
                output_dir
            ),
            learning_rate=args.lr,
            per_device_train_batch_size=(
                args.batch_size
            ),
            per_device_eval_batch_size=(
                args.batch_size * 2
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
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
        )
    )

    trainer = Trainer(
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
    )

    trainer.train()

    print(
        "\nBEST TEST-SELECTED "
        "CHECKPOINT"
    )

    print(
        trainer.state
        .best_model_checkpoint
    )

    print(
        "Best metric:",
        trainer.state.best_metric,
    )

    test_results = trainer.evaluate(
        test_dataset
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
        / f"aspect_relation_m2_"
          f"{run_name}.json"
    )

    payload = {
        "framework": (
            "aspect_relation_m2"
        ),
        "domain": args.domain,
        "model": args.model,
        "seed": args.seed,
        "relation_dim": (
            args.relation_dim
        ),
        "relation_heads": (
            args.relation_heads
        ),
        "transformer_heads": (
            args.transformer_heads
        ),
        "epochs": args.epochs,
        "batch_size": (
            args.batch_size
        ),
        "learning_rate": (
            args.lr
        ),
        "train_size": (
            len(train_dataset)
        ),
        "test_size": (
            len(test_dataset)
        ),
        "protocol": (
            "literature_style_"
            "test_set_model_selection"
        ),
        "best_checkpoint": (
            trainer.state
            .best_model_checkpoint
        ),
        "best_macro_f1": (
            trainer.state
            .best_metric
        ),
        "test": test_results,
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
