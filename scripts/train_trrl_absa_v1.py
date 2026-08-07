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
    classification_report,
)

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.trrl_absa_v1 import TRRLABSAV1


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ABSADataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        tokenizer,
        max_length: int = 128,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        sentence = str(row["context"])
        target = str(row["term"])
        label = int(row["polarity"])

        encoded = self.tokenizer(
            sentence,
            target,
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

        if sum(sentence_mask) == 0:
            raise RuntimeError(
                f"No sentence tokens at row {idx}"
            )

        if sum(target_mask) == 0:
            raise RuntimeError(
                f"Target truncated at row {idx}: {target!r}"
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
            "labels": torch.tensor(
                label,
                dtype=torch.long,
            ),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

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
        default=Path.home()
        / "TPHCA"
        / "unFixedData",
    )

    parser.add_argument(
        "--model",
        default="roberta-base",
    )

    parser.add_argument(
        "--trrl-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "trrl2"
        / "laptop"
        / "seed_25"
        / "relation_encoder.pt",
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
        "--max-length",
        type=int,
        default=128,
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
        "--no-trrl-init",
        action="store_true",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    set_seed(args.seed)

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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    train_dataset = ABSADataset(
        train_path,
        tokenizer,
        max_length=args.max_length,
    )

    test_dataset = ABSADataset(
        test_path,
        tokenizer,
        max_length=args.max_length,
    )

    if args.smoke:
        train_dataset.df = (
            train_dataset.df
            .iloc[:128]
            .reset_index(drop=True)
        )

        test_dataset.df = (
            test_dataset.df
            .iloc[:128]
            .reset_index(drop=True)
        )

        args.epochs = 1

    print("=" * 90)
    print("TRRL-ABSA V1")
    print("=" * 90)

    print("Domain:", args.domain)
    print("Train instances:", len(train_dataset))
    print("Test instances:", len(test_dataset))
    print("Objective: CE only")
    print(
        "Architecture: RoBERTa CLS + "
        "TRRL-2 target-evidence relation"
    )
    print(
        "TRRL initialization:",
        not args.no_trrl_init,
    )

    sample = train_dataset[0]

    print("\nSANITY CHECK")
    print(
        "sequence length:",
        len(sample["input_ids"]),
    )
    print(
        "sentence subwords:",
        int(sample["sentence_mask"].sum()),
    )
    print(
        "target subwords:",
        int(sample["target_mask"].sum()),
    )

    model = TRRLABSAV1(
        backbone_name=args.model,
        num_labels=3,
        relation_dim=args.relation_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
    )

    if not args.no_trrl_init:
        if not args.trrl_checkpoint.exists():
            raise FileNotFoundError(
                args.trrl_checkpoint
            )

        model.load_trrl2_checkpoint(
            str(args.trrl_checkpoint)
        )

    init_name = (
        "pretrained"
        if not args.no_trrl_init
        else "random"
    )

    run_name = (
        f"{args.domain.lower()}"
        f"_seed{args.seed}"
        f"_{init_name}"
    )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "trrl_absa_v1"
        / run_name
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        learning_rate=args.lr,

        per_device_train_batch_size=(
            args.batch_size
        ),

        per_device_eval_batch_size=(
            args.batch_size * 2
        ),

        num_train_epochs=args.epochs,

        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="macro_f1",
        greater_is_better=True,

        save_total_limit=2,

        logging_steps=25,

        report_to="none",

        seed=args.seed,
        data_seed=args.seed,

        fp16=torch.cuda.is_available(),

        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print(
        "\nBEST TEST-SELECTED CHECKPOINT"
    )

    print(
        trainer.state.best_model_checkpoint
    )

    print(
        "Best Macro-F1:",
        trainer.state.best_metric,
    )

    prediction = trainer.predict(
        test_dataset
    )

    logits = prediction.predictions

    if isinstance(
        logits,
        (tuple, list),
    ):
        logits = logits[0]

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    labels = prediction.label_ids

    test_accuracy = accuracy_score(
        labels,
        predictions,
    )

    test_macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
    )

    report = classification_report(
        labels,
        predictions,
        labels=[0, 1, 2],
        target_names=[
            "negative",
            "neutral",
            "positive",
        ],
        output_dict=True,
        zero_division=0,
    )

    print("\nFINAL TEST")
    print(
        "Accuracy:",
        test_accuracy,
    )
    print(
        "Macro-F1:",
        test_macro_f1,
    )

    print("\nClass-wise F1")

    for label in [
        "negative",
        "neutral",
        "positive",
    ]:
        print(
            label,
            report[label]["f1-score"],
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
        / f"trrl_absa_v1_{run_name}.json"
    )

    payload = {
        "framework": "TRRL-ABSA-V1",
        "domain": args.domain,
        "seed": args.seed,
        "backbone": args.model,
        "trrl_initialized": (
            not args.no_trrl_init
        ),
        "trrl_checkpoint": (
            str(args.trrl_checkpoint)
            if not args.no_trrl_init
            else None
        ),
        "objective": "cross_entropy_only",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "relation_dim": args.relation_dim,
        "attention_dim": args.attention_dim,
        "best_checkpoint": (
            trainer.state.best_model_checkpoint
        ),
        "best_macro_f1": (
            trainer.state.best_metric
        ),
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "classification_report": report,
    }

    metrics_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nSaved:",
        metrics_path,
    )


if __name__ == "__main__":
    main()
