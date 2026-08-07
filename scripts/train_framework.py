from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.models.target_evidence import TargetEvidenceModel


LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


class TargetEvidenceDataset(Dataset):

    def __init__(
        self,
        records,
        tokenizer,
        max_length=128,
        max_dependency_distance=10,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_dependency_distance = max_dependency_distance

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        record = self.records[idx]

        words = record["tokens"]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
        )

        word_ids = encoding.word_ids()

        aspect_start = record["aspect_start"]
        aspect_end = record["aspect_end"]

        original_distances = record["aspect_distances"]

        aspect_mask = []
        distance_ids = []
        evidence_mask = []

        # Use an extra ID for non-word/special tokens.
        special_distance = (
            self.max_dependency_distance + 1
        )

        for word_id in word_ids:

            if word_id is None:

                aspect_mask.append(0)
                distance_ids.append(special_distance)
                evidence_mask.append(0)

                continue

            # Target span membership
            is_target = (
                aspect_start
                <= word_id
                < aspect_end
            )

            aspect_mask.append(
                1 if is_target else 0
            )

            distance = original_distances[word_id]

            distance = min(
                int(distance),
                self.max_dependency_distance,
            )

            distance_ids.append(distance)

            # All real sentence tokens may serve as evidence.
            evidence_mask.append(1)

        item = {
            "input_ids": torch.tensor(
                encoding["input_ids"],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                encoding["attention_mask"],
                dtype=torch.long,
            ),
            "aspect_mask": torch.tensor(
                aspect_mask,
                dtype=torch.long,
            ),
            "distance_ids": torch.tensor(
                distance_ids,
                dtype=torch.long,
            ),
            "evidence_mask": torch.tensor(
                evidence_mask,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                LABEL2ID[record["polarity"]],
                dtype=torch.long,
            ),
        }

        return item


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--domain",
        choices=[
            "laptops",
            "restaurants",
            "tweets",
        ],
        required=True,
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
        "--lr",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--max-dependency-distance",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    domain_root = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / args.domain
    )

    train_records = load_jsonl(
        domain_root / "train.jsonl"
    )

    validation_records = load_jsonl(
        domain_root / "validation.jsonl"
    )

    test_records = load_jsonl(
        domain_root / "test.jsonl"
    )

    print("=" * 80)
    print("TARGET-EVIDENCE FRAMEWORK")
    print("=" * 80)
    print("Domain:", args.domain)
    print("Backbone:", args.model)
    print("Train:", len(train_records))
    print("Validation:", len(validation_records))
    print("Test:", len(test_records))
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        add_prefix_space=True,
    )

    train_dataset = TargetEvidenceDataset(
        train_records,
        tokenizer,
        args.max_length,
        args.max_dependency_distance,
    )

    validation_dataset = TargetEvidenceDataset(
        validation_records,
        tokenizer,
        args.max_length,
        args.max_dependency_distance,
    )

    test_dataset = TargetEvidenceDataset(
        test_records,
        tokenizer,
        args.max_length,
        args.max_dependency_distance,
    )

    # Sanity check before expensive training
    sample = train_dataset[0]

    print("\nSANITY CHECK")
    print(
        "input length:",
        len(sample["input_ids"]),
    )
    print(
        "target subwords:",
        int(sample["aspect_mask"].sum()),
    )
    print(
        "evidence subwords:",
        int(sample["evidence_mask"].sum()),
    )

    if sample["aspect_mask"].sum().item() == 0:
        raise RuntimeError(
            "Aspect span vanished during tokenization."
        )

    model = TargetEvidenceModel(
        backbone_name=args.model,
        num_labels=3,
        max_dependency_distance=(
            args.max_dependency_distance
        ),
    )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "target_evidence"
        / args.domain
        / f"seed_{args.seed}"
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n" + "=" * 80)
    print("BEST VALIDATION")
    print("=" * 80)

    validation_results = trainer.evaluate(
        validation_dataset
    )

    print(validation_results)

    print("\n" + "=" * 80)
    print("TEST")
    print("=" * 80)

    test_results = trainer.evaluate(
        test_dataset
    )

    print(test_results)

    metrics_dir = (
        PROJECT_ROOT
        / "outputs"
        / "metrics"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        metrics_dir
        / f"{args.domain}_target_evidence_seed{args.seed}.json"
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "framework": "target_evidence",
                "domain": args.domain,
                "backbone": args.model,
                "seed": args.seed,
                "validation": validation_results,
                "test": test_results,
            },
            f,
            indent=2,
        )

    print("\nSaved:", result_path)


if __name__ == "__main__":
    main()
