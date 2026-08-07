from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


class ABSCDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        sentence = " ".join(record["tokens"])
        aspect = " ".join(record["aspect_tokens"])

        encoded = self.tokenizer(
            sentence,
            aspect,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }

        item["labels"] = torch.tensor(
            LABEL2ID[record["polarity"]],
            dtype=torch.long,
        )

        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
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
        choices=["laptops", "restaurants", "tweets"],
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
        "--seed",
        type=int,
        default=2026,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    root = PROJECT_ROOT / "data" / "processed" / args.domain

    train_records = load_jsonl(root / "train.jsonl")
    validation_records = load_jsonl(root / "validation.jsonl")
    test_records = load_jsonl(root / "test.jsonl")

    print("=" * 80)
    print("Domain:", args.domain)
    print("Train:", len(train_records))
    print("Validation:", len(validation_records))
    print("Test:", len(test_records))
    print("Model:", args.model)
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = ABSCDataset(
        train_records,
        tokenizer,
        args.max_length,
    )

    validation_dataset = ABSCDataset(
        validation_records,
        tokenizer,
        args.max_length,
    )

    test_dataset = ABSCDataset(
        test_records,
        tokenizer,
        args.max_length,
    )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "baseline"
        / args.domain
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
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

    print("\nVALIDATION RESULTS")
    validation_results = trainer.evaluate(validation_dataset)
    print(validation_results)

    print("\nTEST RESULTS")
    test_results = trainer.evaluate(test_dataset)
    print(test_results)

    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / f"{args.domain}_baseline.json"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "domain": args.domain,
                "model": args.model,
                "seed": args.seed,
                "validation": validation_results,
                "test": test_results,
            },
            f,
            indent=2,
        )

    print("\nSaved:", metrics_path)


if __name__ == "__main__":
    main()
