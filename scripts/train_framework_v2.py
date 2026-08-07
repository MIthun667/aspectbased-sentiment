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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from src.models.target_evidence_v2 import TargetEvidenceV2Model
from src.utils.provenance import load_best_checkpoint_path, save_experiment_metadata

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


class TargetEvidenceV2Dataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        max_dependency_distance: int = 10,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_dependency_distance = max_dependency_distance

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        sentence_words = rec["tokens"]
        aspect_words = rec["aspect_tokens"]

        encoding = self.tokenizer(
            sentence_words,
            aspect_words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
        )

        sequence_ids = encoding.sequence_ids()
        word_ids = encoding.word_ids()

        aspect_start = rec["aspect_start"]
        aspect_end = rec["aspect_end"]
        original_distances = rec["aspect_distances"]

        aspect_mask = []
        distance_ids = []
        evidence_mask = []

        special_distance = self.max_dependency_distance + 1

        for seq_id, word_id in zip(sequence_ids, word_ids):
            # Sentence subwords: sequence_id == 0 and word_id is not None
            if seq_id == 0 and word_id is not None:
                is_target = (aspect_start <= word_id < aspect_end)
                aspect_mask.append(1 if is_target else 0)

                dist = original_distances[word_id] if word_id < len(original_distances) else special_distance
                dist = min(int(dist), self.max_dependency_distance)
                distance_ids.append(dist)

                evidence_mask.append(1)  # Only sentence subwords are valid evidence
            else:
                aspect_mask.append(0)
                distance_ids.append(special_distance)
                evidence_mask.append(0)  # Exclude special, padding, and sequence 1 aspect tokens

        item = {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "aspect_mask": torch.tensor(aspect_mask, dtype=torch.long),
            "distance_ids": torch.tensor(distance_ids, dtype=torch.long),
            "evidence_mask": torch.tensor(evidence_mask, dtype=torch.long),
            "labels": torch.tensor(LABEL2ID[rec["polarity"]], dtype=torch.long),
        }

        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    }


def compute_gate_statistics(
    model: TargetEvidenceV2Model,
    dataset: TargetEvidenceV2Dataset,
    device: torch.device,
) -> dict:
    model.eval()
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    all_gates = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            aspect_mask = batch["aspect_mask"].to(device)
            distance_ids = batch["distance_ids"].to(device)
            evidence_mask = batch["evidence_mask"].to(device)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                aspect_mask=aspect_mask,
                distance_ids=distance_ids,
                evidence_mask=evidence_mask,
            )

            gates = output.gate_values.cpu().numpy()  # [B, H]
            mean_sample_gate = np.mean(gates, axis=-1)  # average over hidden dim per sample
            all_gates.extend(mean_sample_gate)

    all_gates = np.array(all_gates)
    return {
        "mean_gate": float(np.mean(all_gates)),
        "std_gate": float(np.std(all_gates)),
        "min_gate": float(np.min(all_gates)),
        "max_gate": float(np.max(all_gates)),
        "median_gate": float(np.median(all_gates)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="laptops", choices=["laptops", "restaurants", "tweets"])
    parser.add_argument("--exp-name", default="tev2a", help="Experiment directory name under outputs/")
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-dependency-distance", type=int, default=10)
    parser.add_argument("--no-distance", action="store_true", help="Ablation: disable dependency distance embeddings")
    parser.add_argument("--use-evidence-supervision", action="store_true", help="Enable self-supervised evidence loss L_EVID (L_SUPP + L_TDC)")
    parser.add_argument("--lambda-evid", type=float, default=0.1, help="Coefficient for evidence loss L_EVID")
    parser.add_argument("--gamma-tdc", type=float, default=0.4, help="Margin for Target-Disambiguation Contrastive Loss")
    parser.add_argument("--lambda-supp", type=float, default=1.0, help="Weight for aspect suppression loss L_SUPP")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    set_seed(args.seed)

    domain_root = PROJECT_ROOT / "data" / "processed" / args.domain
    train_records = load_jsonl(domain_root / "train.jsonl")
    val_records = load_jsonl(domain_root / "validation.jsonl")
    test_records = load_jsonl(domain_root / "test.jsonl")

    print("=" * 80)
    print("TARGET-EVIDENCE V2-A TRAINING")
    print("=" * 80)
    print("Experiment:", args.exp_name)
    print("Domain:", args.domain)
    print("Backbone:", args.model)
    print("Train:", len(train_records))
    print("Validation:", len(val_records))
    print("Test:", len(test_records))
    print("Use Distance Embeddings:", not args.no_distance)
    print("Use Evidence Supervision:", args.use_evidence_supervision)
    if args.use_evidence_supervision:
        print(f"  - lambda_evid: {args.lambda_evid}, gamma_tdc: {args.gamma_tdc}, lambda_supp: {args.lambda_supp}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)

    train_dataset = TargetEvidenceV2Dataset(train_records, tokenizer, args.max_length, args.max_dependency_distance)
    val_dataset = TargetEvidenceV2Dataset(val_records, tokenizer, args.max_length, args.max_dependency_distance)
    test_dataset = TargetEvidenceV2Dataset(test_records, tokenizer, args.max_length, args.max_dependency_distance)

    # Sanity check target mask
    sample = train_dataset[0]
    if sample["aspect_mask"].sum().item() == 0:
        raise RuntimeError("Sanity check failed: Aspect span eliminated during tokenization.")

    model = TargetEvidenceV2Model(
        backbone_name=args.model,
        num_labels=3,
        max_dependency_distance=args.max_dependency_distance,
        use_distance_embedding=(not args.no_distance),
        use_evidence_supervision=args.use_evidence_supervision,
        lambda_evid=args.lambda_evid,
        gamma_tdc=args.gamma_tdc,
        lambda_supp=args.lambda_supp,
    )


    output_dir = PROJECT_ROOT / "outputs" / args.exp_name / args.domain / f"seed_{args.seed}"

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
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\nStarting Training...")
    trainer.train()

    # Resolve best checkpoint explicitly
    best_checkpoint = load_best_checkpoint_path(output_dir)
    print(f"\nResolved Best Validation Checkpoint: {best_checkpoint}")

    # Evaluate best model on Validation
    val_results = trainer.evaluate(val_dataset)
    print("\nBest Validation Results:")
    print(val_results)

    # Evaluate best model on Test (Only once at the end)
    test_results = trainer.evaluate(test_dataset)
    print("\nTest Results (Best Validation Checkpoint):")
    print(test_results)

    # Compute Gate Statistics on Validation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = TargetEvidenceV2Model.from_pretrained if hasattr(model, "from_pretrained") else model
    gate_stats = compute_gate_statistics(best_model, val_dataset, device)
    print("\nValidation Gate Statistics:", gate_stats)

    # Save Provenance Metadata
    hyperparams = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "max_dependency_distance": args.max_dependency_distance,
        "use_distance_embedding": not args.no_distance,
        "weight_decay": 0.01,
        "gate_statistics_val": gate_stats,
    }

    meta_file = save_experiment_metadata(
        output_dir=output_dir,
        experiment_name=args.exp_name,
        model_name="TargetEvidenceV2Model",
        domain=args.domain,
        seed=args.seed,
        hyperparameters=hyperparams,
        best_checkpoint=str(best_checkpoint),
        best_validation_metric=val_results.get("eval_macro_f1", 0.0),
        best_epoch=val_results.get("epoch", 0.0),
        validation_metrics=val_results,
        test_metrics=test_results,
    )

    print("\nSaved Experiment Provenance Metadata to:", meta_file)


if __name__ == "__main__":
    main()
