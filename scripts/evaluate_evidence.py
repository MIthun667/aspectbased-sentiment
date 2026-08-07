from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

from src.models.target_evidence import TargetEvidenceModel
from scripts.train_framework import TargetEvidenceDataset, LABEL2ID

ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_model_with_evidence(
    model: TargetEvidenceModel,
    dataset: TargetEvidenceDataset,
    records: list[dict],
    device: torch.device,
):
    model.eval()
    model.to(device)

    all_preds = []
    all_golds = []
    analyzed_records = []

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    sample_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            aspect_mask = batch["aspect_mask"].to(device)
            distance_ids = batch["distance_ids"].to(device)
            evidence_mask = batch["evidence_mask"].to(device)
            labels = batch["labels"].to(device)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                aspect_mask=aspect_mask,
                distance_ids=distance_ids,
                evidence_mask=evidence_mask,
            )

            logits = output.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            golds = labels.cpu().numpy()
            evidence_attns = output.evidence_attentions.cpu().numpy()

            all_preds.extend(preds)
            all_golds.extend(golds)

            # Process evidence attention alignment for each sample in batch
            batch_size = input_ids.size(0)
            for b in range(batch_size):
                rec = records[sample_idx]
                sample_idx += 1

                encoding = dataset.tokenizer(
                    rec["tokens"],
                    is_split_into_words=True,
                    truncation=True,
                    max_length=dataset.max_length,
                    padding="max_length",
                )
                word_ids = encoding.word_ids()

                subword_attns = evidence_attns[b]

                # Map subword attention back to word-level attention
                word_attn_dict = defaultdict(float)
                for wid, attn_val in zip(word_ids, subword_attns):
                    if wid is not None:
                        word_attn_dict[wid] += float(attn_val)

                # Format word-level evidence details
                word_evidence = []
                tokens = rec["tokens"]
                distances = rec["aspect_distances"]
                for wid, token in enumerate(tokens):
                    word_evidence.append({
                        "word": token,
                        "word_id": wid,
                        "attention": word_attn_dict.get(wid, 0.0),
                        "distance": distances[wid] if wid < len(distances) else -1,
                    })

                # Sort by attention weight descending
                word_evidence_sorted = sorted(
                    word_evidence, key=lambda x: x["attention"], reverse=True
                )

                analyzed_records.append({
                    "instance_id": rec["instance_id"],
                    "sentence_id": rec["sentence_id"],
                    "sentence": " ".join(rec["tokens"]),
                    "aspect_text": rec["aspect_text"],
                    "gold_polarity": rec["polarity"],
                    "pred_polarity": ID2LABEL[preds[b]],
                    "is_correct": bool(preds[b] == golds[b]),
                    "is_multi_aspect": rec.get("is_multi_aspect", False),
                    "number_of_aspects": rec.get("number_of_aspects", 1),
                    "top_evidence": word_evidence_sorted,
                })

    all_preds = np.array(all_preds)
    all_golds = np.array(all_golds)

    # Compute overall metrics
    overall_acc = accuracy_score(all_golds, all_preds)
    overall_f1 = f1_score(all_golds, all_preds, average="macro")

    # Compute subset metrics (single vs multi aspect)
    single_mask = np.array([not r["is_multi_aspect"] for r in analyzed_records])
    multi_mask = np.array([r["is_multi_aspect"] for r in analyzed_records])

    single_acc = accuracy_score(all_golds[single_mask], all_preds[single_mask]) if single_mask.any() else 0.0
    single_f1 = f1_score(all_golds[single_mask], all_preds[single_mask], average="macro") if single_mask.any() else 0.0

    multi_acc = accuracy_score(all_golds[multi_mask], all_preds[multi_mask]) if multi_mask.any() else 0.0
    multi_f1 = f1_score(all_golds[multi_mask], all_preds[multi_mask], average="macro") if multi_mask.any() else 0.0

    summary = {
        "overall": {"accuracy": overall_acc, "macro_f1": overall_f1, "count": len(analyzed_records)},
        "single_aspect": {"accuracy": single_acc, "macro_f1": single_f1, "count": int(single_mask.sum())},
        "multi_aspect": {"accuracy": multi_acc, "macro_f1": multi_f1, "count": int(multi_mask.sum())},
    }

    return summary, analyzed_records


def generate_human_readable_report(analyzed_records: list[dict], max_samples: int = 15) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("DIAGNOSTIC EVIDENCE ATTENTION ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    for i, item in enumerate(analyzed_records[:max_samples]):
        lines.append(f"Sample #{i+1} [{item['instance_id']}]")
        lines.append(f"Sentence:       \"{item['sentence']}\"")
        lines.append(f"Target:         {item['aspect_text']}")
        lines.append(f"Gold Polarity:  {item['gold_polarity']}")
        lines.append(f"Prediction:     {item['pred_polarity']} ({'CORRECT' if item['is_correct'] else 'INCORRECT'})")
        lines.append(f"Multi-Aspect:   {item['is_multi_aspect']} (total aspects: {item['number_of_aspects']})")
        lines.append("Top Evidence Tokens:")

        for ev in item["top_evidence"][:5]:
            lines.append(f"  - {ev['word']:<15} attn={ev['attention']:.4f}  distance={ev['distance']}")

        lines.append("-" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="laptops", choices=["laptops", "restaurants", "tweets"])
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--model-path", default=None, help="Path to checkpoint directory")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_path is None:
        ckpt_dir = PROJECT_ROOT / "outputs" / "target_evidence" / args.domain / f"seed_{args.seed}"
        state_file = list(ckpt_dir.glob("checkpoint-*/trainer_state.json"))
        if state_file:
            with state_file[0].open() as f:
                state = json.load(f)
            args.model_path = state.get("best_model_checkpoint", str(ckpt_dir / "checkpoint-726"))
        else:
            args.model_path = str(ckpt_dir)

    print(f"Loading checkpoint from: {args.model_path}")
    model_path = Path(args.model_path)
    model = TargetEvidenceModel(backbone_name="roberta-base")
    safetensors_path = model_path / "model.safetensors"
    bin_path = model_path / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {model_path}")
    model.load_state_dict(state_dict)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

    split_path = PROJECT_ROOT / "data" / "processed" / args.domain / f"{args.split}.jsonl"
    records = load_jsonl(split_path)

    dataset = TargetEvidenceDataset(
        records=records,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    summary, analyzed_records = evaluate_model_with_evidence(
        model=model,
        dataset=dataset,
        records=records,
        device=device,
    )

    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY ({args.domain.upper()} - {args.split.upper()})")
    print("=" * 80)
    print(json.dumps(summary, indent=2))

    report_text = generate_human_readable_report(analyzed_records)
    print("\n" + report_text)

    # Save output artifacts
    output_dir = PROJECT_ROOT / "outputs" / "evidence_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / f"{args.domain}_{args.split}_evidence_summary.json"
    report_file = output_dir / f"{args.domain}_{args.split}_evidence_report.txt"

    with summary_file.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": analyzed_records}, f, indent=2)

    with report_file.open("w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nSaved analysis summary to: {summary_file}")
    print(f"Saved human-readable report to: {report_file}")


if __name__ == "__main__":
    main()
