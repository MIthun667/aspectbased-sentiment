from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

from src.models.target_evidence_v2 import TargetEvidenceV2Model
from scripts.train_framework_v2 import TargetEvidenceV2Dataset, LABEL2ID, compute_gate_statistics
from src.utils.provenance import load_best_checkpoint_path

ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_model_subgroups(
    model: TargetEvidenceV2Model,
    tokenizer: AutoTokenizer,
    records: list[dict],
    device: torch.device,
    max_length: int = 128,
    use_distance: bool = True,
):
    model.eval()
    model.to(device)

    dataset = TargetEvidenceV2Dataset(
        records=records,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    all_preds = []
    all_golds = []
    all_gates = []
    is_multi_list = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
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

            preds = torch.argmax(output.logits, dim=-1).cpu().numpy()
            golds = labels.cpu().numpy()
            gates = output.gate_values.cpu().numpy()  # [B, H]
            mean_sample_gates = np.mean(gates, axis=-1)

            all_preds.extend(preds)
            all_golds.extend(golds)
            all_gates.extend(mean_sample_gates)

            start_idx = i * 16
            for b in range(len(preds)):
                rec = records[start_idx + b]
                is_multi_list.append(rec["is_multi_aspect"])

    all_preds = np.array(all_preds)
    all_golds = np.array(all_golds)
    all_gates = np.array(all_gates)
    is_multi_list = np.array(is_multi_list)

    single_mask = ~is_multi_list
    multi_mask = is_multi_list

    single_acc = float(accuracy_score(all_golds[single_mask], all_preds[single_mask]))
    single_f1 = float(f1_score(all_golds[single_mask], all_preds[single_mask], average="macro"))

    multi_acc = float(accuracy_score(all_golds[multi_mask], all_preds[multi_mask]))
    multi_f1 = float(f1_score(all_golds[multi_mask], all_preds[multi_mask], average="macro"))

    return {
        "overall": {
            "accuracy": float(accuracy_score(all_golds, all_preds)),
            "macro_f1": float(f1_score(all_golds, all_preds, average="macro")),
        },
        "single_aspect": {
            "count": int(single_mask.sum()),
            "accuracy": single_acc,
            "macro_f1": single_f1,
            "mean_gate": float(np.mean(all_gates[single_mask])),
        },
        "multi_aspect": {
            "count": int(multi_mask.sum()),
            "accuracy": multi_acc,
            "macro_f1": multi_f1,
            "mean_gate": float(np.mean(all_gates[multi_mask])),
        },
        "gate_summary": {
            "mean": float(np.mean(all_gates)),
            "std": float(np.std(all_gates)),
            "min": float(np.min(all_gates)),
            "max": float(np.max(all_gates)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="laptops")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

    test_records = load_jsonl(PROJECT_ROOT / "data" / "processed" / args.domain / "test.jsonl")

    # Evaluate tev2a
    tev2a_dir = PROJECT_ROOT / "outputs" / "tev2a" / args.domain / f"seed_{args.seed}"
    best_v2a_ckpt = load_best_checkpoint_path(tev2a_dir)
    v2a_model = TargetEvidenceV2Model(backbone_name="roberta-base", use_distance_embedding=True)
    v2a_model.load_state_dict(torch.load(best_v2a_ckpt / "model.safetensors" if (best_v2a_ckpt / "model.safetensors").exists() else best_v2a_ckpt / "pytorch_model.bin"))
    v2a_subgroups = evaluate_model_subgroups(v2a_model, tokenizer, test_records, device, use_distance=True)

    # Evaluate tev2a_no_dist ablation
    ablation_dir = PROJECT_ROOT / "outputs" / "tev2a_no_dist" / args.domain / f"seed_{args.seed}"
    best_abl_ckpt = load_best_checkpoint_path(ablation_dir)
    abl_model = TargetEvidenceV2Model(backbone_name="roberta-base", use_distance_embedding=False)
    abl_model.load_state_dict(torch.load(best_abl_ckpt / "model.safetensors" if (best_abl_ckpt / "model.safetensors").exists() else best_abl_ckpt / "pytorch_model.bin"))
    abl_subgroups = evaluate_model_subgroups(abl_model, tokenizer, test_records, device, use_distance=False)

    summary = {
        "tev2a": v2a_subgroups,
        "tev2a_no_dist": abl_subgroups,
    }

    out_file = PROJECT_ROOT / "outputs" / "tev2a" / args.domain / f"seed_{args.seed}" / "subgroup_eval.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
