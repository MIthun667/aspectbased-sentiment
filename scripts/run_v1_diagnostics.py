from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from safetensors.torch import load_file

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


def load_target_evidence_checkpoint(checkpoint_dir: Path) -> TargetEvidenceModel:
    model = TargetEvidenceModel(backbone_name="roberta-base")
    safetensors_path = checkpoint_dir / "model.safetensors"
    bin_path = checkpoint_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        state_dict = load_file(safetensors_path)
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model file found in {checkpoint_dir}")
    
    model.load_state_dict(state_dict)
    return model


def load_baseline_checkpoint(checkpoint_dir: Path) -> AutoModelForSequenceClassification:
    return AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)


def compute_baseline_predictions(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    records: list[dict],
    device: torch.device,
    max_length: int = 128,
) -> list[int]:
    model.eval()
    model.to(device)
    preds = []
    
    for rec in records:
        sentence = " ".join(rec["tokens"])
        aspect = " ".join(rec["aspect_tokens"])
        
        encoded = tokenizer(
            sentence,
            aspect,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()
            preds.append(pred)
            
    return preds


def compute_v1_predictions_and_evidence(
    model: TargetEvidenceModel,
    tokenizer: AutoTokenizer,
    records: list[dict],
    device: torch.device,
    max_length: int = 128,
):
    model.eval()
    model.to(device)

    dataset = TargetEvidenceDataset(
        records=records,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    sample_idx = 0
    all_results = []

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

            logits = output.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            evidence_attns = output.evidence_attentions.cpu().numpy()

            batch_size = input_ids.size(0)
            for b in range(batch_size):
                rec = records[sample_idx]
                sample_idx += 1

                encoding = tokenizer(
                    rec["tokens"],
                    is_split_into_words=True,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                )
                word_ids = encoding.word_ids()

                subword_attns = evidence_attns[b]

                # Map subword attention back to original word tokens
                word_attn_map = defaultdict(float)
                for wid, attn in zip(word_ids, subword_attns):
                    if wid is not None:
                        word_attn_map[wid] += float(attn)

                tokens = rec["tokens"]
                distances = rec["aspect_distances"]
                n_words = len(tokens)

                raw_weights = np.array([word_attn_map.get(i, 0.0) for i in range(n_words)])
                weight_sum = raw_weights.sum()
                normalized_weights = raw_weights / weight_sum if weight_sum > 0 else np.ones(n_words) / n_words

                word_evidence_details = []
                for i in range(n_words):
                    word_evidence_details.append({
                        "token": tokens[i],
                        "word_id": i,
                        "attention": float(normalized_weights[i]),
                        "distance": distances[i] if i < len(distances) else -1,
                    })

                all_results.append({
                    "record": rec,
                    "pred_label": ID2LABEL[preds[b]],
                    "pred_id": int(preds[b]),
                    "gold_label": rec["polarity"],
                    "gold_id": LABEL2ID[rec["polarity"]],
                    "is_correct": bool(preds[b] == LABEL2ID[rec["polarity"]]),
                    "word_weights": normalized_weights,
                    "word_details": word_evidence_details,
                })

    return all_results


def run_analysis_1_single_vs_multi(v1_results: list[dict]):
    single_res = [r for r in v1_results if not r["record"]["is_multi_aspect"]]
    multi_res = [r for r in v1_results if r["record"]["is_multi_aspect"]]

    def summarize_subset(subset: list[dict]):
        golds = [r["gold_id"] for r in subset]
        preds = [r["pred_id"] for r in subset]
        acc = accuracy_score(golds, preds) if golds else 0.0
        f1 = f1_score(golds, preds, average="macro") if golds else 0.0
        gold_dist = dict(Counter([r["gold_label"] for r in subset]))
        pred_dist = dict(Counter([r["pred_label"] for r in subset]))
        return {
            "count": len(subset),
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "gold_class_distribution": gold_dist,
            "pred_class_distribution": pred_dist,
        }

    overall_golds = [r["gold_id"] for r in v1_results]
    overall_preds = [r["pred_id"] for r in v1_results]

    return {
        "overall": {
            "count": len(v1_results),
            "accuracy": float(accuracy_score(overall_golds, overall_preds)),
            "macro_f1": float(f1_score(overall_golds, overall_preds, average="macro")),
        },
        "single_aspect": summarize_subset(single_res),
        "multi_aspect": summarize_subset(multi_res),
    }


def compute_entropy(probs: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    return float(-np.sum(p * np.log2(p)))


def run_analysis_2_evidence_quality(v1_results: list[dict]):
    single_correct = [r for r in v1_results if not r["record"]["is_multi_aspect"] and r["is_correct"]]
    single_incorrect = [r for r in v1_results if not r["record"]["is_multi_aspect"] and not r["is_correct"]]
    multi_correct = [r for r in v1_results if r["record"]["is_multi_aspect"] and r["is_correct"]]
    multi_incorrect = [r for r in v1_results if r["record"]["is_multi_aspect"] and not r["is_correct"]]

    categories = {
        "correct_single_aspect": single_correct[:20],
        "incorrect_single_aspect": single_incorrect[:20],
        "correct_multi_aspect": multi_correct[:20],
        "incorrect_multi_aspect": multi_incorrect[:20],
    }

    metrics_by_category = {}

    for cat_name, samples in categories.items():
        if not samples:
            continue

        aspect_masses = []
        mass_d1 = []
        mass_d2 = []
        mass_d3 = []
        entropies = []
        top1_tokens = Counter()
        top3_tokens = Counter()

        for item in samples:
            weights = item["word_weights"]
            distances = np.array([w["distance"] for w in item["word_details"]])
            tokens = [w["token"] for w in item["word_details"]]

            aspect_masses.append(float(weights[distances == 0].sum()))
            mass_d1.append(float(weights[distances <= 1].sum()))
            mass_d2.append(float(weights[distances <= 2].sum()))
            mass_d3.append(float(weights[distances <= 3].sum()))
            entropies.append(compute_entropy(weights))

            sorted_idx = np.argsort(weights)[::-1]
            top1_tokens[tokens[sorted_idx[0]].lower()] += 1
            for idx in sorted_idx[:3]:
                top3_tokens[tokens[idx].lower()] += 1

        metrics_by_category[cat_name] = {
            "num_samples_analyzed": len(samples),
            "avg_aspect_mass_d0": float(np.mean(aspect_masses)),
            "avg_mass_d_le_1": float(np.mean(mass_d1)),
            "avg_mass_d_le_2": float(np.mean(mass_d2)),
            "avg_mass_d_le_3": float(np.mean(mass_d3)),
            "avg_attention_entropy": float(np.mean(entropies)),
            "top_1_tokens": dict(top1_tokens.most_common(5)),
            "top_3_tokens": dict(top3_tokens.most_common(8)),
        }

    return metrics_by_category


def run_analysis_3_target_specificity(v1_results: list[dict]):
    sentence_map = defaultdict(list)
    for res in v1_results:
        sentence_map[res["record"]["sentence_id"]].append(res)

    multi_target_sentences = {sid: items for sid, items in sentence_map.items() if len(items) > 1}

    tvd_list = []
    jsd_list = []
    examples = []

    for sid, items in multi_target_sentences.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                item_a = items[i]
                item_b = items[j]

                w_a = item_a["word_weights"]
                w_b = item_b["word_weights"]

                # Total Variation Distance: 0.5 * sum(|P - Q|)
                tvd = float(0.5 * np.sum(np.abs(w_a - w_b)))
                
                # Jensen-Shannon Divergence using scipy (returns JS distance, square it for JSD)
                js_dist = jensenshannon(w_a, w_b, base=2)
                jsd = float(js_dist ** 2) if not np.isnan(js_dist) else 0.0

                tvd_list.append(tvd)
                jsd_list.append(jsd)

                if len(examples) < 10:
                    examples.append({
                        "sentence_id": sid,
                        "sentence": " ".join(item_a["record"]["tokens"]),
                        "target_a": item_a["record"]["aspect_text"],
                        "gold_a": item_a["gold_label"],
                        "pred_a": item_a["pred_label"],
                        "target_b": item_b["record"]["aspect_text"],
                        "gold_b": item_b["gold_label"],
                        "pred_b": item_b["pred_label"],
                        "tvd": tvd,
                        "jsd": jsd,
                        "top_evidence_a": [w["token"] for w in sorted(item_a["word_details"], key=lambda x: x["attention"], reverse=True)[:3]],
                        "top_evidence_b": [w["token"] for w in sorted(item_b["word_details"], key=lambda x: x["attention"], reverse=True)[:3]],
                    })

    return {
        "num_multi_target_sentences": len(multi_target_sentences),
        "num_target_pairs_evaluated": len(tvd_list),
        "mean_tvd": float(np.mean(tvd_list)) if tvd_list else 0.0,
        "std_tvd": float(np.std(tvd_list)) if tvd_list else 0.0,
        "mean_jsd": float(np.mean(jsd_list)) if jsd_list else 0.0,
        "std_jsd": float(np.std(jsd_list)) if jsd_list else 0.0,
        "pair_examples": examples,
    }


def run_analysis_4_distractor_behavior(v1_results: list[dict]):
    sentence_map = defaultdict(list)
    for res in v1_results:
        sentence_map[res["record"]["sentence_id"]].append(res)

    distractor_cases = []
    total_multi_target_instances = 0

    for sid, items in sentence_map.items():
        if len(items) <= 1:
            continue

        for current_item in items:
            total_multi_target_instances += 1
            current_rec = current_item["record"]
            current_start = current_rec["aspect_start"]
            current_end = current_rec["aspect_end"]
            
            # Other aspect spans in the same sentence
            other_aspect_ranges = [
                (other["record"]["aspect_start"], other["record"]["aspect_end"], other["record"]["aspect_text"])
                for other in items if other["record"]["instance_id"] != current_rec["instance_id"]
            ]

            weights = current_item["word_weights"]
            top1_idx = int(np.argmax(weights))
            top1_token = current_rec["tokens"][top1_idx]
            dist_to_current = current_rec["aspect_distances"][top1_idx]

            # Check if top1 token is inside or structurally closer to another aspect
            closer_to_other = False
            attracted_other_aspect = None

            for o_start, o_end, o_text in other_aspect_ranges:
                # Approximate distance to other aspect as min token distance to that span
                dist_to_other = min(abs(top1_idx - k) for k in range(o_start, o_end))
                if dist_to_other < dist_to_current:
                    closer_to_other = True
                    attracted_other_aspect = o_text
                    break

            if closer_to_other and top1_idx not in range(current_start, current_end):
                distractor_cases.append({
                    "instance_id": current_rec["instance_id"],
                    "sentence": " ".join(current_rec["tokens"]),
                    "target": current_rec["aspect_text"],
                    "gold": current_item["gold_label"],
                    "pred": current_item["pred_label"],
                    "top1_token": top1_token,
                    "top1_attention": float(weights[top1_idx]),
                    "dist_to_target": int(dist_to_current),
                    "attracted_other_aspect": attracted_other_aspect,
                })

    distractor_rate = len(distractor_cases) / total_multi_target_instances if total_multi_target_instances > 0 else 0.0

    return {
        "total_multi_target_instances": total_multi_target_instances,
        "distractor_attraction_count": len(distractor_cases),
        "distractor_attraction_rate": float(distractor_rate),
        "sample_distractor_cases": distractor_cases[:10],
    }


def run_analysis_5_classwise_performance(
    baseline_preds: list[int],
    v1_results: list[dict],
):
    golds = [r["gold_id"] for r in v1_results]
    v1_preds = [r["pred_id"] for r in v1_results]

    labels = [0, 1, 2]
    target_names = ["negative", "neutral", "positive"]

    b_prec, b_rec, b_f1, _ = precision_recall_fscore_support(golds, baseline_preds, labels=labels)
    v1_prec, v1_rec, v1_f1, _ = precision_recall_fscore_support(golds, v1_preds, labels=labels)

    classwise = {}
    for i, name in enumerate(target_names):
        classwise[name] = {
            "baseline": {
                "precision": float(b_prec[i]),
                "recall": float(b_rec[i]),
                "f1": float(b_f1[i]),
            },
            "target_evidence_v1": {
                "precision": float(v1_prec[i]),
                "recall": float(v1_rec[i]),
                "f1": float(v1_f1[i]),
            },
        }

    return classwise


def build_text_report(
    v1_checkpoint_path: str,
    baseline_checkpoint_path: str,
    analysis_1: dict,
    analysis_2: dict,
    analysis_3: dict,
    analysis_4: dict,
    analysis_5: dict,
    v1_results: list[dict],
    max_examples: int = 10,
) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("TARGET-EVIDENCE V1 DIAGNOSTIC RESEARCH REPORT (LAPTOPS TEST SET)")
    lines.append("=" * 80)
    lines.append(f"V1 Checkpoint Analyzed: {v1_checkpoint_path}")
    lines.append(f"Baseline Checkpoint:    {baseline_checkpoint_path}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("1. OVERALL & SUBGROUP CLASSIFICATION PERFORMANCE")
    lines.append("-" * 80)
    lines.append(f"Overall Test Accuracy:    {analysis_1['overall']['accuracy']:.4f}")
    lines.append(f"Overall Test Macro-F1:    {analysis_1['overall']['macro_f1']:.4f}")
    lines.append("")
    lines.append(f"Single-Aspect Instances ({analysis_1['single_aspect']['count']}):")
    lines.append(f"  - Accuracy: {analysis_1['single_aspect']['accuracy']:.4f}")
    lines.append(f"  - Macro-F1: {analysis_1['single_aspect']['macro_f1']:.4f}")
    lines.append(f"  - Gold Class Dist: {analysis_1['single_aspect']['gold_class_distribution']}")
    lines.append(f"  - Pred Class Dist: {analysis_1['single_aspect']['pred_class_distribution']}")
    lines.append("")
    lines.append(f"Multi-Aspect Instances ({analysis_1['multi_aspect']['count']}):")
    lines.append(f"  - Accuracy: {analysis_1['multi_aspect']['accuracy']:.4f}")
    lines.append(f"  - Macro-F1: {analysis_1['multi_aspect']['macro_f1']:.4f}")
    lines.append(f"  - Gold Class Dist: {analysis_1['multi_aspect']['gold_class_distribution']}")
    lines.append(f"  - Pred Class Dist: {analysis_1['multi_aspect']['pred_class_distribution']}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("2. CLASS-WISE PERFORMANCE COMPARISON (BASELINE VS V1)")
    lines.append("-" * 80)
    lines.append(f"{'Class':<12} | {'Baseline Prec/Rec/F1':<25} | {'V1 Prec/Rec/F1':<25}")
    lines.append("-" * 70)
    for cname in ["negative", "neutral", "positive"]:
        b = analysis_5[cname]["baseline"]
        v = analysis_5[cname]["target_evidence_v1"]
        b_str = f"{b['precision']:.3f} / {b['recall']:.3f} / {b['f1']:.3f}"
        v_str = f"{v['precision']:.3f} / {v['recall']:.3f} / {v['f1']:.3f}"
        lines.append(f"{cname:<12} | {b_str:<25} | {v_str:<25}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("3. EVIDENCE ATTENTION QUALITY DIAGNOSTICS")
    lines.append("-" * 80)
    for cat_name, metrics in analysis_2.items():
        lines.append(f"Category: {cat_name} (N={metrics['num_samples_analyzed']})")
        lines.append(f"  - Avg Aspect Mass (dist=0): {metrics['avg_aspect_mass_d0']:.4f}")
        lines.append(f"  - Avg Mass dist <= 1:       {metrics['avg_mass_d_le_1']:.4f}")
        lines.append(f"  - Avg Mass dist <= 2:       {metrics['avg_mass_d_le_2']:.4f}")
        lines.append(f"  - Avg Mass dist <= 3:       {metrics['avg_mass_d_le_3']:.4f}")
        lines.append(f"  - Attention Entropy:        {metrics['avg_attention_entropy']:.4f} bits")
        lines.append(f"  - Common Top-1 Tokens:     {metrics['top_1_tokens']}")
        lines.append("")

    lines.append("-" * 80)
    lines.append("4. TARGET SPECIFICITY & DISTRACTOR BEHAVIOR")
    lines.append("-" * 80)
    lines.append(f"Multi-Target Sentence Pairs Evaluated: {analysis_3['num_target_pairs_evaluated']}")
    lines.append(f"Mean Jensen-Shannon Divergence (JSD): {analysis_3['mean_jsd']:.4f} ± {analysis_3['std_jsd']:.4f}")
    lines.append(f"Mean Total Variation Distance (TVD):  {analysis_3['mean_tvd']:.4f} ± {analysis_3['std_tvd']:.4f}")
    lines.append(f"Distractor Attraction Rate:           {analysis_4['distractor_attraction_rate'] * 100:.2f}% ({analysis_4['distractor_attraction_count']}/{analysis_4['total_multi_target_instances']})")
    lines.append("")

    lines.append("-" * 80)
    lines.append("5. QUALITATIVE DIAGNOSTIC EXAMPLES")
    lines.append("-" * 80)
    for i, item in enumerate(v1_results[:max_examples]):
        rec = item["record"]
        lines.append(f"Example #{i+1} [{rec['instance_id']}]")
        lines.append(f"Sentence:     \"{ ' '.join(rec['tokens']) }\"")
        lines.append(f"Target:       {rec['aspect_text']}")
        lines.append(f"Gold:         {item['gold_label']}")
        lines.append(f"Prediction:   {item['pred_label']} ({'CORRECT' if item['is_correct'] else 'INCORRECT'})")
        lines.append(f"Multi-Aspect: {rec['is_multi_aspect']}")
        lines.append("Top Evidence Tokens:")
        top_words = sorted(item["word_details"], key=lambda x: x["attention"], reverse=True)[:5]
        for tw in top_words:
            lines.append(f"  - {tw['token']:<15} attn={tw['attention']:.4f}  distance={tw['distance']}")
        lines.append("-" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="laptops")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Locate best V1 checkpoint
    v1_root = PROJECT_ROOT / "outputs" / "target_evidence" / args.domain / f"seed_{args.seed}"
    v1_states = list(v1_root.glob("checkpoint-*/trainer_state.json"))
    if not v1_states:
        raise FileNotFoundError(f"No checkpoint trainer_state.json found under {v1_root}")

    best_v1_ckpt = None
    best_v1_f1 = -1.0
    for state_file in v1_states:
        with state_file.open() as f:
            st = json.load(f)
            if "best_model_checkpoint" in st and st.get("best_metric", -1.0) > best_v1_f1:
                best_v1_f1 = st["best_metric"]
                best_v1_ckpt = Path(st["best_model_checkpoint"])

    if best_v1_ckpt is None or not best_v1_ckpt.exists():
        best_v1_ckpt = v1_root / "checkpoint-242"

    # Locate best baseline checkpoint
    baseline_root = PROJECT_ROOT / "outputs" / "baseline" / args.domain
    baseline_states = list(baseline_root.glob("checkpoint-*/trainer_state.json"))
    best_base_ckpt = None
    best_base_f1 = -1.0
    for state_file in baseline_states:
        with state_file.open() as f:
            st = json.load(f)
            if "best_model_checkpoint" in st and st.get("best_metric", -1.0) > best_base_f1:
                best_base_f1 = st["best_metric"]
                best_base_ckpt = Path(st["best_model_checkpoint"])

    if best_base_ckpt is None or not best_base_ckpt.exists():
        best_base_ckpt = baseline_root / "checkpoint-1089"

    print("=" * 80)
    print("RUNNING DIAGNOSTIC EVALUATION FOR TARGET-EVIDENCE V1")
    print("=" * 80)
    print(f"Domain:                  {args.domain}")
    print(f"Target-Evidence V1 Ckpt: {best_v1_ckpt}")
    print(f"RoBERTa Baseline Ckpt:   {best_base_ckpt}")
    print("=" * 80)

    # Load test records
    test_path = PROJECT_ROOT / "data" / "processed" / args.domain / "test.jsonl"
    records = load_jsonl(test_path)

    # Load models
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    v1_model = load_target_evidence_checkpoint(best_v1_ckpt)
    baseline_model = load_baseline_checkpoint(best_base_ckpt)

    # Compute baseline predictions
    print("\nComputing baseline predictions...")
    baseline_preds = compute_baseline_predictions(
        model=baseline_model,
        tokenizer=tokenizer,
        records=records,
        device=device,
    )

    # Compute V1 predictions & evidence attentions
    print("Computing Target-Evidence V1 predictions & evidence attention...")
    v1_results = compute_v1_predictions_and_evidence(
        model=v1_model,
        tokenizer=tokenizer,
        records=records,
        device=device,
    )

    # Execute all 5 analyses
    print("\nRunning Analysis 1 (Single vs Multi-Aspect)...")
    analysis_1 = run_analysis_1_single_vs_multi(v1_results)

    print("Running Analysis 2 (Evidence Quality)...")
    analysis_2 = run_analysis_2_evidence_quality(v1_results)

    print("Running Analysis 3 (Target Specificity)...")
    analysis_3 = run_analysis_3_target_specificity(v1_results)

    print("Running Analysis 4 (Wrong-Target Distractor Behavior)...")
    analysis_4 = run_analysis_4_distractor_behavior(v1_results)

    print("Running Analysis 5 (Class-wise Performance Comparison)...")
    analysis_5 = run_analysis_5_classwise_performance(baseline_preds, v1_results)

    # Construct report
    report_text = build_text_report(
        v1_checkpoint_path=str(best_v1_ckpt),
        baseline_checkpoint_path=str(best_base_ckpt),
        analysis_1=analysis_1,
        analysis_2=analysis_2,
        analysis_3=analysis_3,
        analysis_4=analysis_4,
        analysis_5=analysis_5,
        v1_results=v1_results,
    )

    # Output directory setup
    output_dir = PROJECT_ROOT / "outputs" / "evidence_analysis" / args.domain
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / "diagnostics_summary.json"
    report_file = output_dir / "diagnostics_report.txt"

    combined_data = {
        "domain": args.domain,
        "best_v1_checkpoint": str(best_v1_ckpt),
        "best_baseline_checkpoint": str(best_base_ckpt),
        "analysis_1_single_vs_multi": analysis_1,
        "analysis_2_evidence_quality": analysis_2,
        "analysis_3_target_specificity": analysis_3,
        "analysis_4_distractor_behavior": analysis_4,
        "analysis_5_classwise": analysis_5,
    }

    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2)

    with report_file.open("w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)
    print("\n" + "=" * 80)
    print(f"Saved machine-readable JSON: {summary_file}")
    print(f"Saved human-readable report: {report_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
