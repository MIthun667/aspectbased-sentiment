from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "negative_log_likelihood",
    "brier_score",
)


@dataclass(frozen=True, slots=True)
class ModelSpecification:
    display_name: str
    experiment_name: str
    seeds: tuple[int, ...]
    deterministic: bool = False


MODEL_SPECIFICATIONS = (
    ModelSpecification(
        display_name="Majority",
        experiment_name="majority_laptops",
        seeds=(2026,),
        deterministic=True,
    ),
    ModelSpecification(
        display_name="TF-IDF sentence",
        experiment_name="tfidf_sentence_laptops",
        seeds=(2026,),
        deterministic=True,
    ),
    ModelSpecification(
        display_name="TF-IDF target",
        experiment_name="tfidf_target_laptops",
        seeds=(2026,),
        deterministic=True,
    ),
    ModelSpecification(
        display_name="Target-aware BiLSTM",
        experiment_name="bilstm_target_laptops",
        seeds=(2026, 2027, 2028),
    ),
    ModelSpecification(
        display_name="DeBERTa pair",
        experiment_name="deberta_pair_laptops",
        seeds=(2026, 2027, 2028),
    ),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate saved baseline metrics across seeds."
        )
    )

    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("artifacts/experiments"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/summaries"),
    )

    parser.add_argument(
        "--domain",
        default="laptops",
    )

    parser.add_argument(
        "--split",
        default="test",
    )

    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_seed_metrics(
    *,
    experiments_root: Path,
    experiment_name: str,
    seed: int,
    domain: str,
    split: str,
) -> tuple[dict[str, float], int | None]:
    run_root = (
        experiments_root
        / experiment_name
        / f"seed_{seed}"
    )

    metrics_payload = load_json(
        run_root
        / "metrics"
        / f"{domain}__{split}.json"
    )

    metadata = load_json(
        run_root / "run_metadata.json"
    )

    if metadata["status"] != "COMPLETE":
        raise ValueError(
            "Run is not complete: "
            f"{experiment_name}/seed_{seed}"
        )

    metrics = {
        name: float(
            metrics_payload["metrics"][name]
        )
        for name in METRIC_NAMES
    }

    checkpoint_path = (
        run_root
        / "checkpoints"
        / "best.pt"
    )

    best_epoch: int | None = None

    if checkpoint_path.is_file():
        import torch

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if "epoch" in checkpoint:
            best_epoch = int(
                checkpoint["epoch"]
            )

    return metrics, best_epoch


def aggregate_values(
    values: list[float],
) -> dict[str, float]:
    if not values:
        raise ValueError(
            "Cannot aggregate an empty value list"
        )

    mean = statistics.fmean(values)

    standard_deviation = (
        statistics.stdev(values)
        if len(values) > 1
        else 0.0
    )

    return {
        "mean": mean,
        "standard_deviation": (
            standard_deviation
        ),
        "minimum": min(values),
        "maximum": max(values),
    }


def format_mean_std(
    summary: dict[str, float],
    *,
    deterministic: bool,
) -> str:
    if deterministic:
        return f"{summary['mean']:.4f}"

    return (
        f"{summary['mean']:.4f} "
        f"± {summary['standard_deviation']:.4f}"
    )


def summarize_model(
    specification: ModelSpecification,
    *,
    experiments_root: Path,
    domain: str,
    split: str,
) -> dict[str, Any]:
    seed_results = []
    best_epochs = []

    for seed in specification.seeds:
        metrics, best_epoch = load_seed_metrics(
            experiments_root=experiments_root,
            experiment_name=(
                specification.experiment_name
            ),
            seed=seed,
            domain=domain,
            split=split,
        )

        seed_results.append(
            {
                "seed": seed,
                "metrics": metrics,
                "best_epoch": best_epoch,
            }
        )

        if best_epoch is not None:
            best_epochs.append(
                float(best_epoch)
            )

    aggregated_metrics = {
        metric_name: aggregate_values(
            [
                result["metrics"][metric_name]
                for result in seed_results
            ]
        )
        for metric_name in METRIC_NAMES
    }

    best_epoch_summary = (
        aggregate_values(best_epochs)
        if best_epochs
        else None
    )

    return {
        "display_name": specification.display_name,
        "experiment_name": (
            specification.experiment_name
        ),
        "deterministic": specification.deterministic,
        "seeds": list(specification.seeds),
        "number_of_runs": len(seed_results),
        "seed_results": seed_results,
        "metrics": aggregated_metrics,
        "best_epoch": best_epoch_summary,
    }


def write_csv(
    path: Path,
    *,
    summaries: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "model",
        "seeds",
        "number_of_runs",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "negative_log_likelihood",
        "brier_score",
        "best_epoch",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for summary in summaries:
            deterministic = bool(
                summary["deterministic"]
            )

            best_epoch = summary["best_epoch"]

            writer.writerow(
                {
                    "model": summary["display_name"],
                    "seeds": ",".join(
                        str(seed)
                        for seed in summary["seeds"]
                    ),
                    "number_of_runs": (
                        summary["number_of_runs"]
                    ),
                    **{
                        metric_name: format_mean_std(
                            summary["metrics"][
                                metric_name
                            ],
                            deterministic=deterministic,
                        )
                        for metric_name in METRIC_NAMES
                    },
                    "best_epoch": (
                        ""
                        if best_epoch is None
                        else format_mean_std(
                            best_epoch,
                            deterministic=deterministic,
                        )
                    ),
                }
            )


def main() -> None:
    arguments = parse_arguments()

    summaries = [
        summarize_model(
            specification,
            experiments_root=(
                arguments.experiments_root
            ),
            domain=arguments.domain,
            split=arguments.split,
        )
        for specification in MODEL_SPECIFICATIONS
    ]

    arguments.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        arguments.output_root
        / f"{arguments.domain}_baseline_summary.json"
    )

    csv_path = (
        arguments.output_root
        / f"{arguments.domain}_baseline_summary.csv"
    )

    json_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "domain": arguments.domain,
                "split": arguments.split,
                "models": summaries,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    write_csv(
        csv_path,
        summaries=summaries,
    )

    print("BASELINE SUMMARY")
    print("=" * 80)

    for summary in summaries:
        metric = summary["metrics"][
            "macro_f1"
        ]

        print(
            f"{summary['display_name']:<24}"
            f"{format_mean_std(metric, deterministic=summary['deterministic'])}"
        )

    print("=" * 80)
    print("JSON:", json_path)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
