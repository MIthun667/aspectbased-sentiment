from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


INTERVENTIONS = (
    "original",
    "empty",
    "aspect_only",
    "full_sentence",
    "random_same_sentence",
    "shuffled_cross_instance",
)

SCALAR_FIELDS = (
    "prediction_flip_rate",
    "mean_l1_probability_shift",
    "mean_absolute_probability_shift",
    "mean_gold_probability_change",
    "mean_absolute_gold_probability_change",
    "mean_confidence_change",
    "mean_absolute_confidence_change",
    "fraction_gold_probability_decreased",
    "correct_to_incorrect_rate",
    "incorrect_to_correct_rate",
)

CLASSIFICATION_FIELDS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "negative_log_likelihood",
    "brier_score",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate evidence-intervention "
            "evaluations across training and "
            "intervention seeds."
        )
    )

    parser.add_argument(
        "experiment_root",
        type=Path,
        help=(
            "Root containing seed_2026, seed_2027, "
            "and seed_2028 experiment directories."
        ),
    )

    parser.add_argument(
        "--training-seeds",
        type=int,
        nargs="+",
        default=[2026, 2027, 2028],
    )

    parser.add_argument(
        "--intervention-seeds",
        type=int,
        nargs="+",
        default=[2026, 2027, 2028],
    )

    parser.add_argument(
        "--domain",
        default="laptops",
    )

    parser.add_argument(
        "--split",
        default="test",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Summary file not found: {path}"
        )

    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return value


def summarize_values(
    values: list[float],
) -> dict[str, float | int]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size == 0:
        raise ValueError(
            "Cannot summarize an empty value list"
        )

    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "standard_deviation": float(
            array.std(ddof=1)
            if array.size > 1
            else 0.0
        ),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def run_summary(
    *,
    experiment_root: Path,
    training_seeds: list[int],
    intervention_seeds: list[int],
    domain: str,
    split: str,
    output: Path | None,
) -> Path:
    collected: dict[
        str,
        dict[str, list[float]],
    ] = {
        intervention: {}
        for intervention in INTERVENTIONS
    }

    runs: list[dict[str, Any]] = []

    for training_seed in training_seeds:
        for intervention_seed in (
            intervention_seeds
        ):
            path = (
                experiment_root
                / f"seed_{training_seed}"
                / "evidence_interventions"
                / f"{domain}__{split}"
                / (
                    "intervention_seed_"
                    f"{intervention_seed}"
                )
                / "summary.json"
            )

            summary = load_json(path)

            runs.append(
                {
                    "training_seed": (
                        training_seed
                    ),
                    "intervention_seed": (
                        intervention_seed
                    ),
                    "path": str(path),
                }
            )

            for intervention in (
                INTERVENTIONS
            ):
                result = summary[
                    "interventions"
                ][intervention]

                classification = result[
                    "classification_metrics"
                ]

                paired = result[
                    "paired_metrics"
                ]

                fields = collected[
                    intervention
                ]

                for field in (
                    CLASSIFICATION_FIELDS
                ):
                    key = (
                        f"classification.{field}"
                    )

                    fields.setdefault(
                        key,
                        [],
                    ).append(
                        float(
                            classification[field]
                        )
                    )

                for field in SCALAR_FIELDS:
                    fields.setdefault(
                        f"paired.{field}",
                        [],
                    ).append(
                        float(paired[field])
                    )

                metric_deltas = paired[
                    "metric_deltas"
                ]

                for field, value in (
                    metric_deltas.items()
                ):
                    fields.setdefault(
                        f"delta.{field}",
                        [],
                    ).append(
                        float(value)
                    )

    aggregated = {
        "schema_version": "1.0",
        "experiment_root": str(
            experiment_root
        ),
        "domain": domain,
        "split": split,
        "training_seeds": (
            training_seeds
        ),
        "intervention_seeds": (
            intervention_seeds
        ),
        "number_of_runs": len(runs),
        "runs": runs,
        "interventions": {},
    }

    for intervention, fields in (
        collected.items()
    ):
        aggregated["interventions"][
            intervention
        ] = {
            field: summarize_values(
                values
            )
            for field, values in sorted(
                fields.items()
            )
        }

    output_path = (
        output
        if output is not None
        else (
            experiment_root
            / "evidence_intervention_summary.json"
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            aggregated,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print(
        "MULTI-SEED EVIDENCE INTERVENTION SUMMARY"
    )
    print("=" * 100)
    print(
        f"Runs: {len(runs)}"
    )

    for intervention in INTERVENTIONS:
        values = aggregated[
            "interventions"
        ][intervention]

        macro_f1 = values[
            "classification.macro_f1"
        ]

        nll = values[
            (
                "classification."
                "negative_log_likelihood"
            )
        ]

        flip = values[
            "paired.prediction_flip_rate"
        ]

        gold_change = values[
            (
                "paired."
                "mean_gold_probability_change"
            )
        ]

        print("-" * 100)
        print(intervention)
        print(
            "Macro-F1: "
            f"{macro_f1['mean']:.4f} "
            f"± "
            f"{macro_f1['standard_deviation']:.4f}"
        )
        print(
            "NLL:      "
            f"{nll['mean']:.4f} "
            f"± "
            f"{nll['standard_deviation']:.4f}"
        )
        print(
            "Flip:     "
            f"{flip['mean']:.4f} "
            f"± "
            f"{flip['standard_deviation']:.4f}"
        )
        print(
            "Gold Δp:  "
            f"{gold_change['mean']:.4f} "
            f"± "
            f"{gold_change['standard_deviation']:.4f}"
        )

    print("-" * 100)
    print(f"Written: {output_path}")

    return output_path


def main() -> None:
    arguments = parse_arguments()

    run_summary(
        experiment_root=(
            arguments.experiment_root
        ),
        training_seeds=list(
            arguments.training_seeds
        ),
        intervention_seeds=list(
            arguments.intervention_seeds
        ),
        domain=arguments.domain,
        split=arguments.split,
        output=arguments.output,
    )


if __name__ == "__main__":
    main()
