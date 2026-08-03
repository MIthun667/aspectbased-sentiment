from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_evidence_interventions import (
    run_summary,
    summarize_values,
)


def test_summarize_values() -> None:
    result = summarize_values(
        [1.0, 2.0, 3.0]
    )

    assert result["count"] == 3
    assert result["mean"] == 2.0
    assert result[
        "standard_deviation"
    ] == pytest.approx(1.0)


def test_single_value_has_zero_std() -> None:
    result = summarize_values(
        [0.5]
    )

    assert result["count"] == 1
    assert result[
        "standard_deviation"
    ] == 0.0


def make_summary() -> dict[str, object]:
    interventions = {}

    for name in (
        "original",
        "empty",
        "aspect_only",
        "full_sentence",
        "random_same_sentence",
        "shuffled_cross_instance",
    ):
        interventions[name] = {
            "classification_metrics": {
                "accuracy": 0.8,
                "balanced_accuracy": 0.75,
                "macro_f1": 0.76,
                "weighted_f1": 0.79,
                "negative_log_likelihood": 0.9,
                "brier_score": 0.3,
            },
            "paired_metrics": {
                "prediction_flip_rate": 0.01,
                "mean_l1_probability_shift": 0.02,
                "mean_absolute_probability_shift": (
                    0.01
                ),
                "mean_gold_probability_change": (
                    -0.001
                ),
                (
                    "mean_absolute_"
                    "gold_probability_change"
                ): 0.01,
                "mean_confidence_change": -0.002,
                (
                    "mean_absolute_"
                    "confidence_change"
                ): 0.01,
                (
                    "fraction_gold_"
                    "probability_decreased"
                ): 0.5,
                "correct_to_incorrect_rate": 0.01,
                "incorrect_to_correct_rate": 0.0,
                "metric_deltas": {
                    "accuracy": -0.01,
                    "balanced_accuracy": -0.01,
                    "macro_f1": -0.01,
                    (
                        "negative_log_"
                        "likelihood"
                    ): 0.02,
                    "brier_score": 0.01,
                },
            },
        }

    return {
        "interventions": interventions
    }


def test_run_summary(
    tmp_path: Path,
) -> None:
    experiment_root = (
        tmp_path / "experiment"
    )

    path = (
        experiment_root
        / "seed_2026"
        / "evidence_interventions"
        / "laptops__test"
        / "intervention_seed_2026"
        / "summary.json"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(make_summary()),
        encoding="utf-8",
    )

    output = tmp_path / "aggregate.json"

    result_path = run_summary(
        experiment_root=(
            experiment_root
        ),
        training_seeds=[2026],
        intervention_seeds=[2026],
        domain="laptops",
        split="test",
        output=output,
    )

    assert result_path == output
    assert output.is_file()

    value = json.loads(
        output.read_text(
            encoding="utf-8"
        )
    )

    assert value[
        "number_of_runs"
    ] == 1

    assert (
        value["interventions"]["empty"][
            "classification.macro_f1"
        ]["mean"]
        == pytest.approx(0.76)
    )


def test_missing_run_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
    ):
        run_summary(
            experiment_root=tmp_path,
            training_seeds=[2026],
            intervention_seeds=[2026],
            domain="laptops",
            split="test",
            output=None,
        )
