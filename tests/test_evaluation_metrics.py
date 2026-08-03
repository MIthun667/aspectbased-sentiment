from __future__ import annotations

import json

import numpy as np
import pytest

from src.aspect_sentiment.evaluation.metrics import (
    LABEL_IDS,
    LABEL_NAMES,
    compute_classification_metrics,
    multiclass_brier_score,
    multiclass_negative_log_likelihood,
)


def test_label_order_matches_canonical_schema() -> None:
    assert LABEL_IDS == (0, 1, 2)
    assert LABEL_NAMES == (
        "negative",
        "neutral",
        "positive",
    )


def test_perfect_predictions_produce_perfect_metrics() -> None:
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]

    result = compute_classification_metrics(
        y_true,
        y_pred,
    )

    assert result["accuracy"] == pytest.approx(1.0)
    assert result["balanced_accuracy"] == pytest.approx(
        1.0
    )
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["weighted_f1"] == pytest.approx(1.0)
    assert result["confusion_matrix"] == [
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
    ]


def test_missing_prediction_class_remains_reported() -> None:
    result = compute_classification_metrics(
        y_true=[0, 1, 2],
        y_pred=[0, 0, 0],
    )

    assert set(result["per_class"]) == {
        "negative",
        "neutral",
        "positive",
    }
    assert (
        result["per_class"]["positive"]["f1"]
        == pytest.approx(0.0)
    )


def test_probability_metrics_are_computed() -> None:
    y_true = [0, 1, 2]
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float64,
    )

    result = compute_classification_metrics(
        y_true,
        probabilities.argmax(axis=1),
        probabilities=probabilities,
    )

    assert result["negative_log_likelihood"] > 0.0
    assert result["brier_score"] > 0.0

    assert result["negative_log_likelihood"] == pytest.approx(
        multiclass_negative_log_likelihood(
            y_true,
            probabilities,
        )
    )
    assert result["brier_score"] == pytest.approx(
        multiclass_brier_score(
            y_true,
            probabilities,
        )
    )

    json.dumps(result)


def test_mismatched_label_lengths_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        compute_classification_metrics(
            y_true=[0, 1, 2],
            y_pred=[0, 1],
        )


def test_invalid_probability_shape_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must have shape",
    ):
        compute_classification_metrics(
            y_true=[0, 1],
            y_pred=[0, 1],
            probabilities=[
                [0.8, 0.2],
                [0.1, 0.9],
            ],
        )


def test_probability_argmax_must_match_predictions() -> None:
    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        compute_classification_metrics(
            y_true=[0, 1],
            y_pred=[1, 0],
            probabilities=[
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
            ],
        )
