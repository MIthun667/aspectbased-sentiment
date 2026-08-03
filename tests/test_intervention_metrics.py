from __future__ import annotations

import numpy as np
import pytest

from src.aspect_sentiment.evaluation import (
    compute_intervention_metrics,
)


def baseline_values():
    labels = np.asarray(
        [0, 1, 2, 0],
        dtype=np.int64,
    )

    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
        ],
        dtype=np.float64,
    )

    predictions = probabilities.argmax(
        axis=1
    )

    return (
        labels,
        predictions,
        probabilities,
    )


def test_identical_predictions_have_zero_shift() -> None:
    (
        labels,
        predictions,
        probabilities,
    ) = baseline_values()

    result = compute_intervention_metrics(
        labels=labels,
        original_predictions=predictions,
        original_probabilities=probabilities,
        intervened_predictions=predictions,
        intervened_probabilities=probabilities,
    )

    assert (
        result["prediction_flip_rate"]
        == 0.0
    )

    assert (
        result[
            "mean_absolute_probability_shift"
        ]
        == 0.0
    )

    assert (
        result[
            "mean_gold_probability_change"
        ]
        == 0.0
    )

    assert all(
        value == 0.0
        for value in result[
            "metric_deltas"
        ].values()
    )


def test_prediction_flips_are_counted() -> None:
    (
        labels,
        predictions,
        probabilities,
    ) = baseline_values()

    intervened_probabilities = (
        probabilities.copy()
    )

    intervened_probabilities[0] = [
        0.1,
        0.8,
        0.1,
    ]

    intervened_predictions = (
        intervened_probabilities.argmax(
            axis=1
        )
    )

    result = compute_intervention_metrics(
        labels=labels,
        original_predictions=predictions,
        original_probabilities=probabilities,
        intervened_predictions=(
            intervened_predictions
        ),
        intervened_probabilities=(
            intervened_probabilities
        ),
    )

    assert (
        result[
            "number_of_prediction_flips"
        ]
        == 1
    )

    assert (
        result["prediction_flip_rate"]
        == pytest.approx(0.25)
    )

    assert (
        result[
            "number_correct_to_incorrect"
        ]
        == 1
    )

    assert (
        result[
            "number_incorrect_to_correct"
        ]
        == 0
    )


def test_gold_probability_drop_is_signed() -> None:
    (
        labels,
        predictions,
        probabilities,
    ) = baseline_values()

    intervened_probabilities = (
        probabilities.copy()
    )

    intervened_probabilities[0] = [
        0.6,
        0.3,
        0.1,
    ]

    intervened_predictions = (
        intervened_probabilities.argmax(
            axis=1
        )
    )

    result = compute_intervention_metrics(
        labels=labels,
        original_predictions=predictions,
        original_probabilities=probabilities,
        intervened_predictions=(
            intervened_predictions
        ),
        intervened_probabilities=(
            intervened_probabilities
        ),
    )

    assert (
        result[
            "mean_gold_probability_change"
        ]
        == pytest.approx(-0.05)
    )

    assert (
        result[
            "fraction_gold_probability_decreased"
        ]
        == pytest.approx(0.25)
    )


def test_incorrect_to_correct_is_counted() -> None:
    labels = np.asarray(
        [0, 1],
        dtype=np.int64,
    )

    original_probabilities = np.asarray(
        [
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
        ]
    )

    intervened_probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
        ]
    )

    result = compute_intervention_metrics(
        labels=labels,
        original_predictions=(
            original_probabilities.argmax(
                axis=1
            )
        ),
        original_probabilities=(
            original_probabilities
        ),
        intervened_predictions=(
            intervened_probabilities.argmax(
                axis=1
            )
        ),
        intervened_probabilities=(
            intervened_probabilities
        ),
    )

    assert (
        result[
            "number_incorrect_to_correct"
        ]
        == 1
    )

    assert (
        result[
            "incorrect_to_correct_rate"
        ]
        == pytest.approx(0.5)
    )


def test_per_class_flip_rates() -> None:
    (
        labels,
        predictions,
        probabilities,
    ) = baseline_values()

    intervened_probabilities = (
        probabilities.copy()
    )

    intervened_probabilities[0] = [
        0.1,
        0.8,
        0.1,
    ]

    intervened_predictions = (
        intervened_probabilities.argmax(
            axis=1
        )
    )

    result = compute_intervention_metrics(
        labels=labels,
        original_predictions=predictions,
        original_probabilities=probabilities,
        intervened_predictions=(
            intervened_predictions
        ),
        intervened_probabilities=(
            intervened_probabilities
        ),
    )

    assert (
        result["per_class_flip_rate"]["0"]
        == pytest.approx(0.5)
    )

    assert (
        result["per_class_flip_rate"]["1"]
        == 0.0
    )

    assert (
        result["per_class_flip_rate"]["2"]
        == 0.0
    )


def test_shape_mismatch_rejected() -> None:
    (
        labels,
        predictions,
        probabilities,
    ) = baseline_values()

    with pytest.raises(
        ValueError,
        match="must match",
    ):
        compute_intervention_metrics(
            labels=labels,
            original_predictions=(
                predictions[:-1]
            ),
            original_probabilities=(
                probabilities
            ),
            intervened_predictions=(
                predictions
            ),
            intervened_probabilities=(
                probabilities
            ),
        )


def test_original_argmax_mismatch_rejected() -> None:
    (
        labels,
        predictions,
        probabilities,
    ) = baseline_values()

    invalid_predictions = (
        predictions.copy()
    )

    invalid_predictions[0] = 1

    with pytest.raises(
        ValueError,
        match="original_predictions",
    ):
        compute_intervention_metrics(
            labels=labels,
            original_predictions=(
                invalid_predictions
            ),
            original_probabilities=(
                probabilities
            ),
            intervened_predictions=(
                predictions
            ),
            intervened_probabilities=(
                probabilities
            ),
        )


def test_intervened_argmax_mismatch_rejected() -> None:
    (
        labels,
        predictions,
        probabilities,
    ) = baseline_values()

    invalid_predictions = (
        predictions.copy()
    )

    invalid_predictions[1] = 0

    with pytest.raises(
        ValueError,
        match="intervened_predictions",
    ):
        compute_intervention_metrics(
            labels=labels,
            original_predictions=(
                predictions
            ),
            original_probabilities=(
                probabilities
            ),
            intervened_predictions=(
                invalid_predictions
            ),
            intervened_probabilities=(
                probabilities
            ),
        )
