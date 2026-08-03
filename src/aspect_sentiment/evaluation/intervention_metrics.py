from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .metrics import (
    LABEL_IDS,
    _as_label_array,
    _validate_probabilities,
    compute_classification_metrics,
)


def compute_intervention_metrics(
    *,
    labels: Sequence[int] | np.ndarray,
    original_predictions: (
        Sequence[int] | np.ndarray
    ),
    original_probabilities: (
        Sequence[Sequence[float]]
        | np.ndarray
    ),
    intervened_predictions: (
        Sequence[int] | np.ndarray
    ),
    intervened_probabilities: (
        Sequence[Sequence[float]]
        | np.ndarray
    ),
) -> dict[str, Any]:
    label_array = _as_label_array(
        labels,
        name="labels",
    )

    original_prediction_array = (
        _as_label_array(
            original_predictions,
            name="original_predictions",
        )
    )

    intervened_prediction_array = (
        _as_label_array(
            intervened_predictions,
            name="intervened_predictions",
        )
    )

    number_of_instances = int(
        label_array.size
    )

    if (
        original_prediction_array.shape
        != label_array.shape
    ):
        raise ValueError(
            "original_predictions must match "
            "the label shape"
        )

    if (
        intervened_prediction_array.shape
        != label_array.shape
    ):
        raise ValueError(
            "intervened_predictions must match "
            "the label shape"
        )

    original_probability_array = (
        _validate_probabilities(
            original_probabilities,
            number_of_instances=(
                number_of_instances
            ),
        )
    )

    intervened_probability_array = (
        _validate_probabilities(
            intervened_probabilities,
            number_of_instances=(
                number_of_instances
            ),
        )
    )

    if not np.array_equal(
        original_probability_array.argmax(
            axis=1
        ),
        original_prediction_array,
    ):
        raise ValueError(
            "original_predictions do not match "
            "argmax(original_probabilities)"
        )

    if not np.array_equal(
        intervened_probability_array.argmax(
            axis=1
        ),
        intervened_prediction_array,
    ):
        raise ValueError(
            "intervened_predictions do not match "
            "argmax(intervened_probabilities)"
        )

    original_metrics = (
        compute_classification_metrics(
            label_array,
            original_prediction_array,
            probabilities=(
                original_probability_array
            ),
        )
    )

    intervened_metrics = (
        compute_classification_metrics(
            label_array,
            intervened_prediction_array,
            probabilities=(
                intervened_probability_array
            ),
        )
    )

    row_indices = np.arange(
        number_of_instances
    )

    original_gold_probabilities = (
        original_probability_array[
            row_indices,
            label_array,
        ]
    )

    intervened_gold_probabilities = (
        intervened_probability_array[
            row_indices,
            label_array,
        ]
    )

    original_confidences = (
        original_probability_array.max(
            axis=1
        )
    )

    intervened_confidences = (
        intervened_probability_array.max(
            axis=1
        )
    )

    probability_difference = (
        intervened_probability_array
        - original_probability_array
    )

    gold_probability_change = (
        intervened_gold_probabilities
        - original_gold_probabilities
    )

    confidence_change = (
        intervened_confidences
        - original_confidences
    )

    prediction_changed = (
        intervened_prediction_array
        != original_prediction_array
    )

    originally_correct = (
        original_prediction_array
        == label_array
    )

    intervened_correct = (
        intervened_prediction_array
        == label_array
    )

    correct_to_incorrect = (
        originally_correct
        & ~intervened_correct
    )

    incorrect_to_correct = (
        ~originally_correct
        & intervened_correct
    )

    per_class_flip_rate: dict[
        str,
        float,
    ] = {}

    for label_id in LABEL_IDS:
        class_mask = (
            label_array == label_id
        )

        if not class_mask.any():
            per_class_flip_rate[
                str(label_id)
            ] = 0.0
        else:
            per_class_flip_rate[
                str(label_id)
            ] = float(
                prediction_changed[
                    class_mask
                ].mean()
            )

    return {
        "number_of_instances": (
            number_of_instances
        ),
        "original_metrics": (
            original_metrics
        ),
        "intervened_metrics": (
            intervened_metrics
        ),
        "metric_deltas": {
            "accuracy": float(
                intervened_metrics[
                    "accuracy"
                ]
                - original_metrics[
                    "accuracy"
                ]
            ),
            "balanced_accuracy": float(
                intervened_metrics[
                    "balanced_accuracy"
                ]
                - original_metrics[
                    "balanced_accuracy"
                ]
            ),
            "macro_f1": float(
                intervened_metrics[
                    "macro_f1"
                ]
                - original_metrics[
                    "macro_f1"
                ]
            ),
            "negative_log_likelihood": float(
                intervened_metrics[
                    "negative_log_likelihood"
                ]
                - original_metrics[
                    "negative_log_likelihood"
                ]
            ),
            "brier_score": float(
                intervened_metrics[
                    "brier_score"
                ]
                - original_metrics[
                    "brier_score"
                ]
            ),
        },
        "prediction_flip_rate": float(
            prediction_changed.mean()
        ),
        "number_of_prediction_flips": int(
            prediction_changed.sum()
        ),
        "correct_to_incorrect_rate": float(
            correct_to_incorrect.mean()
        ),
        "incorrect_to_correct_rate": float(
            incorrect_to_correct.mean()
        ),
        "number_correct_to_incorrect": int(
            correct_to_incorrect.sum()
        ),
        "number_incorrect_to_correct": int(
            incorrect_to_correct.sum()
        ),
        "mean_absolute_probability_shift": (
            float(
                np.abs(
                    probability_difference
                ).mean()
            )
        ),
        "mean_l1_probability_shift": float(
            np.abs(
                probability_difference
            ).sum(axis=1).mean()
        ),
        "mean_gold_probability_change": (
            float(
                gold_probability_change.mean()
            )
        ),
        "mean_absolute_gold_probability_change": (
            float(
                np.abs(
                    gold_probability_change
                ).mean()
            )
        ),
        "mean_confidence_change": float(
            confidence_change.mean()
        ),
        "mean_absolute_confidence_change": (
            float(
                np.abs(
                    confidence_change
                ).mean()
            )
        ),
        "fraction_gold_probability_decreased": (
            float(
                (
                    gold_probability_change
                    < 0.0
                ).mean()
            )
        ),
        "per_class_flip_rate": (
            per_class_flip_rate
        ),
    }
