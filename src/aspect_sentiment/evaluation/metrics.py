from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


LABEL_IDS = (0, 1, 2)

LABEL_NAMES = (
    "negative",
    "neutral",
    "positive",
)

EPSILON = 1e-12


def _as_label_array(
    values: Sequence[int] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional, "
            f"received shape {array.shape}"
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty"
        )

    invalid = sorted(
        set(array.tolist()).difference(LABEL_IDS)
    )

    if invalid:
        raise ValueError(
            f"{name} contains invalid labels: {invalid}"
        )

    return array


def _validate_probabilities(
    probabilities: Sequence[Sequence[float]]
    | np.ndarray,
    *,
    number_of_instances: int,
) -> np.ndarray:
    array = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    expected_shape = (
        number_of_instances,
        len(LABEL_IDS),
    )

    if array.shape != expected_shape:
        raise ValueError(
            "probabilities must have shape "
            f"{expected_shape}, received {array.shape}"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "probabilities contain non-finite values"
        )

    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(
            "probabilities must be within [0, 1]"
        )

    row_sums = array.sum(axis=1)

    if not np.allclose(
        row_sums,
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "every probability row must sum to 1"
        )

    return array


def multiclass_negative_log_likelihood(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[Sequence[float]]
    | np.ndarray,
) -> float:
    true_array = _as_label_array(
        y_true,
        name="y_true",
    )

    probability_array = _validate_probabilities(
        probabilities,
        number_of_instances=true_array.size,
    )

    true_probabilities = probability_array[
        np.arange(true_array.size),
        true_array,
    ]

    return float(
        -np.mean(
            np.log(
                np.clip(
                    true_probabilities,
                    EPSILON,
                    1.0,
                )
            )
        )
    )


def multiclass_brier_score(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[Sequence[float]]
    | np.ndarray,
) -> float:
    true_array = _as_label_array(
        y_true,
        name="y_true",
    )

    probability_array = _validate_probabilities(
        probabilities,
        number_of_instances=true_array.size,
    )

    one_hot = np.eye(
        len(LABEL_IDS),
        dtype=np.float64,
    )[true_array]

    return float(
        np.mean(
            np.sum(
                np.square(
                    probability_array - one_hot
                ),
                axis=1,
            )
        )
    )


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    probabilities: Sequence[Sequence[float]]
    | np.ndarray
    | None = None,
) -> dict[str, Any]:
    """
    Compute canonical three-class ABSA evaluation metrics.

    All outputs are plain Python values and can be serialized directly
    to JSON.
    """
    true_array = _as_label_array(
        y_true,
        name="y_true",
    )
    predicted_array = _as_label_array(
        y_pred,
        name="y_pred",
    )

    if true_array.shape != predicted_array.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape: "
            f"{true_array.shape} != {predicted_array.shape}"
        )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            true_array,
            predicted_array,
            labels=LABEL_IDS,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        true_array,
        predicted_array,
        labels=LABEL_IDS,
    )

    per_class = {
        label_name: {
            "label_id": int(label_id),
            "precision": float(
                precision[label_id]
            ),
            "recall": float(
                recall[label_id]
            ),
            "f1": float(
                f1[label_id]
            ),
            "support": int(
                support[label_id]
            ),
        }
        for label_id, label_name in zip(
            LABEL_IDS,
            LABEL_NAMES,
        )
    }

    result: dict[str, Any] = {
        "number_of_instances": int(
            true_array.size
        ),
        "label_order": list(LABEL_NAMES),
        "label_ids": list(LABEL_IDS),
        "accuracy": float(
            accuracy_score(
                true_array,
                predicted_array,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                true_array,
                predicted_array,
            )
        ),
        "macro_f1": float(
            f1_score(
                true_array,
                predicted_array,
                labels=LABEL_IDS,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true_array,
                predicted_array,
                labels=LABEL_IDS,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": per_class,
        "confusion_matrix": (
            matrix.astype(int).tolist()
        ),
    }

    if probabilities is not None:
        probability_array = (
            _validate_probabilities(
                probabilities,
                number_of_instances=(
                    true_array.size
                ),
            )
        )

        probability_predictions = (
            probability_array.argmax(axis=1)
        )

        if not np.array_equal(
            probability_predictions,
            predicted_array,
        ):
            raise ValueError(
                "y_pred does not match argmax(probabilities)"
            )

        result["negative_log_likelihood"] = (
            multiclass_negative_log_likelihood(
                true_array,
                probability_array,
            )
        )
        result["brier_score"] = (
            multiclass_brier_score(
                true_array,
                probability_array,
            )
        )

    return result
