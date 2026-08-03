from __future__ import annotations

import numpy as np
import pytest

from src.aspect_sentiment.models.baselines import (
    MajorityClassBaseline,
)


def test_fit_selects_majority_label() -> None:
    model = MajorityClassBaseline.fit(
        [0, 1, 2, 2, 2, 1]
    )

    assert model.majority_label_id == 2
    assert model.class_counts == (1, 2, 3)


def test_fit_computes_training_distribution() -> None:
    model = MajorityClassBaseline.fit(
        [0, 1, 2, 2]
    )

    assert model.class_probabilities == pytest.approx(
        (
            0.25,
            0.25,
            0.50,
        )
    )


def test_predict_returns_majority_label() -> None:
    model = MajorityClassBaseline.fit(
        [0, 2, 2]
    )

    predictions = model.predict(4)

    assert np.array_equal(
        predictions,
        np.asarray([2, 2, 2, 2]),
    )


def test_predict_proba_has_expected_shape() -> None:
    model = MajorityClassBaseline.fit(
        [0, 1, 2, 2]
    )

    probabilities = model.predict_proba(3)

    assert probabilities.shape == (3, 3)
    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )


def test_ties_use_lowest_label_id() -> None:
    model = MajorityClassBaseline.fit(
        [0, 1, 2]
    )

    assert model.majority_label_id == 0


def test_empty_training_labels_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        MajorityClassBaseline.fit([])


def test_invalid_labels_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid labels",
    ):
        MajorityClassBaseline.fit(
            [0, 1, 3]
        )


def test_negative_prediction_size_is_rejected() -> None:
    model = MajorityClassBaseline.fit(
        [0, 1, 1]
    )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        model.predict(-1)
