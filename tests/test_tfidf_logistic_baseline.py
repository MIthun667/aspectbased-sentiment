from __future__ import annotations

import numpy as np
import pytest

from src.aspect_sentiment.models.baselines import (
    TfidfLogisticRegressionBaseline,
    records_to_texts,
    sentence_text,
    target_marked_text,
)


def make_record(
    *,
    tokens: list[str],
    aspect_start: int,
    aspect_end: int,
) -> dict[str, object]:
    return {
        "tokens": tokens,
        "aspect_start": aspect_start,
        "aspect_end": aspect_end,
    }


def training_records() -> list[dict[str, object]]:
    return [
        make_record(
            tokens=[
                "battery",
                "life",
                "is",
                "excellent",
            ],
            aspect_start=0,
            aspect_end=2,
        ),
        make_record(
            tokens=[
                "screen",
                "quality",
                "is",
                "poor",
            ],
            aspect_start=0,
            aspect_end=2,
        ),
        make_record(
            tokens=[
                "keyboard",
                "feels",
                "average",
            ],
            aspect_start=0,
            aspect_end=1,
        ),
        make_record(
            tokens=[
                "performance",
                "is",
                "excellent",
            ],
            aspect_start=0,
            aspect_end=1,
        ),
        make_record(
            tokens=[
                "touchpad",
                "is",
                "poor",
            ],
            aspect_start=0,
            aspect_end=1,
        ),
        make_record(
            tokens=[
                "design",
                "is",
                "average",
            ],
            aspect_start=0,
            aspect_end=1,
        ),
    ]


def test_sentence_text_joins_tokens() -> None:
    record = make_record(
        tokens=["battery", "is", "good"],
        aspect_start=0,
        aspect_end=1,
    )

    assert sentence_text(record) == (
        "battery is good"
    )


def test_target_marked_text_inserts_markers() -> None:
    record = make_record(
        tokens=[
            "the",
            "battery",
            "life",
            "is",
            "excellent",
        ],
        aspect_start=1,
        aspect_end=3,
    )

    assert target_marked_text(record) == (
        "the <TARGET> battery life "
        "</TARGET> is excellent"
    )


def test_records_to_texts_rejects_unknown_representation() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported representation",
    ):
        records_to_texts(
            [],
            representation="unknown",
        )


def test_sentence_model_fits_and_predicts() -> None:
    records = training_records()
    labels = [2, 0, 1, 2, 0, 1]

    model = (
        TfidfLogisticRegressionBaseline.create(
            representation="sentence",
            min_df=1,
            max_features=None,
            random_state=2026,
        )
    )

    model.fit(records, labels)

    predictions = model.predict(records)
    probabilities = model.predict_proba(records)

    assert predictions.shape == (6,)
    assert probabilities.shape == (6, 3)
    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )
    assert model.vocabulary_size > 0


def test_target_marked_model_fits_and_predicts() -> None:
    records = training_records()
    labels = [2, 0, 1, 2, 0, 1]

    model = (
        TfidfLogisticRegressionBaseline.create(
            representation="target_marked",
            min_df=1,
            max_features=None,
            random_state=2026,
        )
    )

    model.fit(records, labels)

    probabilities = model.predict_proba(records)

    assert probabilities.shape == (6, 3)
    assert np.isfinite(probabilities).all()


def test_fit_rejects_empty_records() -> None:
    model = (
        TfidfLogisticRegressionBaseline.create(
            representation="sentence",
        )
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        model.fit([], [])


def test_fit_rejects_length_mismatch() -> None:
    model = (
        TfidfLogisticRegressionBaseline.create(
            representation="sentence",
            min_df=1,
        )
    )

    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        model.fit(
            training_records(),
            [0, 1],
        )


def test_fit_rejects_single_class() -> None:
    model = (
        TfidfLogisticRegressionBaseline.create(
            representation="sentence",
            min_df=1,
        )
    )

    with pytest.raises(
        ValueError,
        match="at least two classes",
    ):
        model.fit(
            training_records(),
            [1, 1, 1, 1, 1, 1],
        )


def test_invalid_ngram_range_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="two integers",
    ):
        TfidfLogisticRegressionBaseline.create(
            representation="sentence",
            ngram_range=(1,),
        )
