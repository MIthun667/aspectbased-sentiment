from __future__ import annotations

import pytest

from src.aspect_sentiment.evidence import (
    AspectSpan,
    aspect_spans_from_sentence_records,
    attribute_candidate_to_aspects,
    identify_current_aspect_index,
)


def make_sentence_records() -> list[
    dict[str, object]
]:
    tokens = [
        "The",
        "cord",
        "is",
        "heavy",
        "but",
        "battery",
        "life",
        "is",
        "good",
        ".",
    ]

    common = {
        "sentence_id": "sentence-1",
        "tokens": tokens,
    }

    return [
        {
            **common,
            "instance_id": "sentence-1:aspect-0",
            "aspect_text": "cord",
            "aspect_start": 1,
            "aspect_end": 2,
        },
        {
            **common,
            "instance_id": "sentence-1:aspect-1",
            "aspect_text": "battery life",
            "aspect_start": 5,
            "aspect_end": 7,
        },
    ]


def test_builds_ordered_aspect_spans() -> None:
    spans = aspect_spans_from_sentence_records(
        make_sentence_records()
    )

    assert len(spans) == 2
    assert spans[0].text == "cord"
    assert spans[1].text == "battery life"
    assert spans[1].aspect_index == 1


def test_identifies_current_aspect() -> None:
    records = make_sentence_records()

    spans = aspect_spans_from_sentence_records(
        records
    )

    assert identify_current_aspect_index(
        records[1],
        aspect_spans=spans,
    ) == 1


def test_good_is_unique_to_battery() -> None:
    spans = (
        AspectSpan(
            aspect_index=0,
            start=1,
            end=2,
            text="cord",
        ),
        AspectSpan(
            aspect_index=1,
            start=5,
            end=7,
            text="battery life",
        ),
    )

    attribution = (
        attribute_candidate_to_aspects(
            token_index=8,
            current_aspect_index=1,
            aspect_spans=spans,
        )
    )

    assert attribution.is_current_aspect_nearest
    assert attribution.is_unique_to_current_aspect
    assert not attribution.is_tied
    assert (
        attribution.nearest_aspect_indices
        == (1,)
    )


def test_good_is_competing_for_cord() -> None:
    spans = (
        AspectSpan(
            aspect_index=0,
            start=1,
            end=2,
            text="cord",
        ),
        AspectSpan(
            aspect_index=1,
            start=5,
            end=7,
            text="battery life",
        ),
    )

    attribution = (
        attribute_candidate_to_aspects(
            token_index=8,
            current_aspect_index=0,
            aspect_spans=spans,
        )
    )

    assert not attribution.is_current_aspect_nearest
    assert attribution.is_closer_to_competing_aspect
    assert (
        attribution.nearest_aspect_indices
        == (1,)
    )


def test_midpoint_can_be_tied() -> None:
    spans = (
        AspectSpan(
            aspect_index=0,
            start=1,
            end=2,
            text="left",
        ),
        AspectSpan(
            aspect_index=1,
            start=5,
            end=6,
            text="right",
        ),
    )

    attribution = (
        attribute_candidate_to_aspects(
            token_index=3,
            current_aspect_index=0,
            aspect_spans=spans,
        )
    )

    assert attribution.is_tied
    assert attribution.is_current_aspect_nearest
    assert not attribution.is_unique_to_current_aspect
    assert (
        attribution.nearest_aspect_indices
        == (0, 1)
    )


def test_rejects_mixed_sentences() -> None:
    records = make_sentence_records()
    records[1]["sentence_id"] = "other"

    with pytest.raises(
        ValueError,
        match="one sentence",
    ):
        aspect_spans_from_sentence_records(
            records
        )


def test_rejects_unknown_current_aspect() -> None:
    records = make_sentence_records()

    spans = aspect_spans_from_sentence_records(
        records
    )

    unknown = {
        **records[0],
        "aspect_text": "unknown",
    }

    with pytest.raises(
        ValueError,
        match="uniquely",
    ):
        identify_current_aspect_index(
            unknown,
            aspect_spans=spans,
        )
