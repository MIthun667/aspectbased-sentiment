from __future__ import annotations

from src.aspect_sentiment.evaluation import (
    exclude_text_overlaps,
    normalize_text,
    normalized_record_text,
)


def make_record(
    *,
    instance_id: str,
    tokens: list[str],
) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "tokens": tokens,
    }


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text(
        "  Battery   LIFE  "
    ) == "battery life"


def test_normalized_record_text_uses_tokens() -> None:
    record = make_record(
        instance_id="one",
        tokens=["Battery", "Life"],
    )

    assert normalized_record_text(record) == (
        "battery life"
    )


def test_overlap_records_are_excluded() -> None:
    training_records = [
        make_record(
            instance_id="train-1",
            tokens=[
                "Battery",
                "life",
                "is",
                "good",
            ],
        )
    ]

    evaluation_records = [
        make_record(
            instance_id="test-1",
            tokens=[
                "battery",
                "life",
                "is",
                "good",
            ],
        ),
        make_record(
            instance_id="test-2",
            tokens=[
                "screen",
                "is",
                "bright",
            ],
        ),
    ]

    retained, excluded = exclude_text_overlaps(
        evaluation_records,
        reference_records=training_records,
    )

    assert [
        record["instance_id"]
        for record in retained
    ] == ["test-2"]

    assert [
        record["instance_id"]
        for record in excluded
    ] == ["test-1"]


def test_multiple_aspects_of_duplicate_sentence_are_excluded() -> None:
    training_records = [
        make_record(
            instance_id="train-1",
            tokens=["food", "was", "excellent"],
        )
    ]

    evaluation_records = [
        make_record(
            instance_id="test-1",
            tokens=["food", "was", "excellent"],
        ),
        make_record(
            instance_id="test-2",
            tokens=["food", "was", "excellent"],
        ),
    ]

    retained, excluded = exclude_text_overlaps(
        evaluation_records,
        reference_records=training_records,
    )

    assert retained == []
    assert len(excluded) == 2


def test_non_overlapping_records_are_preserved() -> None:
    retained, excluded = exclude_text_overlaps(
        [
            make_record(
                instance_id="test-1",
                tokens=["new", "sentence"],
            )
        ],
        reference_records=[
            make_record(
                instance_id="train-1",
                tokens=["other", "sentence"],
            )
        ],
    )

    assert len(retained) == 1
    assert excluded == []
