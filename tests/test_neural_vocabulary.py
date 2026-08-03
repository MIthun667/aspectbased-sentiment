from __future__ import annotations

import pytest

from src.aspect_sentiment.models.neural import (
    PAD_TOKEN,
    TARGET_END_TOKEN,
    TARGET_START_TOKEN,
    UNK_TOKEN,
    Vocabulary,
    normalized_target_marked_tokens,
    target_marked_tokens,
)


def make_record(
    *,
    tokens: list[str],
    aspect_start: int,
    aspect_end: int,
    instance_id: str = "instance-1",
    label_id: int = 2,
) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "tokens": tokens,
        "aspect_start": aspect_start,
        "aspect_end": aspect_end,
        "label_id": label_id,
    }


def test_target_marked_tokens_preserve_span() -> None:
    record = make_record(
        tokens=[
            "The",
            "battery",
            "life",
            "is",
            "excellent",
        ],
        aspect_start=1,
        aspect_end=3,
    )

    assert target_marked_tokens(record) == [
        "The",
        TARGET_START_TOKEN,
        "battery",
        "life",
        TARGET_END_TOKEN,
        "is",
        "excellent",
    ]


def test_normalized_tokens_casefold_regular_tokens() -> None:
    record = make_record(
        tokens=["Battery", "Is", "GOOD"],
        aspect_start=0,
        aspect_end=1,
    )

    assert normalized_target_marked_tokens(
        record
    ) == [
        TARGET_START_TOKEN,
        "battery",
        TARGET_END_TOKEN,
        "is",
        "good",
    ]


def test_vocabulary_uses_fixed_special_ids() -> None:
    vocabulary = Vocabulary.build(
        [
            make_record(
                tokens=["battery", "is", "good"],
                aspect_start=0,
                aspect_end=1,
            )
        ]
    )

    assert vocabulary.lookup_id(0) == PAD_TOKEN
    assert vocabulary.lookup_id(1) == UNK_TOKEN
    assert (
        vocabulary.lookup_id(2)
        == TARGET_START_TOKEN
    )
    assert (
        vocabulary.lookup_id(3)
        == TARGET_END_TOKEN
    )


def test_vocabulary_is_deterministic() -> None:
    first_records = [
        make_record(
            tokens=["zebra", "battery"],
            aspect_start=1,
            aspect_end=2,
        ),
        make_record(
            tokens=["apple", "screen"],
            aspect_start=1,
            aspect_end=2,
        ),
    ]

    second_records = list(
        reversed(first_records)
    )

    first = Vocabulary.build(first_records)
    second = Vocabulary.build(second_records)

    assert first.id_to_token == second.id_to_token
    assert first.token_to_id == second.token_to_id


def test_minimum_frequency_filters_tokens() -> None:
    vocabulary = Vocabulary.build(
        [
            make_record(
                tokens=["common", "rare"],
                aspect_start=0,
                aspect_end=1,
            ),
            make_record(
                tokens=["common", "other"],
                aspect_start=0,
                aspect_end=1,
            ),
        ],
        minimum_frequency=2,
    )

    assert "common" in vocabulary.token_to_id
    assert "rare" not in vocabulary.token_to_id
    assert "other" not in vocabulary.token_to_id


def test_unknown_token_uses_unk_id() -> None:
    vocabulary = Vocabulary.build(
        [
            make_record(
                tokens=["known"],
                aspect_start=0,
                aspect_end=1,
            )
        ]
    )

    assert (
        vocabulary.lookup_token("missing")
        == vocabulary.unk_id
    )


def test_target_centered_truncation_preserves_markers() -> None:
    record = make_record(
        tokens=[
            "one",
            "two",
            "three",
            "target",
            "five",
            "six",
            "seven",
            "eight",
        ],
        aspect_start=3,
        aspect_end=4,
    )

    vocabulary = Vocabulary.build([record])

    encoded = vocabulary.encode_record(
        record,
        maximum_length=5,
    )

    decoded = vocabulary.decode(encoded)

    assert len(decoded) == 5
    assert TARGET_START_TOKEN in decoded
    assert TARGET_END_TOKEN in decoded
    assert "target" in decoded


def test_too_short_maximum_length_is_rejected() -> None:
    record = make_record(
        tokens=[
            "multi",
            "word",
            "target",
        ],
        aspect_start=0,
        aspect_end=3,
    )

    vocabulary = Vocabulary.build([record])

    with pytest.raises(
        ValueError,
        match="too small",
    ):
        vocabulary.encode_record(
            record,
            maximum_length=4,
        )


def test_empty_vocabulary_records_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        Vocabulary.build([])
