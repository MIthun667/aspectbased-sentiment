from __future__ import annotations

import pytest
from transformers import (
    AutoTokenizer,
)

from src.aspect_sentiment.data import (
    SubwordAlignment,
    align_words_to_subwords,
)


MODEL_NAME = "microsoft/deberta-v3-base"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        local_files_only=True,
    )


def test_aligns_sentence_and_aspect_pair(
    tokenizer,
) -> None:
    alignment = align_words_to_subwords(
        tokenizer=tokenizer,
        tokens=[
            "The",
            "battery",
            "life",
            "is",
            "excellent",
            ".",
        ],
        aspect_start=1,
        aspect_end=3,
        selected_evidence_indices=(4,),
        maximum_length=128,
    )

    assert isinstance(
        alignment,
        SubwordAlignment,
    )

    assert len(alignment.input_ids) > 0

    expected_length = len(
        alignment.input_ids
    )

    assert len(
        alignment.attention_mask
    ) == expected_length

    assert len(
        alignment.word_ids
    ) == expected_length

    assert len(
        alignment.sequence_ids
    ) == expected_length

    assert any(
        alignment.sentence_subword_mask
    )

    assert any(
        alignment.pair_aspect_subword_mask
    )

    assert any(
        alignment.aspect_subword_mask
    )

    assert any(
        alignment.evidence_subword_mask
    )

    assert not alignment.aspect_truncated
    assert not alignment.evidence_truncated


def test_special_tokens_receive_no_masks(
    tokenizer,
) -> None:
    alignment = align_words_to_subwords(
        tokenizer=tokenizer,
        tokens=[
            "battery",
            "is",
            "good",
        ],
        aspect_start=0,
        aspect_end=1,
        selected_evidence_indices=(2,),
        maximum_length=64,
    )

    for index, (
        word_id,
        sequence_id,
    ) in enumerate(
        zip(
            alignment.word_ids,
            alignment.sequence_ids,
            strict=True,
        )
    ):
        if (
            word_id is None
            or sequence_id is None
        ):
            assert not (
                alignment
                .sentence_subword_mask[
                    index
                ]
            )

            assert not (
                alignment
                .aspect_subword_mask[
                    index
                ]
            )

            assert not (
                alignment
                .evidence_subword_mask[
                    index
                ]
            )

            assert not (
                alignment
                .pair_aspect_subword_mask[
                    index
                ]
            )


def test_all_subwords_of_evidence_word_are_marked(
    tokenizer,
) -> None:
    tokens = [
        "The",
        "battery",
        "is",
        "unbelievably",
        "good",
    ]

    alignment = align_words_to_subwords(
        tokenizer=tokenizer,
        tokens=tokens,
        aspect_start=1,
        aspect_end=2,
        selected_evidence_indices=(3,),
        maximum_length=128,
    )

    evidence_positions = [
        index
        for index, (
            word_id,
            sequence_id,
        ) in enumerate(
            zip(
                alignment.word_ids,
                alignment.sequence_ids,
                strict=True,
            )
        )
        if (
            sequence_id == 0
            and word_id == 3
        )
    ]

    assert evidence_positions

    assert all(
        alignment.evidence_subword_mask[
            index
        ]
        for index in evidence_positions
    )


def test_repeated_words_use_word_indices(
    tokenizer,
) -> None:
    alignment = align_words_to_subwords(
        tokenizer=tokenizer,
        tokens=[
            "good",
            "battery",
            "but",
            "good",
            "screen",
        ],
        aspect_start=1,
        aspect_end=2,
        selected_evidence_indices=(0,),
        maximum_length=128,
    )

    first_good_positions = [
        index
        for index, (
            word_id,
            sequence_id,
        ) in enumerate(
            zip(
                alignment.word_ids,
                alignment.sequence_ids,
                strict=True,
            )
        )
        if (
            sequence_id == 0
            and word_id == 0
        )
    ]

    second_good_positions = [
        index
        for index, (
            word_id,
            sequence_id,
        ) in enumerate(
            zip(
                alignment.word_ids,
                alignment.sequence_ids,
                strict=True,
            )
        )
        if (
            sequence_id == 0
            and word_id == 3
        )
    ]

    assert first_good_positions
    assert second_good_positions

    assert all(
        alignment.evidence_subword_mask[
            index
        ]
        for index in first_good_positions
    )

    assert all(
        not alignment.evidence_subword_mask[
            index
        ]
        for index in second_good_positions
    )


def test_aspect_mask_only_marks_sentence_side(
    tokenizer,
) -> None:
    alignment = align_words_to_subwords(
        tokenizer=tokenizer,
        tokens=[
            "The",
            "battery",
            "life",
            "is",
            "good",
        ],
        aspect_start=1,
        aspect_end=3,
        selected_evidence_indices=(4,),
        maximum_length=128,
    )

    for index, sequence_id in enumerate(
        alignment.sequence_ids
    ):
        if sequence_id == 1:
            assert not (
                alignment
                .aspect_subword_mask[
                    index
                ]
            )

    assert any(
        alignment
        .pair_aspect_subword_mask
    )


def test_empty_evidence_produces_empty_mask(
    tokenizer,
) -> None:
    alignment = align_words_to_subwords(
        tokenizer=tokenizer,
        tokens=[
            "battery",
            "is",
            "acceptable",
        ],
        aspect_start=0,
        aspect_end=1,
        selected_evidence_indices=(),
        maximum_length=64,
    )

    assert not any(
        alignment.evidence_subword_mask
    )

    assert not alignment.evidence_truncated


def test_truncation_is_reported(
    tokenizer,
) -> None:
    tokens = [
        f"token{index}"
        for index in range(100)
    ]

    alignment = align_words_to_subwords(
        tokenizer=tokenizer,
        tokens=tokens,
        aspect_start=95,
        aspect_end=97,
        selected_evidence_indices=(90,),
        maximum_length=24,
    )

    assert alignment.aspect_truncated
    assert alignment.evidence_truncated


def test_invalid_aspect_span_rejected(
    tokenizer,
) -> None:
    with pytest.raises(
        ValueError,
        match="Invalid aspect span",
    ):
        align_words_to_subwords(
            tokenizer=tokenizer,
            tokens=["battery", "good"],
            aspect_start=2,
            aspect_end=3,
            selected_evidence_indices=(1,),
            maximum_length=64,
        )


def test_unsorted_evidence_indices_rejected(
    tokenizer,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be sorted",
    ):
        align_words_to_subwords(
            tokenizer=tokenizer,
            tokens=[
                "very",
                "good",
                "battery",
            ],
            aspect_start=2,
            aspect_end=3,
            selected_evidence_indices=(
                1,
                0,
            ),
            maximum_length=64,
        )


def test_out_of_range_evidence_rejected(
    tokenizer,
) -> None:
    with pytest.raises(
        ValueError,
        match="out of range",
    ):
        align_words_to_subwords(
            tokenizer=tokenizer,
            tokens=["battery", "good"],
            aspect_start=0,
            aspect_end=1,
            selected_evidence_indices=(9,),
            maximum_length=64,
        )
