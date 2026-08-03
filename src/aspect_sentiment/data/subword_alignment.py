from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from transformers import (
    BatchEncoding,
    PreTrainedTokenizerBase,
)


@dataclass(frozen=True, slots=True)
class SubwordAlignment:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    token_type_ids: tuple[int, ...] | None
    word_ids: tuple[int | None, ...]
    sequence_ids: tuple[int | None, ...]
    sentence_subword_mask: tuple[bool, ...]
    aspect_subword_mask: tuple[bool, ...]
    evidence_subword_mask: tuple[bool, ...]
    pair_aspect_subword_mask: tuple[bool, ...]
    aspect_truncated: bool
    evidence_truncated: bool

    def __post_init__(self) -> None:
        length = len(self.input_ids)

        aligned_lengths = {
            len(self.attention_mask),
            len(self.word_ids),
            len(self.sequence_ids),
            len(self.sentence_subword_mask),
            len(self.aspect_subword_mask),
            len(self.evidence_subword_mask),
            len(self.pair_aspect_subword_mask),
        }

        if self.token_type_ids is not None:
            aligned_lengths.add(
                len(self.token_type_ids)
            )

        if aligned_lengths != {length}:
            raise ValueError(
                "All subword-aligned fields must "
                "have the same length"
            )


def _require_fast_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    if not tokenizer.is_fast:
        raise ValueError(
            "Subword alignment requires a fast tokenizer"
        )


def _validate_word_inputs(
    tokens: Sequence[str],
    *,
    aspect_start: int,
    aspect_end: int,
    selected_evidence_indices: Sequence[int],
    maximum_length: int,
) -> None:
    if not tokens:
        raise ValueError(
            "tokens must not be empty"
        )

    if not all(
        isinstance(token, str) and token
        for token in tokens
    ):
        raise TypeError(
            "tokens must contain non-empty strings"
        )

    if maximum_length <= 0:
        raise ValueError(
            "maximum_length must be positive"
        )

    if not (
        0
        <= aspect_start
        < aspect_end
        <= len(tokens)
    ):
        raise ValueError(
            "Invalid aspect span"
        )

    if not all(
        isinstance(index, int)
        for index in selected_evidence_indices
    ):
        raise TypeError(
            "selected_evidence_indices "
            "must contain integers"
        )

    if tuple(
        selected_evidence_indices
    ) != tuple(
        sorted(selected_evidence_indices)
    ):
        raise ValueError(
            "selected_evidence_indices "
            "must be sorted"
        )

    if len(
        selected_evidence_indices
    ) != len(
        set(selected_evidence_indices)
    ):
        raise ValueError(
            "selected_evidence_indices "
            "must not contain duplicates"
        )

    for index in selected_evidence_indices:
        if not (
            0 <= index < len(tokens)
        ):
            raise ValueError(
                "Evidence index out of range: "
                f"{index}"
            )


def _normalise_optional_int_list(
    values: object,
    *,
    field_name: str,
    expected_length: int,
) -> tuple[int, ...] | None:
    if values is None:
        return None

    if not isinstance(values, list):
        raise TypeError(
            f"{field_name} must be a list"
        )

    if len(values) != expected_length:
        raise ValueError(
            f"{field_name} length does not match input_ids"
        )

    return tuple(
        int(value)
        for value in values
    )


def _extract_alignment_metadata(
    encoded: BatchEncoding,
) -> tuple[
    tuple[int | None, ...],
    tuple[int | None, ...],
]:
    word_ids = encoded.word_ids(
        batch_index=0
    )

    sequence_ids = encoded.sequence_ids(
        batch_index=0
    )

    if word_ids is None:
        raise ValueError(
            "Tokenizer did not return word_ids"
        )

    if sequence_ids is None:
        raise ValueError(
            "Tokenizer did not return sequence_ids"
        )

    return (
        tuple(word_ids),
        tuple(sequence_ids),
    )


def align_words_to_subwords(
    *,
    tokenizer: PreTrainedTokenizerBase,
    tokens: Sequence[str],
    aspect_start: int,
    aspect_end: int,
    selected_evidence_indices: Sequence[int],
    maximum_length: int,
) -> SubwordAlignment:
    _require_fast_tokenizer(tokenizer)

    _validate_word_inputs(
        tokens,
        aspect_start=aspect_start,
        aspect_end=aspect_end,
        selected_evidence_indices=(
            selected_evidence_indices
        ),
        maximum_length=maximum_length,
    )

    sentence_words = list(tokens)

    aspect_words = list(
        tokens[
            aspect_start:aspect_end
        ]
    )

    encoded = tokenizer(
        sentence_words,
        text_pair=aspect_words,
        is_split_into_words=True,
        truncation=True,
        max_length=maximum_length,
        padding=False,
        return_attention_mask=True,
        return_token_type_ids=True,
    )

    input_ids = encoded.get(
        "input_ids"
    )

    attention_mask = encoded.get(
        "attention_mask"
    )

    if not isinstance(input_ids, list):
        raise TypeError(
            "Tokenizer did not return input_ids"
        )

    if not isinstance(
        attention_mask,
        list,
    ):
        raise TypeError(
            "Tokenizer did not return attention_mask"
        )

    if not input_ids:
        raise ValueError(
            "Tokenizer produced an empty sequence"
        )

    if len(input_ids) != len(
        attention_mask
    ):
        raise ValueError(
            "input_ids and attention_mask "
            "lengths do not match"
        )

    word_ids, sequence_ids = (
        _extract_alignment_metadata(
            encoded
        )
    )

    if len(word_ids) != len(input_ids):
        raise ValueError(
            "word_ids length does not match input_ids"
        )

    if len(sequence_ids) != len(
        input_ids
    ):
        raise ValueError(
            "sequence_ids length does not match input_ids"
        )

    evidence_index_set = set(
        selected_evidence_indices
    )

    sentence_subword_mask: list[bool] = []
    aspect_subword_mask: list[bool] = []
    evidence_subword_mask: list[bool] = []
    pair_aspect_subword_mask: list[bool] = []

    observed_sentence_words: set[int] = set()

    for word_id, sequence_id in zip(
        word_ids,
        sequence_ids,
        strict=True,
    ):
        is_sentence_subword = (
            sequence_id == 0
            and word_id is not None
        )

        is_pair_aspect_subword = (
            sequence_id == 1
            and word_id is not None
        )

        sentence_subword_mask.append(
            is_sentence_subword
        )

        pair_aspect_subword_mask.append(
            is_pair_aspect_subword
        )

        if is_sentence_subword:
            assert word_id is not None

            observed_sentence_words.add(
                word_id
            )

            aspect_subword_mask.append(
                aspect_start
                <= word_id
                < aspect_end
            )

            evidence_subword_mask.append(
                word_id
                in evidence_index_set
            )
        else:
            aspect_subword_mask.append(
                False
            )

            evidence_subword_mask.append(
                False
            )

    aspect_word_indices = set(
        range(
            aspect_start,
            aspect_end,
        )
    )

    missing_aspect_words = (
        aspect_word_indices
        - observed_sentence_words
    )

    missing_evidence_words = (
        evidence_index_set
        - observed_sentence_words
    )

    token_type_ids = (
        _normalise_optional_int_list(
            encoded.get(
                "token_type_ids"
            ),
            field_name="token_type_ids",
            expected_length=len(input_ids),
        )
    )

    return SubwordAlignment(
        input_ids=tuple(
            int(value)
            for value in input_ids
        ),
        attention_mask=tuple(
            int(value)
            for value in attention_mask
        ),
        token_type_ids=token_type_ids,
        word_ids=word_ids,
        sequence_ids=sequence_ids,
        sentence_subword_mask=tuple(
            sentence_subword_mask
        ),
        aspect_subword_mask=tuple(
            aspect_subword_mask
        ),
        evidence_subword_mask=tuple(
            evidence_subword_mask
        ),
        pair_aspect_subword_mask=tuple(
            pair_aspect_subword_mask
        ),
        aspect_truncated=bool(
            missing_aspect_words
        ),
        evidence_truncated=bool(
            missing_evidence_words
        ),
    )
