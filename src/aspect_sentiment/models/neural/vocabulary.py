from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
TARGET_START_TOKEN = "<TARGET>"
TARGET_END_TOKEN = "</TARGET>"

SPECIAL_TOKENS = (
    PAD_TOKEN,
    UNK_TOKEN,
    TARGET_START_TOKEN,
    TARGET_END_TOKEN,
)


def validate_tokens(
    record: dict[str, Any],
) -> list[str]:
    tokens = record.get("tokens")

    if not isinstance(tokens, list):
        raise TypeError(
            "record tokens must be a list"
        )

    if not all(
        isinstance(token, str)
        for token in tokens
    ):
        raise TypeError(
            "record tokens must contain strings"
        )

    return tokens


def target_marked_tokens(
    record: dict[str, Any],
) -> list[str]:
    tokens = validate_tokens(record)

    aspect_start = record.get("aspect_start")
    aspect_end = record.get("aspect_end")

    if not isinstance(aspect_start, int):
        raise TypeError(
            "aspect_start must be an integer"
        )

    if not isinstance(aspect_end, int):
        raise TypeError(
            "aspect_end must be an integer"
        )

    if not (
        0 <= aspect_start < aspect_end
        <= len(tokens)
    ):
        raise ValueError(
            "Invalid aspect span: "
            f"[{aspect_start}, {aspect_end}) "
            f"for {len(tokens)} tokens"
        )

    return [
        *tokens[:aspect_start],
        TARGET_START_TOKEN,
        *tokens[aspect_start:aspect_end],
        TARGET_END_TOKEN,
        *tokens[aspect_end:],
    ]


def normalize_token(
    token: str,
) -> str:
    if token in {
        TARGET_START_TOKEN,
        TARGET_END_TOKEN,
    }:
        return token

    return token.casefold()


def normalized_target_marked_tokens(
    record: dict[str, Any],
) -> list[str]:
    return [
        normalize_token(token)
        for token in target_marked_tokens(record)
    ]


@dataclass(frozen=True, slots=True)
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: tuple[str, ...]
    minimum_frequency: int

    @classmethod
    def build(
        cls,
        records: Iterable[dict[str, Any]],
        *,
        minimum_frequency: int = 1,
    ) -> "Vocabulary":
        if minimum_frequency <= 0:
            raise ValueError(
                "minimum_frequency must be positive"
            )

        counter: Counter[str] = Counter()

        record_count = 0

        for record in records:
            counter.update(
                normalized_target_marked_tokens(
                    record
                )
            )
            record_count += 1

        if record_count == 0:
            raise ValueError(
                "Cannot build vocabulary from empty records"
            )

        regular_tokens = sorted(
            token
            for token, frequency in counter.items()
            if (
                frequency >= minimum_frequency
                and token not in SPECIAL_TOKENS
            )
        )

        ordered_tokens = (
            *SPECIAL_TOKENS,
            *regular_tokens,
        )

        token_to_id = {
            token: index
            for index, token in enumerate(
                ordered_tokens
            )
        }

        return cls(
            token_to_id=token_to_id,
            id_to_token=ordered_tokens,
            minimum_frequency=minimum_frequency,
        )

    def __len__(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def target_start_id(self) -> int:
        return self.token_to_id[
            TARGET_START_TOKEN
        ]

    @property
    def target_end_id(self) -> int:
        return self.token_to_id[
            TARGET_END_TOKEN
        ]

    def lookup_token(
        self,
        token: str,
    ) -> int:
        normalized = normalize_token(token)

        return self.token_to_id.get(
            normalized,
            self.unk_id,
        )

    def lookup_id(
        self,
        token_id: int,
    ) -> str:
        if not (
            0 <= token_id < len(self)
        ):
            raise IndexError(
                f"Token ID out of range: {token_id}"
            )

        return self.id_to_token[token_id]

    def encode(
        self,
        tokens: Sequence[str],
    ) -> list[int]:
        return [
            self.lookup_token(token)
            for token in tokens
        ]

    def decode(
        self,
        token_ids: Sequence[int],
    ) -> list[str]:
        return [
            self.lookup_id(token_id)
            for token_id in token_ids
        ]

    def encode_record(
        self,
        record: dict[str, Any],
        *,
        maximum_length: int,
    ) -> list[int]:
        if maximum_length <= 0:
            raise ValueError(
                "maximum_length must be positive"
            )

        tokens = normalized_target_marked_tokens(
            record
        )

        if len(tokens) <= maximum_length:
            return self.encode(tokens)

        target_start = tokens.index(
            TARGET_START_TOKEN
        )
        target_end = tokens.index(
            TARGET_END_TOKEN
        )

        target_width = (
            target_end - target_start + 1
        )

        if target_width > maximum_length:
            raise ValueError(
                "maximum_length is too small to preserve "
                "the complete target span"
            )

        remaining = (
            maximum_length - target_width
        )

        left_available = target_start
        right_available = (
            len(tokens) - target_end - 1
        )

        left_budget = min(
            left_available,
            remaining // 2,
        )

        right_budget = min(
            right_available,
            remaining - left_budget,
        )

        unused = (
            remaining
            - left_budget
            - right_budget
        )

        if unused > 0:
            additional_left = min(
                left_available - left_budget,
                unused,
            )
            left_budget += additional_left
            unused -= additional_left

        if unused > 0:
            additional_right = min(
                right_available - right_budget,
                unused,
            )
            right_budget += additional_right

        start = (
            target_start - left_budget
        )

        end = (
            target_end
            + right_budget
            + 1
        )

        truncated = tokens[start:end]

        if len(truncated) != maximum_length:
            raise RuntimeError(
                "Target-centered truncation produced "
                "an unexpected sequence length"
            )

        if TARGET_START_TOKEN not in truncated:
            raise RuntimeError(
                "Target start marker was lost"
            )

        if TARGET_END_TOKEN not in truncated:
            raise RuntimeError(
                "Target end marker was lost"
            )

        return self.encode(truncated)
