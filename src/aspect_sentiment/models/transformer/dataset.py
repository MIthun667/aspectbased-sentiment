from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)


def sentence_from_record(
    record: dict[str, Any],
) -> str:
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

    if not tokens:
        raise ValueError(
            "record tokens must not be empty"
        )

    return " ".join(tokens)


def aspect_from_record(
    record: dict[str, Any],
) -> str:
    aspect_text = record.get("aspect_text")

    if not isinstance(aspect_text, str):
        raise TypeError(
            "aspect_text must be a string"
        )

    aspect_text = aspect_text.strip()

    if not aspect_text:
        raise ValueError(
            "aspect_text must not be empty"
        )

    return aspect_text


@dataclass(frozen=True, slots=True)
class TransformerInstance:
    instance_id: str
    sentence_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    token_type_ids: tuple[int, ...] | None
    label_id: int


class TransformerAspectDataset(
    Dataset[TransformerInstance]
):
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        tokenizer: PreTrainedTokenizerBase,
        maximum_length: int,
    ) -> None:
        if not records:
            raise ValueError(
                "Dataset records must not be empty"
            )

        if maximum_length <= 0:
            raise ValueError(
                "maximum_length must be positive"
            )

        self._instances = tuple(
            self._encode_record(
                record,
                tokenizer=tokenizer,
                maximum_length=maximum_length,
            )
            for record in records
        )

    @staticmethod
    def _encode_record(
        record: dict[str, Any],
        *,
        tokenizer: PreTrainedTokenizerBase,
        maximum_length: int,
    ) -> TransformerInstance:
        instance_id = record.get("instance_id")
        sentence_id = record.get("sentence_id")
        label_id = record.get("label_id")

        if not isinstance(instance_id, str):
            raise TypeError(
                "instance_id must be a string"
            )

        if not isinstance(sentence_id, str):
            raise TypeError(
                "sentence_id must be a string"
            )

        if label_id not in {
            0,
            1,
            2,
        }:
            raise ValueError(
                "label_id must be 0, 1, or 2"
            )

        encoded = tokenizer(
            sentence_from_record(record),
            text_pair=aspect_from_record(record),
            truncation=True,
            max_length=maximum_length,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        input_ids = encoded.get("input_ids")
        attention_mask = encoded.get(
            "attention_mask"
        )
        token_type_ids = encoded.get(
            "token_type_ids"
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
                "Tokenizer did not return "
                "attention_mask"
            )

        if len(input_ids) != len(
            attention_mask
        ):
            raise ValueError(
                "input_ids and attention_mask "
                "lengths do not match"
            )

        if not input_ids:
            raise ValueError(
                "Tokenizer produced an empty sequence"
            )

        normalized_token_types: (
            tuple[int, ...] | None
        )

        if token_type_ids is None:
            normalized_token_types = None
        else:
            if not isinstance(
                token_type_ids,
                list,
            ):
                raise TypeError(
                    "token_type_ids must be a list"
                )

            if len(token_type_ids) != len(
                input_ids
            ):
                raise ValueError(
                    "token_type_ids and input_ids "
                    "lengths do not match"
                )

            normalized_token_types = tuple(
                int(value)
                for value in token_type_ids
            )

        return TransformerInstance(
            instance_id=instance_id,
            sentence_id=sentence_id,
            input_ids=tuple(
                int(value)
                for value in input_ids
            ),
            attention_mask=tuple(
                int(value)
                for value in attention_mask
            ),
            token_type_ids=(
                normalized_token_types
            ),
            label_id=int(label_id),
        )

    def __len__(self) -> int:
        return len(self._instances)

    def __getitem__(
        self,
        index: int,
    ) -> TransformerInstance:
        return self._instances[index]


@dataclass(frozen=True, slots=True)
class TransformerBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None
    labels: torch.Tensor
    instance_ids: tuple[str, ...]
    sentence_ids: tuple[str, ...]


@dataclass(slots=True)
class TransformerBatchCollator:
    tokenizer: PreTrainedTokenizerBase
    _padding_collator: DataCollatorWithPadding = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._padding_collator = (
            DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True,
                return_tensors="pt",
            )
        )

    def __call__(
        self,
        instances: Sequence[
            TransformerInstance
        ],
    ) -> TransformerBatch:
        if not instances:
            raise ValueError(
                "Cannot collate an empty batch"
            )

        features: list[
            dict[str, list[int]]
        ] = []

        use_token_type_ids = all(
            instance.token_type_ids
            is not None
            for instance in instances
        )

        for instance in instances:
            feature = {
                "input_ids": list(
                    instance.input_ids
                ),
                "attention_mask": list(
                    instance.attention_mask
                ),
            }

            if use_token_type_ids:
                assert (
                    instance.token_type_ids
                    is not None
                )

                feature["token_type_ids"] = list(
                    instance.token_type_ids
                )

            features.append(feature)

        padded = self._padding_collator(
            features
        )

        token_type_tensor = padded.get(
            "token_type_ids"
        )

        return TransformerBatch(
            input_ids=padded["input_ids"],
            attention_mask=(
                padded["attention_mask"]
            ),
            token_type_ids=(
                token_type_tensor
                if token_type_tensor is not None
                else None
            ),
            labels=torch.tensor(
                [
                    instance.label_id
                    for instance in instances
                ],
                dtype=torch.long,
            ),
            instance_ids=tuple(
                instance.instance_id
                for instance in instances
            ),
            sentence_ids=tuple(
                instance.sentence_id
                for instance in instances
            ),
        )
