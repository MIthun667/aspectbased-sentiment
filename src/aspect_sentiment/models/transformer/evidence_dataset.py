from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch.utils.data import Dataset
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)

from src.aspect_sentiment.data import (
    EvidenceAwareInstance,
    align_words_to_subwords,
)


@dataclass(frozen=True, slots=True)
class EvidenceTransformerInstance:
    instance_id: str
    sentence_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    token_type_ids: tuple[int, ...] | None
    aspect_subword_mask: tuple[bool, ...]
    evidence_subword_mask: tuple[bool, ...]
    sentence_subword_mask: tuple[bool, ...]
    pair_aspect_subword_mask: tuple[bool, ...]
    label_id: int
    evidence_is_empty: bool


class EvidenceTransformerDataset(
    Dataset[EvidenceTransformerInstance]
):
    def __init__(
        self,
        instances: Sequence[
            EvidenceAwareInstance
        ],
        *,
        tokenizer: PreTrainedTokenizerBase,
        maximum_length: int = 128,
        reject_truncation: bool = True,
    ) -> None:
        if not instances:
            raise ValueError(
                "Dataset instances must not be empty"
            )

        if maximum_length <= 0:
            raise ValueError(
                "maximum_length must be positive"
            )

        self._instances = tuple(
            self._encode_instance(
                instance,
                tokenizer=tokenizer,
                maximum_length=maximum_length,
                reject_truncation=(
                    reject_truncation
                ),
            )
            for instance in instances
        )

    @staticmethod
    def _encode_instance(
        instance: EvidenceAwareInstance,
        *,
        tokenizer: PreTrainedTokenizerBase,
        maximum_length: int,
        reject_truncation: bool,
    ) -> EvidenceTransformerInstance:
        alignment = align_words_to_subwords(
            tokenizer=tokenizer,
            tokens=instance.tokens,
            aspect_start=instance.aspect_start,
            aspect_end=instance.aspect_end,
            selected_evidence_indices=(
                instance
                .selected_evidence_indices
            ),
            maximum_length=maximum_length,
        )

        if (
            reject_truncation
            and alignment.aspect_truncated
        ):
            raise ValueError(
                "Aspect was truncated for "
                f"{instance.instance_id!r}"
            )

        if (
            reject_truncation
            and alignment.evidence_truncated
        ):
            raise ValueError(
                "Evidence was truncated for "
                f"{instance.instance_id!r}"
            )

        return EvidenceTransformerInstance(
            instance_id=instance.instance_id,
            sentence_id=instance.sentence_id,
            input_ids=alignment.input_ids,
            attention_mask=(
                alignment.attention_mask
            ),
            token_type_ids=(
                alignment.token_type_ids
            ),
            aspect_subword_mask=(
                alignment.aspect_subword_mask
            ),
            evidence_subword_mask=(
                alignment.evidence_subword_mask
            ),
            sentence_subword_mask=(
                alignment
                .sentence_subword_mask
            ),
            pair_aspect_subword_mask=(
                alignment
                .pair_aspect_subword_mask
            ),
            label_id=instance.label_id,
            evidence_is_empty=(
                instance.evidence_is_empty
            ),
        )

    def __len__(self) -> int:
        return len(self._instances)

    def __getitem__(
        self,
        index: int,
    ) -> EvidenceTransformerInstance:
        return self._instances[index]


@dataclass(frozen=True, slots=True)
class EvidenceTransformerBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None
    aspect_subword_mask: torch.Tensor
    evidence_subword_mask: torch.Tensor
    sentence_subword_mask: torch.Tensor
    pair_aspect_subword_mask: torch.Tensor
    labels: torch.Tensor
    evidence_is_empty: torch.Tensor
    instance_ids: tuple[str, ...]
    sentence_ids: tuple[str, ...]


@dataclass(slots=True)
class EvidenceTransformerBatchCollator:
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

    @staticmethod
    def _pad_bool_sequences(
        sequences: Sequence[
            tuple[bool, ...]
        ],
        *,
        maximum_length: int,
    ) -> torch.Tensor:
        output = torch.zeros(
            (
                len(sequences),
                maximum_length,
            ),
            dtype=torch.bool,
        )

        for row_index, sequence in enumerate(
            sequences
        ):
            output[
                row_index,
                : len(sequence),
            ] = torch.tensor(
                sequence,
                dtype=torch.bool,
            )

        return output

    def __call__(
        self,
        instances: Sequence[
            EvidenceTransformerInstance
        ],
    ) -> EvidenceTransformerBatch:
        if not instances:
            raise ValueError(
                "Cannot collate an empty batch"
            )

        use_token_type_ids = all(
            instance.token_type_ids
            is not None
            for instance in instances
        )

        features: list[
            dict[str, list[int]]
        ] = []

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

        maximum_length = int(
            padded["input_ids"].shape[1]
        )

        token_type_tensor = padded.get(
            "token_type_ids"
        )

        return EvidenceTransformerBatch(
            input_ids=padded["input_ids"],
            attention_mask=(
                padded["attention_mask"]
            ),
            token_type_ids=(
                token_type_tensor
                if token_type_tensor is not None
                else None
            ),
            aspect_subword_mask=(
                self._pad_bool_sequences(
                    [
                        instance
                        .aspect_subword_mask
                        for instance in instances
                    ],
                    maximum_length=(
                        maximum_length
                    ),
                )
            ),
            evidence_subword_mask=(
                self._pad_bool_sequences(
                    [
                        instance
                        .evidence_subword_mask
                        for instance in instances
                    ],
                    maximum_length=(
                        maximum_length
                    ),
                )
            ),
            sentence_subword_mask=(
                self._pad_bool_sequences(
                    [
                        instance
                        .sentence_subword_mask
                        for instance in instances
                    ],
                    maximum_length=(
                        maximum_length
                    ),
                )
            ),
            pair_aspect_subword_mask=(
                self._pad_bool_sequences(
                    [
                        instance
                        .pair_aspect_subword_mask
                        for instance in instances
                    ],
                    maximum_length=(
                        maximum_length
                    ),
                )
            ),
            labels=torch.tensor(
                [
                    instance.label_id
                    for instance in instances
                ],
                dtype=torch.long,
            ),
            evidence_is_empty=torch.tensor(
                [
                    instance.evidence_is_empty
                    for instance in instances
                ],
                dtype=torch.bool,
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
