from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .vocabulary import Vocabulary


@dataclass(frozen=True, slots=True)
class EncodedInstance:
    instance_id: str
    token_ids: tuple[int, ...]
    label_id: int


class AspectSentimentDataset(
    Dataset[EncodedInstance]
):
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        vocabulary: Vocabulary,
        maximum_length: int,
    ) -> None:
        if not records:
            raise ValueError(
                "Dataset records must not be empty"
            )

        self._instances: tuple[
            EncodedInstance,
            ...
        ] = tuple(
            self._encode_record(
                record,
                vocabulary=vocabulary,
                maximum_length=maximum_length,
            )
            for record in records
        )

    @staticmethod
    def _encode_record(
        record: dict[str, Any],
        *,
        vocabulary: Vocabulary,
        maximum_length: int,
    ) -> EncodedInstance:
        instance_id = record.get(
            "instance_id"
        )
        label_id = record.get(
            "label_id"
        )

        if not isinstance(instance_id, str):
            raise TypeError(
                "instance_id must be a string"
            )

        if label_id not in {
            0,
            1,
            2,
        }:
            raise ValueError(
                "label_id must be 0, 1, or 2"
            )

        token_ids = vocabulary.encode_record(
            record,
            maximum_length=maximum_length,
        )

        if not token_ids:
            raise ValueError(
                "Encoded token sequence must not be empty"
            )

        return EncodedInstance(
            instance_id=instance_id,
            token_ids=tuple(token_ids),
            label_id=int(label_id),
        )

    def __len__(self) -> int:
        return len(self._instances)

    def __getitem__(
        self,
        index: int,
    ) -> EncodedInstance:
        return self._instances[index]


@dataclass(frozen=True, slots=True)
class NeuralBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    instance_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NeuralBatchCollator:
    pad_id: int

    def __call__(
        self,
        instances: Sequence[EncodedInstance],
    ) -> NeuralBatch:
        if not instances:
            raise ValueError(
                "Cannot collate an empty batch"
            )

        maximum_length = max(
            len(instance.token_ids)
            for instance in instances
        )

        batch_size = len(instances)

        input_ids = torch.full(
            (
                batch_size,
                maximum_length,
            ),
            fill_value=self.pad_id,
            dtype=torch.long,
        )

        attention_mask = torch.zeros(
            (
                batch_size,
                maximum_length,
            ),
            dtype=torch.bool,
        )

        labels = torch.empty(
            batch_size,
            dtype=torch.long,
        )

        instance_ids: list[str] = []

        for row_index, instance in enumerate(
            instances
        ):
            sequence_length = len(
                instance.token_ids
            )

            input_ids[
                row_index,
                :sequence_length,
            ] = torch.tensor(
                instance.token_ids,
                dtype=torch.long,
            )

            attention_mask[
                row_index,
                :sequence_length,
            ] = True

            labels[row_index] = (
                instance.label_id
            )

            instance_ids.append(
                instance.instance_id
            )

        return NeuralBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            instance_ids=tuple(
                instance_ids
            ),
        )
