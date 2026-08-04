from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterator, Sequence

from torch.utils.data import Dataset

from src.aspect_sentiment.data import (
    load_canonical_split,
)

from .schema import LLMABSAInstance


class LLMABSADataset(
    Dataset[LLMABSAInstance]
):
    def __init__(
        self,
        instances: Sequence[
            LLMABSAInstance
        ],
    ) -> None:
        if not instances:
            raise ValueError(
                "instances must not be empty"
            )

        validated = []

        for instance in instances:
            instance.validate()
            validated.append(instance)

        duplicate_ids = sorted(
            instance_id
            for instance_id, count in Counter(
                instance.instance_id
                for instance in validated
            ).items()
            if count != 1
        )

        if duplicate_ids:
            raise ValueError(
                "Duplicate instance IDs: "
                f"{duplicate_ids[:10]}"
            )

        self._instances = tuple(
            validated
        )

    @classmethod
    def from_split(
        cls,
        processed_root: str | Path,
        *,
        domain: str,
        split: str,
    ) -> "LLMABSADataset":
        canonical_records = (
            load_canonical_split(
                processed_root,
                domain=domain,
                split=split,
            )
        )

        instances = [
            LLMABSAInstance.from_canonical_record(
                record
            )
            for record in canonical_records
        ]

        return cls(instances)

    def __len__(self) -> int:
        return len(self._instances)

    def __getitem__(
        self,
        index: int,
    ) -> LLMABSAInstance:
        return self._instances[index]

    def __iter__(
        self,
    ) -> Iterator[LLMABSAInstance]:
        return iter(self._instances)

    @property
    def instances(
        self,
    ) -> tuple[LLMABSAInstance, ...]:
        return self._instances

    def instance_ids(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            instance.instance_id
            for instance in self._instances
        )

    def multi_aspect_count(
        self,
    ) -> int:
        return sum(
            instance.is_multi_aspect
            for instance in self._instances
        )
