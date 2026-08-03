from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from torch.utils.data import Dataset

from .loader import load_canonical_split


@dataclass(frozen=True, slots=True)
class EvidenceAwareInstance:
    instance_id: str
    sentence_id: str
    domain: str
    split: str
    tokens: tuple[str, ...]
    aspect_text: str
    aspect_start: int
    aspect_end: int
    polarity: str
    label_id: int
    selected_evidence_indices: tuple[int, ...]
    selected_evidence_tokens: tuple[str, ...]
    evidence_mask: tuple[bool, ...]
    evidence_is_empty: bool
    candidate_count: int
    selection_config: dict[str, object]



def load_evidence_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    source_path = Path(path)

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Evidence JSONL file not found: {source_path}"
        )

    records: list[dict[str, Any]] = []

    with source_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON in "
                    f"{source_path} at line "
                    f"{line_number}: {error}"
                ) from error

            if not isinstance(value, dict):
                raise TypeError(
                    "Each evidence JSONL row must "
                    "be an object: "
                    f"{source_path} line {line_number}"
                )

            records.append(value)

    if not records:
        raise ValueError(
            "Evidence JSONL file contains "
            f"no records: {source_path}"
        )

    return records


def evidence_split_path(
    evidence_root: str | Path,
    *,
    domain: str,
    split: str,
) -> Path:
    if domain not in {
        "laptops",
        "restaurants",
        "tweets",
    }:
        raise ValueError(
            f"Unsupported domain: {domain!r}"
        )

    if split not in {
        "train",
        "validation",
        "test",
        "train_full",
    }:
        raise ValueError(
            f"Unsupported split: {split!r}"
        )

    return (
        Path(evidence_root)
        / domain
        / f"{split}.jsonl"
    )


def _index_unique_records(
    records: Sequence[dict[str, Any]],
    *,
    record_type: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[
        str,
        dict[str, Any],
    ] = {}

    for record in records:
        instance_id = record.get(
            "instance_id"
        )

        if not isinstance(instance_id, str):
            raise TypeError(
                f"{record_type} instance_id "
                "must be a string"
            )

        if instance_id in indexed:
            raise ValueError(
                f"Duplicate {record_type} "
                f"instance_id: {instance_id!r}"
            )

        indexed[instance_id] = record

    return indexed


def _validate_string_field(
    canonical: dict[str, Any],
    evidence: dict[str, Any],
    *,
    field_name: str,
    instance_id: str,
) -> None:
    canonical_value = canonical.get(
        field_name
    )
    evidence_value = evidence.get(
        field_name
    )

    if canonical_value != evidence_value:
        raise ValueError(
            f"{field_name} mismatch for "
            f"{instance_id!r}: "
            f"{canonical_value!r} != "
            f"{evidence_value!r}"
        )


def _validate_evidence_record(
    canonical: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    instance_id = canonical[
        "instance_id"
    ]

    for field_name in (
        "sentence_id",
        "domain",
        "split",
        "aspect_text",
        "polarity",
    ):
        _validate_string_field(
            canonical,
            evidence,
            field_name=field_name,
            instance_id=instance_id,
        )

    if canonical["tokens"] != evidence.get(
        "tokens"
    ):
        raise ValueError(
            f"tokens mismatch for "
            f"{instance_id!r}"
        )

    for field_name in (
        "aspect_start",
        "aspect_end",
    ):
        if canonical[field_name] != (
            evidence.get(field_name)
        ):
            raise ValueError(
                f"{field_name} mismatch for "
                f"{instance_id!r}: "
                f"{canonical[field_name]!r} != "
                f"{evidence.get(field_name)!r}"
            )

    indices = evidence.get(
        "selected_evidence_indices"
    )

    if not isinstance(indices, list):
        raise TypeError(
            "selected_evidence_indices "
            "must be a list"
        )

    if not all(
        isinstance(index, int)
        for index in indices
    ):
        raise TypeError(
            "selected_evidence_indices "
            "must contain integers"
        )

    if indices != sorted(indices):
        raise ValueError(
            "selected_evidence_indices "
            f"must be sorted for {instance_id!r}"
        )

    if len(indices) != len(set(indices)):
        raise ValueError(
            "selected_evidence_indices "
            f"contain duplicates for "
            f"{instance_id!r}"
        )

    number_of_tokens = len(
        canonical["tokens"]
    )

    for index in indices:
        if not (
            0 <= index < number_of_tokens
        ):
            raise ValueError(
                "Evidence index out of range "
                f"for {instance_id!r}: {index}"
            )

    selected_tokens = evidence.get(
        "selected_evidence_tokens"
    )

    if not isinstance(
        selected_tokens,
        list,
    ):
        raise TypeError(
            "selected_evidence_tokens "
            "must be a list"
        )

    expected_tokens = [
        canonical["tokens"][index]
        for index in indices
    ]

    if selected_tokens != expected_tokens:
        raise ValueError(
            "selected_evidence_tokens do not "
            f"match indices for {instance_id!r}"
        )

    expected_empty = len(indices) == 0

    if evidence.get(
        "evidence_is_empty"
    ) is not expected_empty:
        raise ValueError(
            "evidence_is_empty does not "
            f"match indices for {instance_id!r}"
        )

    candidate_count = evidence.get(
        "candidate_count"
    )

    if not isinstance(
        candidate_count,
        int,
    ):
        raise TypeError(
            "candidate_count must be "
            "an integer"
        )

    if candidate_count < len(indices):
        raise ValueError(
            "candidate_count cannot be smaller "
            "than selected evidence count"
        )

    selection_config = evidence.get(
        "selection_config"
    )

    if not isinstance(
        selection_config,
        dict,
    ):
        raise TypeError(
            "selection_config must be a mapping"
        )


def join_canonical_and_evidence_records(
    canonical_records: Sequence[
        dict[str, Any]
    ],
    evidence_records: Sequence[
        dict[str, Any]
    ],
) -> tuple[EvidenceAwareInstance, ...]:
    if not canonical_records:
        raise ValueError(
            "Canonical records must not be empty"
        )

    if not evidence_records:
        raise ValueError(
            "Evidence records must not be empty"
        )

    canonical_by_id = (
        _index_unique_records(
            canonical_records,
            record_type="canonical",
        )
    )

    evidence_by_id = (
        _index_unique_records(
            evidence_records,
            record_type="evidence",
        )
    )

    canonical_ids = set(
        canonical_by_id
    )

    evidence_ids = set(
        evidence_by_id
    )

    missing_evidence = sorted(
        canonical_ids - evidence_ids
    )

    extra_evidence = sorted(
        evidence_ids - canonical_ids
    )

    if missing_evidence:
        raise ValueError(
            "Missing evidence records for "
            f"instance_ids: {missing_evidence}"
        )

    if extra_evidence:
        raise ValueError(
            "Evidence contains unknown "
            f"instance_ids: {extra_evidence}"
        )

    instances: list[
        EvidenceAwareInstance
    ] = []

    for canonical in canonical_records:
        instance_id = canonical[
            "instance_id"
        ]

        evidence = evidence_by_id[
            instance_id
        ]

        _validate_evidence_record(
            canonical,
            evidence,
        )

        tokens = tuple(
            str(token)
            for token in canonical[
                "tokens"
            ]
        )

        selected_indices = tuple(
            int(index)
            for index in evidence[
                "selected_evidence_indices"
            ]
        )

        selected_index_set = set(
            selected_indices
        )

        evidence_mask = tuple(
            index in selected_index_set
            for index in range(
                len(tokens)
            )
        )

        instances.append(
            EvidenceAwareInstance(
                instance_id=str(
                    canonical["instance_id"]
                ),
                sentence_id=str(
                    canonical["sentence_id"]
                ),
                domain=str(
                    canonical["domain"]
                ),
                split=str(
                    canonical["split"]
                ),
                tokens=tokens,
                aspect_text=str(
                    canonical["aspect_text"]
                ),
                aspect_start=int(
                    canonical["aspect_start"]
                ),
                aspect_end=int(
                    canonical["aspect_end"]
                ),
                polarity=str(
                    canonical["polarity"]
                ),
                label_id=int(
                    canonical["label_id"]
                ),
                selected_evidence_indices=(
                    selected_indices
                ),
                selected_evidence_tokens=tuple(
                    str(token)
                    for token in evidence[
                        "selected_evidence_tokens"
                    ]
                ),
                evidence_mask=evidence_mask,
                evidence_is_empty=bool(
                    evidence[
                        "evidence_is_empty"
                    ]
                ),
                candidate_count=int(
                    evidence[
                        "candidate_count"
                    ]
                ),
                selection_config=dict(
                    evidence[
                        "selection_config"
                    ]
                ),
            )
        )

    return tuple(instances)


class EvidenceAwareDataset(
    Dataset[EvidenceAwareInstance]
):
    def __init__(
        self,
        canonical_records: Sequence[
            dict[str, Any]
        ],
        evidence_records: Sequence[
            dict[str, Any]
        ],
    ) -> None:
        self._instances = (
            join_canonical_and_evidence_records(
                canonical_records,
                evidence_records,
            )
        )

    @classmethod
    def from_split(
        cls,
        *,
        processed_root: str | Path,
        evidence_root: str | Path,
        domain: str,
        split: str,
    ) -> EvidenceAwareDataset:
        canonical_records = (
            load_canonical_split(
                processed_root,
                domain=domain,
                split=split,
            )
        )

        evidence_path = (
            evidence_split_path(
                evidence_root,
                domain=domain,
                split=split,
            )
        )

        evidence_records = (
            load_evidence_jsonl(
                evidence_path
            )
        )

        return cls(
            canonical_records,
            evidence_records,
        )

    def __len__(self) -> int:
        return len(self._instances)

    def __getitem__(
        self,
        index: int,
    ) -> EvidenceAwareInstance:
        return self._instances[index]
