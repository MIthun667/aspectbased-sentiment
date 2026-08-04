from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from torch.utils.data import Dataset


HARM_TARGET_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EvidenceHarmTarget:
    schema_version: int

    instance_id: str
    sentence_id: str
    domain: str
    split: str

    selected_gold_probability: float
    empty_gold_probability: float
    utility_delta: float

    selected_prediction_id: int
    empty_prediction_id: int
    gold_label_id: int

    selected_correct: bool
    empty_correct: bool
    selected_causes_error: bool

    harm_threshold: float
    probability_harm: bool
    decisive_harm: bool
    harm_target: bool
    harm_severity: float

    original_compatibility_score: float | None
    original_gate_value: float | None

    source_experiment: str
    source_checkpoint: str
    source_checkpoint_epoch: int
    source_intervention_directory: str
    source_audit_file: str


def load_harm_target_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    source_path = Path(path)

    if not source_path.is_file():
        raise FileNotFoundError(
            "Harm-target JSONL file not found: "
            f"{source_path}"
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
                    f"{line_number}"
                ) from error

            if not isinstance(value, dict):
                raise TypeError(
                    "Each harm-target JSONL row "
                    "must be an object"
                )

            records.append(value)

    if not records:
        raise ValueError(
            "Harm-target JSONL contains no "
            f"records: {source_path}"
        )

    return records


def _require_string(
    record: Mapping[str, Any],
    field: str,
) -> str:
    value = record.get(field)

    if not isinstance(value, str):
        raise TypeError(
            f"{field} must be a string"
        )

    if not value.strip():
        raise ValueError(
            f"{field} must not be empty"
        )

    return value


def _require_bool(
    record: Mapping[str, Any],
    field: str,
) -> bool:
    value = record.get(field)

    if not isinstance(value, bool):
        raise TypeError(
            f"{field} must be a boolean"
        )

    return value


def _require_int(
    record: Mapping[str, Any],
    field: str,
) -> int:
    value = record.get(field)

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{field} must be an integer"
        )

    return value


def _require_float(
    record: Mapping[str, Any],
    field: str,
) -> float:
    value = record.get(field)

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{field} must be numeric"
        )

    return float(value)


def _optional_float(
    record: Mapping[str, Any],
    field: str,
) -> float | None:
    value = record.get(field)

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{field} must be numeric or null"
        )

    return float(value)


def parse_harm_target_record(
    record: Mapping[str, Any],
) -> EvidenceHarmTarget:
    schema_version = _require_int(
        record,
        "schema_version",
    )

    if (
        schema_version
        != HARM_TARGET_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported harm-target "
            f"schema_version: {schema_version}"
        )

    selected_gold_probability = (
        _require_float(
            record,
            "selected_gold_probability",
        )
    )

    empty_gold_probability = _require_float(
        record,
        "empty_gold_probability",
    )

    utility_delta = _require_float(
        record,
        "utility_delta",
    )

    expected_delta = (
        selected_gold_probability
        - empty_gold_probability
    )

    if abs(
        utility_delta - expected_delta
    ) > 1e-8:
        raise ValueError(
            "utility_delta is inconsistent "
            "with the stored probabilities"
        )

    harm_threshold = _require_float(
        record,
        "harm_threshold",
    )

    if harm_threshold < 0.0:
        raise ValueError(
            "harm_threshold must be "
            "non-negative"
        )

    selected_correct = _require_bool(
        record,
        "selected_correct",
    )

    empty_correct = _require_bool(
        record,
        "empty_correct",
    )

    selected_causes_error = _require_bool(
        record,
        "selected_causes_error",
    )

    expected_decisive = (
        not selected_correct
        and empty_correct
    )

    if (
        selected_causes_error
        != expected_decisive
    ):
        raise ValueError(
            "selected_causes_error is "
            "inconsistent"
        )

    probability_harm = _require_bool(
        record,
        "probability_harm",
    )

    expected_probability_harm = (
        utility_delta <= -harm_threshold
    )

    if (
        probability_harm
        != expected_probability_harm
    ):
        raise ValueError(
            "probability_harm is inconsistent"
        )

    decisive_harm = _require_bool(
        record,
        "decisive_harm",
    )

    if decisive_harm != expected_decisive:
        raise ValueError(
            "decisive_harm is inconsistent"
        )

    harm_target = _require_bool(
        record,
        "harm_target",
    )

    if harm_target != (
        probability_harm or decisive_harm
    ):
        raise ValueError(
            "harm_target is inconsistent"
        )

    harm_severity = _require_float(
        record,
        "harm_severity",
    )

    expected_severity = max(
        empty_gold_probability
        - selected_gold_probability,
        0.0,
    )

    if abs(
        harm_severity - expected_severity
    ) > 1e-8:
        raise ValueError(
            "harm_severity is inconsistent"
        )

    for probability_name, probability in (
        (
            "selected_gold_probability",
            selected_gold_probability,
        ),
        (
            "empty_gold_probability",
            empty_gold_probability,
        ),
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"{probability_name} must "
                "be in [0, 1]"
            )

    return EvidenceHarmTarget(
        schema_version=schema_version,
        instance_id=_require_string(
            record,
            "instance_id",
        ),
        sentence_id=_require_string(
            record,
            "sentence_id",
        ),
        domain=_require_string(
            record,
            "domain",
        ),
        split=_require_string(
            record,
            "split",
        ),
        selected_gold_probability=(
            selected_gold_probability
        ),
        empty_gold_probability=(
            empty_gold_probability
        ),
        utility_delta=utility_delta,
        selected_prediction_id=_require_int(
            record,
            "selected_prediction_id",
        ),
        empty_prediction_id=_require_int(
            record,
            "empty_prediction_id",
        ),
        gold_label_id=_require_int(
            record,
            "gold_label_id",
        ),
        selected_correct=selected_correct,
        empty_correct=empty_correct,
        selected_causes_error=(
            selected_causes_error
        ),
        harm_threshold=harm_threshold,
        probability_harm=probability_harm,
        decisive_harm=decisive_harm,
        harm_target=harm_target,
        harm_severity=harm_severity,
        original_compatibility_score=(
            _optional_float(
                record,
                "original_compatibility_score",
            )
        ),
        original_gate_value=_optional_float(
            record,
            "original_gate_value",
        ),
        source_experiment=_require_string(
            record,
            "source_experiment",
        ),
        source_checkpoint=_require_string(
            record,
            "source_checkpoint",
        ),
        source_checkpoint_epoch=_require_int(
            record,
            "source_checkpoint_epoch",
        ),
        source_intervention_directory=(
            _require_string(
                record,
                "source_intervention_directory",
            )
        ),
        source_audit_file=_require_string(
            record,
            "source_audit_file",
        ),
    )


def index_harm_targets(
    records: Sequence[
        EvidenceHarmTarget
    ],
) -> dict[str, EvidenceHarmTarget]:
    indexed: dict[
        str,
        EvidenceHarmTarget,
    ] = {}

    for record in records:
        if record.instance_id in indexed:
            raise ValueError(
                "Duplicate harm-target "
                f"instance_id: {record.instance_id}"
            )

        indexed[record.instance_id] = record

    return indexed


class EvidenceHarmTargetDataset(
    Dataset[EvidenceHarmTarget]
):
    def __init__(
        self,
        records: Sequence[
            Mapping[str, Any]
            | EvidenceHarmTarget
        ],
    ) -> None:
        parsed: list[
            EvidenceHarmTarget
        ] = []

        for record in records:
            if isinstance(
                record,
                EvidenceHarmTarget,
            ):
                parsed.append(record)
            else:
                parsed.append(
                    parse_harm_target_record(
                        record
                    )
                )

        if not parsed:
            raise ValueError(
                "EvidenceHarmTargetDataset "
                "requires at least one record"
            )

        index_harm_targets(parsed)

        self._records = tuple(parsed)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
    ) -> EvidenceHarmTargetDataset:
        return cls(
            load_harm_target_jsonl(path)
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(
        self,
        index: int,
    ) -> EvidenceHarmTarget:
        return self._records[index]
