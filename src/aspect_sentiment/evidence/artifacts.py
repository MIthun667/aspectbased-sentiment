from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .candidates import (
    ScoredEvidenceCandidate,
    score_candidates_for_record,
    select_clause_aware_weak_evidence_indices,
)
from .clauses import evaluate_clause_compatibility


@dataclass(frozen=True, slots=True)
class WeakEvidenceConfig:
    schema_version: int = 1
    maximum_dependency_distance: int = 3
    minimum_score: float = 2.5
    maximum_tokens: int = 3
    allow_ties: bool = False
    exclude_competing_predicate_clauses: bool = True

    def validate(self) -> None:
        if self.schema_version <= 0:
            raise ValueError(
                "schema_version must be positive"
            )

        if self.maximum_dependency_distance < 0:
            raise ValueError(
                "maximum_dependency_distance "
                "must be non-negative"
            )

        if self.maximum_tokens <= 0:
            raise ValueError(
                "maximum_tokens must be positive"
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


def utc_timestamp() -> str:
    return datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    ).isoformat()


def canonical_json_dumps(
    value: object,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def group_records_by_sentence(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        sentence_id = str(
            record["sentence_id"]
        )

        grouped[sentence_id].append(
            dict(record)
        )

    for sentence_records in grouped.values():
        sentence_records.sort(
            key=lambda record: (
                int(record["aspect_start"]),
                int(record["aspect_end"]),
                str(record["instance_id"]),
            )
        )

    return dict(grouped)


def scored_candidate_to_artifact(
    record: Mapping[str, Any],
    item: ScoredEvidenceCandidate,
    *,
    selected_indices: set[int],
) -> dict[str, object]:
    token_index = item.candidate.token_index

    clause = evaluate_clause_compatibility(
        dict(record),
        candidate_index=token_index,
    )

    return {
        "candidate": item.candidate.to_dict(),
        "attribution": item.attribution.to_dict(),
        "score": item.score.to_dict(),
        "clause": clause.to_dict(),
        "is_selected": (
            token_index in selected_indices
        ),
    }


def build_weak_evidence_record(
    record: Mapping[str, Any],
    *,
    sentence_records: Sequence[
        Mapping[str, Any]
    ],
    config: WeakEvidenceConfig,
) -> dict[str, object]:
    config.validate()

    materialized_record = dict(record)
    materialized_sentence_records = [
        dict(item)
        for item in sentence_records
    ]

    scored = score_candidates_for_record(
        materialized_record,
        sentence_records=(
            materialized_sentence_records
        ),
        maximum_dependency_distance=(
            config.maximum_dependency_distance
        ),
    )

    selected_indices = (
        select_clause_aware_weak_evidence_indices(
            materialized_record,
            scored,
            minimum_score=config.minimum_score,
            maximum_tokens=config.maximum_tokens,
            allow_ties=config.allow_ties,
            exclude_competing_predicate_clauses=(
                config
                .exclude_competing_predicate_clauses
            ),
        )
    )

    selected_set = set(selected_indices)
    tokens = list(materialized_record["tokens"])

    selected_tokens = [
        tokens[index]
        for index in selected_indices
    ]

    candidates = [
        scored_candidate_to_artifact(
            materialized_record,
            item,
            selected_indices=selected_set,
        )
        for item in scored
    ]

    return {
        "schema_version": config.schema_version,
        "instance_id": str(
            materialized_record["instance_id"]
        ),
        "sentence_id": str(
            materialized_record["sentence_id"]
        ),
        "domain": materialized_record.get(
            "domain"
        ),
        "split": materialized_record.get(
            "split"
        ),
        "aspect_text": str(
            materialized_record["aspect_text"]
        ),
        "aspect_start": int(
            materialized_record["aspect_start"]
        ),
        "aspect_end": int(
            materialized_record["aspect_end"]
        ),
        "polarity": str(
            materialized_record["polarity"]
        ),
        "is_multi_aspect": bool(
            materialized_record[
                "is_multi_aspect"
            ]
        ),
        "tokens": tokens,
        "selected_evidence_indices": list(
            selected_indices
        ),
        "selected_evidence_tokens": (
            selected_tokens
        ),
        "evidence_is_empty": (
            len(selected_indices) == 0
        ),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selection_config": config.to_dict(),
    }


def build_weak_evidence_records(
    records: Sequence[Mapping[str, Any]],
    *,
    config: WeakEvidenceConfig,
) -> list[dict[str, object]]:
    config.validate()

    grouped = group_records_by_sentence(
        records
    )

    output: list[dict[str, object]] = []

    for record in records:
        sentence_id = str(
            record["sentence_id"]
        )

        output.append(
            build_weak_evidence_record(
                record,
                sentence_records=grouped[
                    sentence_id
                ],
                config=config,
            )
        )

    return output


def summarize_weak_evidence_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    record_count = len(records)

    selected_counts = [
        len(
            record[
                "selected_evidence_indices"
            ]
        )
        for record in records
    ]

    candidate_counts = [
        int(record["candidate_count"])
        for record in records
    ]

    empty_count = sum(
        count == 0
        for count in selected_counts
    )

    polarity_counts = Counter(
        str(record["polarity"])
        for record in records
    )

    selected_by_polarity: Counter[str] = (
        Counter()
    )

    for record in records:
        selected_by_polarity[
            str(record["polarity"])
        ] += len(
            record[
                "selected_evidence_indices"
            ]
        )

    mean_selected = (
        sum(selected_counts) / record_count
        if record_count
        else 0.0
    )

    mean_candidates = (
        sum(candidate_counts) / record_count
        if record_count
        else 0.0
    )

    return {
        "record_count": record_count,
        "empty_evidence_count": empty_count,
        "coverage": (
            1.0 - empty_count / record_count
            if record_count
            else 0.0
        ),
        "mean_selected_tokens": mean_selected,
        "maximum_selected_tokens": (
            max(selected_counts)
            if selected_counts
            else 0
        ),
        "mean_candidates": mean_candidates,
        "maximum_candidates": (
            max(candidate_counts)
            if candidate_counts
            else 0
        ),
        "polarity_counts": dict(
            sorted(polarity_counts.items())
        ),
        "selected_tokens_by_polarity": dict(
            sorted(
                selected_by_polarity.items()
            )
        ),
    }


def write_jsonl(
    records: Iterable[Mapping[str, Any]],
    *,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for record in records:
            handle.write(
                canonical_json_dumps(record)
            )
            handle.write("\n")


def write_json(
    value: Mapping[str, Any],
    *,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )

    output_path.write_text(
        text + "\n",
        encoding="utf-8",
    )


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def build_split_manifest(
    *,
    source_path: Path,
    output_path: Path,
    records: Sequence[Mapping[str, Any]],
    config: WeakEvidenceConfig,
    generated_at_utc: str | None = None,
) -> dict[str, object]:
    config.validate()

    return {
        "schema_version": config.schema_version,
        "generated_at_utc": (
            generated_at_utc
            if generated_at_utc is not None
            else utc_timestamp()
        ),
        "source_path": source_path.as_posix(),
        "source_sha256": sha256_file(
            source_path
        ),
        "output_path": output_path.as_posix(),
        "output_sha256": sha256_file(
            output_path
        ),
        "configuration": config.to_dict(),
        "statistics": (
            summarize_weak_evidence_records(
                records
            )
        ),
    }


def load_weak_evidence_config(
    path: Path,
) -> WeakEvidenceConfig:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "PyYAML is required to load evidence configuration"
        ) from error

    raw = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(raw, dict):
        raise ValueError(
            "Evidence configuration must be a mapping"
        )

    candidate_generation = raw.get(
        "candidate_generation",
        {},
    )

    selection = raw.get(
        "selection",
        {},
    )

    if not isinstance(
        candidate_generation,
        dict,
    ):
        raise ValueError(
            "candidate_generation must be a mapping"
        )

    if not isinstance(selection, dict):
        raise ValueError(
            "selection must be a mapping"
        )

    config = WeakEvidenceConfig(
        schema_version=int(
            raw.get("schema_version", 1)
        ),
        maximum_dependency_distance=int(
            candidate_generation.get(
                "maximum_dependency_distance",
                3,
            )
        ),
        minimum_score=float(
            selection.get(
                "minimum_score",
                2.5,
            )
        ),
        maximum_tokens=int(
            selection.get(
                "maximum_tokens",
                3,
            )
        ),
        allow_ties=bool(
            selection.get(
                "allow_ties",
                False,
            )
        ),
        exclude_competing_predicate_clauses=bool(
            selection.get(
                "exclude_competing_predicate_clauses",
                True,
            )
        ),
    )

    config.validate()
    return config


def build_evidence_split(
    *,
    source_path: Path,
    output_path: Path,
    manifest_path: Path,
    config: WeakEvidenceConfig,
    generated_at_utc: str | None = None,
) -> dict[str, object]:
    source_records = read_jsonl(
        source_path
    )

    artifacts = (
        build_weak_evidence_records(
            source_records,
            config=config,
        )
    )

    write_jsonl(
        artifacts,
        output_path=output_path,
    )

    manifest = build_split_manifest(
        source_path=source_path,
        output_path=output_path,
        records=artifacts,
        config=config,
        generated_at_utc=generated_at_utc,
    )

    write_json(
        manifest,
        output_path=manifest_path,
    )

    return manifest


def build_domain_manifest(
    *,
    domain: str,
    split_manifests: Mapping[
        str,
        Mapping[str, Any],
    ],
    config: WeakEvidenceConfig,
    output_path: Path,
    generated_at_utc: str | None = None,
) -> dict[str, object]:
    config.validate()

    ordered_splits = {
        split: dict(split_manifests[split])
        for split in sorted(
            split_manifests
        )
    }

    total_records = sum(
        int(
            manifest["statistics"][
                "record_count"
            ]
        )
        for manifest in ordered_splits.values()
    )

    total_empty = sum(
        int(
            manifest["statistics"][
                "empty_evidence_count"
            ]
        )
        for manifest in ordered_splits.values()
    )

    manifest = {
        "schema_version": config.schema_version,
        "domain": domain,
        "generated_at_utc": (
            generated_at_utc
            if generated_at_utc is not None
            else utc_timestamp()
        ),
        "configuration": config.to_dict(),
        "splits": ordered_splits,
        "aggregate_statistics": {
            "record_count": total_records,
            "empty_evidence_count": total_empty,
            "coverage": (
                1.0
                - total_empty / total_records
                if total_records
                else 0.0
            ),
        },
    }

    write_json(
        manifest,
        output_path=output_path,
    )

    return manifest

