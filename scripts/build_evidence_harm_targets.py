from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
DEFAULT_HARM_THRESHOLD = 0.02


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_json(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return value


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"JSONL file not found: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open(
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            value = json.loads(stripped)

            if not isinstance(value, dict):
                raise TypeError(
                    "Audit JSONL row must be "
                    "an object at line "
                    f"{line_number}"
                )

            records.append(value)

    if not records:
        raise ValueError(
            f"No audit records found: {path}"
        )

    return records


def require_bool(
    record: Mapping[str, Any],
    field: str,
) -> bool:
    value = record.get(field)

    if not isinstance(value, bool):
        raise TypeError(
            f"{field} must be boolean"
        )

    return value


def require_int(
    record: Mapping[str, Any],
    field: str,
) -> int:
    value = record.get(field)

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{field} must be integer"
        )

    return value


def require_float(
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


def require_string(
    record: Mapping[str, Any],
    field: str,
) -> str:
    value = record.get(field)

    if not isinstance(value, str):
        raise TypeError(
            f"{field} must be string"
        )

    if not value.strip():
        raise ValueError(
            f"{field} must not be empty"
        )

    return value


def optional_float(
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


def build_harm_target_record(
    audit_record: Mapping[str, Any],
    *,
    harm_threshold: float,
    source_experiment: str,
    source_checkpoint: str,
    source_checkpoint_epoch: int,
    source_intervention_directory: str,
    source_audit_file: str,
) -> dict[str, Any]:
    if harm_threshold < 0.0:
        raise ValueError(
            "harm_threshold must be "
            "non-negative"
        )

    selected_gold_probability = (
        require_float(
            audit_record,
            "selected_gold_probability",
        )
    )

    empty_gold_probability = (
        require_float(
            audit_record,
            "empty_gold_probability",
        )
    )

    utility_delta = (
        selected_gold_probability
        - empty_gold_probability
    )

    selected_correct = require_bool(
        audit_record,
        "selected_correct",
    )

    empty_correct = require_bool(
        audit_record,
        "empty_correct",
    )

    selected_causes_error = (
        not selected_correct
        and empty_correct
    )

    probability_harm = (
        utility_delta <= -harm_threshold
    )

    decisive_harm = (
        selected_causes_error
    )

    harm_target = (
        probability_harm
        or decisive_harm
    )

    harm_severity = max(
        empty_gold_probability
        - selected_gold_probability,
        0.0,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "instance_id": require_string(
            audit_record,
            "instance_id",
        ),
        "sentence_id": require_string(
            audit_record,
            "sentence_id",
        ),
        "domain": require_string(
            audit_record,
            "domain",
        ),
        "split": require_string(
            audit_record,
            "split",
        ),
        "selected_gold_probability": (
            selected_gold_probability
        ),
        "empty_gold_probability": (
            empty_gold_probability
        ),
        "utility_delta": utility_delta,
        "selected_prediction_id": (
            require_int(
                audit_record,
                "selected_prediction_id",
            )
        ),
        "empty_prediction_id": (
            require_int(
                audit_record,
                "empty_prediction_id",
            )
        ),
        "gold_label_id": require_int(
            audit_record,
            "gold_label_id",
        ),
        "selected_correct": (
            selected_correct
        ),
        "empty_correct": empty_correct,
        "selected_causes_error": (
            selected_causes_error
        ),
        "harm_threshold": (
            harm_threshold
        ),
        "probability_harm": (
            probability_harm
        ),
        "decisive_harm": decisive_harm,
        "harm_target": harm_target,
        "harm_severity": harm_severity,
        "original_compatibility_score": (
            optional_float(
                audit_record,
                "original_compatibility_score",
            )
        ),
        "original_gate_value": (
            optional_float(
                audit_record,
                "original_gate_value",
            )
        ),
        "source_experiment": (
            source_experiment
        ),
        "source_checkpoint": (
            source_checkpoint
        ),
        "source_checkpoint_epoch": (
            source_checkpoint_epoch
        ),
        "source_intervention_directory": (
            source_intervention_directory
        ),
        "source_audit_file": (
            source_audit_file
        ),
    }


def build_harm_target_records(
    audit_records: Iterable[
        Mapping[str, Any]
    ],
    *,
    harm_threshold: float,
    source_experiment: str,
    source_checkpoint: str,
    source_checkpoint_epoch: int,
    source_intervention_directory: str,
    source_audit_file: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    for audit_record in audit_records:
        record = build_harm_target_record(
            audit_record,
            harm_threshold=harm_threshold,
            source_experiment=(
                source_experiment
            ),
            source_checkpoint=(
                source_checkpoint
            ),
            source_checkpoint_epoch=(
                source_checkpoint_epoch
            ),
            source_intervention_directory=(
                source_intervention_directory
            ),
            source_audit_file=(
                source_audit_file
            ),
        )

        instance_id = str(
            record["instance_id"]
        )

        if instance_id in seen:
            raise ValueError(
                "Duplicate harm-target "
                f"instance_id: {instance_id}"
            )

        seen.add(instance_id)
        output.append(record)

    if not output:
        raise ValueError(
            "No harm-target records built"
        )

    return output


def summarize_records(
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    target_counts = Counter(
        bool(record["harm_target"])
        for record in records
    )

    label_counts = Counter(
        int(record["gold_label_id"])
        for record in records
    )

    positive_records = [
        record
        for record in records
        if bool(record["harm_target"])
    ]

    severity_values = [
        float(record["harm_severity"])
        for record in records
    ]

    positive_severity_values = [
        float(record["harm_severity"])
        for record in positive_records
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "record_count": len(records),
        "harm_positive_count": (
            target_counts[True]
        ),
        "harm_negative_count": (
            target_counts[False]
        ),
        "harm_positive_rate": (
            target_counts[True]
            / len(records)
        ),
        "probability_harm_count": sum(
            bool(record["probability_harm"])
            for record in records
        ),
        "decisive_harm_count": sum(
            bool(record["decisive_harm"])
            for record in records
        ),
        "gold_label_counts": {
            str(label): count
            for label, count in sorted(
                label_counts.items()
            )
        },
        "mean_harm_severity": (
            sum(severity_values)
            / len(severity_values)
        ),
        "mean_positive_harm_severity": (
            sum(positive_severity_values)
            / len(
                positive_severity_values
            )
            if positive_severity_values
            else None
        ),
        "maximum_harm_severity": max(
            severity_values
        ),
    }


def write_jsonl(
    records: Iterable[
        Mapping[str, Any]
    ],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for record in records:
            handle.write(
                canonical_json_dumps(record)
            )
            handle.write("\n")


def write_csv(
    records: list[Mapping[str, Any]],
    path: Path,
) -> None:
    if not records:
        raise ValueError(
            "Cannot write empty CSV"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        records[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(record)


def write_json(
    payload: Mapping[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_creation_command() -> str:
    return " ".join(
        shlex.quote(argument)
        for argument in sys.argv
    )


def resolve_checkpoint_path(
    experiment_directory: Path,
) -> Path:
    candidates = (
        experiment_directory
        / "checkpoints"
        / "best.pt",
        experiment_directory
        / "best.pt",
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate best checkpoint "
        f"under {experiment_directory}"
    )


def run_builder(
    *,
    intervention_directory: Path,
    experiment_directory: Path,
    output_directory: Path,
    harm_threshold: float,
    overwrite: bool,
) -> dict[str, Path]:
    intervention_directory = (
        intervention_directory.resolve()
    )

    experiment_directory = (
        experiment_directory.resolve()
    )

    output_directory = (
        output_directory.resolve()
    )

    if (
        output_directory.exists()
        and not overwrite
    ):
        raise FileExistsError(
            "Harm-target output directory "
            f"already exists: {output_directory}"
        )

    if output_directory.exists():
        shutil.rmtree(
            output_directory
        )

    audit_path = (
        intervention_directory
        / "utility_audit"
        / "evidence_utility_audit.jsonl"
    )

    intervention_summary_path = (
        intervention_directory
        / "summary.json"
    )

    intervention_summary = read_json(
        intervention_summary_path
    )

    checkpoint_epoch = int(
        intervention_summary[
            "checkpoint_epoch"
        ]
    )

    checkpoint_path = (
        resolve_checkpoint_path(
            experiment_directory
        )
    )

    records = build_harm_target_records(
        read_jsonl(audit_path),
        harm_threshold=harm_threshold,
        source_experiment=str(
            experiment_directory
        ),
        source_checkpoint=str(
            checkpoint_path
        ),
        source_checkpoint_epoch=(
            checkpoint_epoch
        ),
        source_intervention_directory=(
            str(intervention_directory)
        ),
        source_audit_file=str(
            audit_path.resolve()
        ),
    )

    jsonl_path = (
        output_directory
        / "harm_targets.jsonl"
    )

    csv_path = (
        output_directory
        / "harm_targets.csv"
    )

    summary_path = (
        output_directory
        / "summary.json"
    )

    manifest_path = (
        output_directory
        / "manifest.json"
    )

    write_jsonl(
        records,
        jsonl_path,
    )

    write_csv(
        records,
        csv_path,
    )

    summary = summarize_records(
        records
    )

    write_json(
        summary,
        summary_path,
    )

    manifest = {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "created_at": utc_timestamp(),
        "creation_command": (
            build_creation_command()
        ),
        "harm_threshold": (
            harm_threshold
        ),
        "source_experiment": str(
            experiment_directory
        ),
        "source_checkpoint": str(
            checkpoint_path
        ),
        "source_checkpoint_sha256": (
            sha256_file(checkpoint_path)
        ),
        "source_checkpoint_epoch": (
            checkpoint_epoch
        ),
        "source_checkpoint_type": (
            intervention_summary[
                "checkpoint_type"
            ]
        ),
        "source_intervention_directory": (
            str(intervention_directory)
        ),
        "source_intervention_seed": (
            intervention_summary[
                "intervention_seed"
            ]
        ),
        "source_audit_file": str(
            audit_path.resolve()
        ),
        "source_audit_sha256": (
            sha256_file(audit_path)
        ),
        "output_file": str(
            jsonl_path
        ),
        "output_sha256": (
            sha256_file(jsonl_path)
        ),
        **summary,
    }

    write_json(
        manifest,
        manifest_path,
    )

    return {
        "jsonl": jsonl_path,
        "csv": csv_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build detached evidence-harm "
            "targets from a utility audit."
        )
    )

    parser.add_argument(
        "intervention_directory",
        type=Path,
    )

    parser.add_argument(
        "--experiment-directory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--harm-threshold",
        type=float,
        default=DEFAULT_HARM_THRESHOLD,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser


def main() -> None:
    arguments = (
        build_argument_parser()
        .parse_args()
    )

    paths = run_builder(
        intervention_directory=(
            arguments.intervention_directory
        ),
        experiment_directory=(
            arguments.experiment_directory
        ),
        output_directory=(
            arguments.output_directory
        ),
        harm_threshold=(
            arguments.harm_threshold
        ),
        overwrite=arguments.overwrite,
    )

    print("=" * 88)
    print(
        "DETACHED EVIDENCE HARM TARGETS"
    )
    print("=" * 88)

    for name, path in paths.items():
        print(
            f"{name.capitalize():<10} "
            f"{path}"
        )


if __name__ == "__main__":
    main()
