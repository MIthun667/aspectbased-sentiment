from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPOSITORY_ROOT),
    )


SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    ).isoformat()


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


def load_json(path: Path) -> dict[str, Any]:
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


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"JSONL file not found: {path}"
        )

    records = []

    for line_number, line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        value = json.loads(line)

        if not isinstance(value, dict):
            raise TypeError(
                "Expected JSON object at "
                f"{path}:{line_number}"
            )

        records.append(value)

    if not records:
        raise ValueError(
            f"JSONL file is empty: {path}"
        )

    return records


def write_json(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    records: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")


def write_csv(
    path: Path,
    records: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    if not records:
        raise ValueError(
            "CSV records must not be empty"
        )

    fieldnames = sorted(
        {
            key
            for record in records
            for key in record
        }
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


def validate_fold_records(
    *,
    fold_index: int,
    records: Sequence[
        Mapping[str, Any]
    ],
    expected_instance_ids: set[str],
) -> list[dict[str, Any]]:
    actual_ids = []

    output = []

    for source_record in records:
        record = dict(source_record)

        instance_id = record.get(
            "instance_id"
        )

        if (
            not isinstance(
                instance_id,
                str,
            )
            or not instance_id
        ):
            raise ValueError(
                "Every record must contain "
                "a non-empty instance_id"
            )

        actual_ids.append(instance_id)

        if record.get("split") != "train":
            raise ValueError(
                "Cross-fitted harm targets "
                "must originate from split=train"
            )

        record["source_fold"] = fold_index
        record[
            "target_generation_mode"
        ] = "cross_fitted_out_of_fold"

        output.append(record)

    duplicates = sorted(
        instance_id
        for instance_id, count in Counter(
            actual_ids
        ).items()
        if count != 1
    )

    if duplicates:
        raise ValueError(
            "Fold contains duplicate "
            f"instance IDs: {duplicates[:10]}"
        )

    actual_set = set(actual_ids)

    missing = sorted(
        expected_instance_ids
        - actual_set
    )

    extra = sorted(
        actual_set
        - expected_instance_ids
    )

    if missing:
        raise ValueError(
            "Fold is missing expected "
            f"instance IDs: {missing[:10]}"
        )

    if extra:
        raise ValueError(
            "Fold contains unexpected "
            f"instance IDs: {extra[:10]}"
        )

    return output


def summarize_records(
    records: Sequence[
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    positive = sum(
        bool(record["harm_target"])
        for record in records
    )

    negative = len(records) - positive

    probability_harm = sum(
        bool(record["probability_harm"])
        for record in records
    )

    decisive_harm = sum(
        bool(record["decisive_harm"])
        for record in records
    )

    selected_causes_error = sum(
        bool(
            record.get(
                "selected_causes_error",
                False,
            )
        )
        for record in records
    )

    severities = [
        float(record["harm_severity"])
        for record in records
    ]

    positive_severities = [
        float(record["harm_severity"])
        for record in records
        if bool(record["harm_target"])
    ]

    fold_counts = Counter(
        int(record["source_fold"])
        for record in records
    )

    fold_positive_counts = Counter(
        int(record["source_fold"])
        for record in records
        if bool(record["harm_target"])
    )

    return {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "record_count": len(records),
        "harm_positive_count": positive,
        "harm_negative_count": negative,
        "harm_positive_rate": (
            positive / len(records)
        ),
        "negative_positive_ratio": (
            negative / positive
            if positive
            else None
        ),
        "probability_harm_count": (
            probability_harm
        ),
        "decisive_harm_count": (
            decisive_harm
        ),
        "selected_causes_error_count": (
            selected_causes_error
        ),
        "mean_harm_severity": (
            sum(severities)
            / len(severities)
        ),
        "mean_positive_harm_severity": (
            sum(positive_severities)
            / len(positive_severities)
            if positive_severities
            else 0.0
        ),
        "maximum_harm_severity": max(
            severities
        ),
        "fold_record_counts": {
            str(key): value
            for key, value in sorted(
                fold_counts.items()
            )
        },
        "fold_positive_counts": {
            str(key): value
            for key, value in sorted(
                fold_positive_counts.items()
            )
        },
    }


def merge_crossfit_targets(
    *,
    crossfit_manifest_path: Path,
    fold_directories: Sequence[Path],
    output_directory: Path,
    overwrite: bool,
) -> Path:
    crossfit_manifest_path = (
        crossfit_manifest_path.resolve()
    )

    crossfit_manifest = load_json(
        crossfit_manifest_path
    )

    folds = crossfit_manifest.get(
        "folds"
    )

    if not isinstance(folds, list):
        raise TypeError(
            "Cross-fit manifest folds "
            "must be a list"
        )

    expected_by_fold = {
        int(fold["fold_index"]): set(
            fold["heldout_instance_ids"]
        )
        for fold in folds
    }

    if len(fold_directories) != len(
        expected_by_fold
    ):
        raise ValueError(
            "Number of fold directories "
            "does not match manifest folds"
        )

    output_directory = (
        output_directory.resolve()
    )

    if output_directory.exists():
        if not overwrite:
            raise FileExistsError(
                "Output directory already "
                f"exists: {output_directory}"
            )

        shutil.rmtree(
            output_directory
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged_records = []
    source_payloads = []

    for fold_index, directory in enumerate(
        fold_directories
    ):
        directory = directory.resolve()

        if fold_index not in expected_by_fold:
            raise ValueError(
                "Unexpected fold index: "
                f"{fold_index}"
            )

        target_path = (
            directory
            / "harm_targets.jsonl"
        )

        manifest_path = (
            directory
            / "manifest.json"
        )

        records = load_jsonl(
            target_path
        )

        validated = validate_fold_records(
            fold_index=fold_index,
            records=records,
            expected_instance_ids=(
                expected_by_fold[
                    fold_index
                ]
            ),
        )

        merged_records.extend(validated)

        source_payloads.append(
            {
                "fold_index": fold_index,
                "directory": str(
                    directory
                ),
                "target_file": str(
                    target_path
                ),
                "target_sha256": (
                    sha256_file(
                        target_path
                    )
                ),
                "manifest_file": str(
                    manifest_path
                ),
                "manifest_sha256": (
                    sha256_file(
                        manifest_path
                    )
                ),
                "record_count": len(
                    validated
                ),
            }
        )

    merged_records.sort(
        key=lambda record: str(
            record["instance_id"]
        )
    )

    occurrences = Counter(
        str(record["instance_id"])
        for record in merged_records
    )

    duplicates = sorted(
        instance_id
        for instance_id, count in (
            occurrences.items()
        )
        if count != 1
    )

    if duplicates:
        raise ValueError(
            "Merged targets contain duplicate "
            f"instance IDs: {duplicates[:10]}"
        )

    expected_global_count = int(
        crossfit_manifest[
            "global_statistics"
        ]["instance_count"]
    )

    if len(merged_records) != (
        expected_global_count
    ):
        raise ValueError(
            "Merged target count does not "
            "match cross-fit manifest"
        )

    target_path = (
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
        target_path,
        merged_records,
    )

    write_csv(
        csv_path,
        merged_records,
    )

    summary = summarize_records(
        merged_records
    )

    write_json(
        summary_path,
        summary,
    )

    manifest = {
        **summary,
        "created_at": utc_now_iso(),
        "target_generation_mode": (
            "cross_fitted_out_of_fold"
        ),
        "crossfit_manifest_file": str(
            crossfit_manifest_path
        ),
        "crossfit_manifest_sha256": (
            sha256_file(
                crossfit_manifest_path
            )
        ),
        "output_file": str(
            target_path
        ),
        "output_sha256": sha256_file(
            target_path
        ),
        "source_folds": source_payloads,
    }

    write_json(
        manifest_path,
        manifest,
    )

    return manifest_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge validated out-of-fold "
            "evidence harm targets."
        )
    )

    parser.add_argument(
        "--crossfit-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--fold-directory",
        type=Path,
        action="append",
        required=True,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    manifest_path = (
        merge_crossfit_targets(
            crossfit_manifest_path=(
                arguments.crossfit_manifest
            ),
            fold_directories=(
                arguments.fold_directory
            ),
            output_directory=(
                arguments.output_directory
            ),
            overwrite=arguments.overwrite,
        )
    )

    manifest = load_json(
        manifest_path
    )

    print("=" * 88)
    print("MERGED CROSSFIT HARM TARGETS")
    print("=" * 88)
    print("Manifest:", manifest_path)
    print(
        "Records:",
        manifest["record_count"],
    )
    print(
        "Harm positive:",
        manifest[
            "harm_positive_count"
        ],
    )
    print(
        "Positive rate:",
        f"{manifest['harm_positive_rate']:.6f}",
    )


if __name__ == "__main__":
    main()
