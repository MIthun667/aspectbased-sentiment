from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DOMAINS = ("laptops", "restaurants", "tweets")
MAIN_SPLITS = ("train", "validation", "test")

EXPECTED_FIXED_COUNTS = {
    "laptops": {
        "train_full": 2282,
        "test": 632,
    },
    "restaurants": {
        "train_full": 3608,
        "test": 1119,
    },
    "tweets": {
        "train_full": 6051,
        "test": 677,
    },
}


@dataclass
class SplitAudit:
    domain: str
    split: str
    path: str
    instances: int = 0
    sentences: int = 0
    label_counts: Counter[str] = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(
                    f"{path}:{line_number}: expected object"
                )

            records.append(record)

    return records


def normalize_text(tokens: list[str]) -> str:
    text = " ".join(str(token) for token in tokens)
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text


def sentence_signature(record: dict[str, Any]) -> str:
    return normalize_text(record["tokens"])


def target_signature(record: dict[str, Any]) -> tuple[str, str]:
    return (
        sentence_signature(record),
        normalize_text(record["aspect_tokens"]),
    )


def record_fingerprint(record: dict[str, Any]) -> str:
    canonical = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def source_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["sentence_id"],
        record["aspect_start"],
        record["aspect_end"],
        record["polarity"],
    )


def audit_record_schema(
    record: dict[str, Any],
    *,
    domain: str,
    expected_split: str,
    record_index: int,
) -> list[str]:
    errors: list[str] = []

    required = {
        "instance_id",
        "sentence_id",
        "domain",
        "split",
        "tokens",
        "pos_tags",
        "dependency_heads",
        "dependency_relations",
        "aspect_text",
        "aspect_tokens",
        "aspect_start",
        "aspect_end",
        "polarity",
        "label_id",
        "root_indices",
        "graph_edges",
        "aspect_distances",
        "number_of_aspects",
        "is_multi_aspect",
    }

    missing = required.difference(record)

    if missing:
        errors.append(
            f"record={record_index}: missing fields {sorted(missing)}"
        )
        return errors

    if record["domain"] != domain:
        errors.append(
            f"record={record_index}: domain={record['domain']!r}, "
            f"expected={domain!r}"
        )

    if record["split"] != expected_split:
        errors.append(
            f"record={record_index}: split={record['split']!r}, "
            f"expected={expected_split!r}"
        )

    tokens = record["tokens"]
    aligned = {
        "pos_tags": record["pos_tags"],
        "dependency_heads": record["dependency_heads"],
        "dependency_relations": record["dependency_relations"],
        "aspect_distances": record["aspect_distances"],
    }

    for field_name, value in aligned.items():
        if len(value) != len(tokens):
            errors.append(
                f"record={record_index}: {field_name} length "
                f"{len(value)} != token length {len(tokens)}"
            )

    start = record["aspect_start"]
    end = record["aspect_end"]

    if tokens[start:end] != record["aspect_tokens"]:
        errors.append(
            f"record={record_index}: aspect span mismatch"
        )

    expected_multi = record["number_of_aspects"] > 1

    if record["is_multi_aspect"] != expected_multi:
        errors.append(
            f"record={record_index}: inconsistent is_multi_aspect"
        )

    return errors


def audit_split(
    *,
    domain: str,
    split: str,
    path: Path,
    expected_split: str,
    root: Path,
) -> tuple[SplitAudit, list[dict[str, Any]]]:
    records = load_jsonl(path)

    audit = SplitAudit(
        domain=domain,
        split=split,
        path=str(path.relative_to(root)),
        instances=len(records),
        sentences=len(
            {
                record["sentence_id"]
                for record in records
            }
        ),
    )

    instance_ids: set[str] = set()
    identities: set[tuple[Any, ...]] = set()

    for record_index, record in enumerate(records):
        audit.errors.extend(
            audit_record_schema(
                record,
                domain=domain,
                expected_split=expected_split,
                record_index=record_index,
            )
        )

        instance_id = record["instance_id"]

        if instance_id in instance_ids:
            audit.errors.append(
                f"record={record_index}: duplicate instance_id "
                f"{instance_id!r}"
            )

        instance_ids.add(instance_id)

        identity = source_identity(record)

        if identity in identities:
            audit.errors.append(
                f"record={record_index}: duplicate source identity "
                f"{identity!r}"
            )

        identities.add(identity)
        audit.label_counts[record["polarity"]] += 1

    expected_count = EXPECTED_FIXED_COUNTS.get(
        domain,
        {},
    ).get(split)

    if (
        expected_count is not None
        and len(records) != expected_count
    ):
        audit.errors.append(
            f"instance count={len(records)}, "
            f"expected={expected_count}"
        )

    return audit, records


def overlap_report(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
) -> dict[str, int]:
    left_sentence_ids = {
        record["sentence_id"]
        for record in left_records
    }
    right_sentence_ids = {
        record["sentence_id"]
        for record in right_records
    }

    left_texts = {
        sentence_signature(record)
        for record in left_records
    }
    right_texts = {
        sentence_signature(record)
        for record in right_records
    }

    left_targets = {
        target_signature(record)
        for record in left_records
    }
    right_targets = {
        target_signature(record)
        for record in right_records
    }

    return {
        "sentence_id_overlap": len(
            left_sentence_ids & right_sentence_ids
        ),
        "normalized_text_overlap": len(
            left_texts & right_texts
        ),
        "target_signature_overlap": len(
            left_targets & right_targets
        ),
    }


def verify_partition_reconstruction(
    train_full: list[dict[str, Any]],
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []

    full_identities = {
        source_identity(record)
        for record in train_full
    }
    partition_identities = {
        source_identity(record)
        for record in train + validation
    }

    missing = full_identities - partition_identities
    extra = partition_identities - full_identities

    if missing:
        errors.append(
            f"partition is missing {len(missing)} source instances"
        )

    if extra:
        errors.append(
            f"partition contains {len(extra)} extra source instances"
        )

    if len(train) + len(validation) != len(train_full):
        errors.append(
            "train + validation count does not equal train_full"
        )

    train_sentence_ids = {
        record["sentence_id"]
        for record in train
    }
    validation_sentence_ids = {
        record["sentence_id"]
        for record in validation
    }

    overlap = train_sentence_ids & validation_sentence_ids

    if overlap:
        errors.append(
            f"train/validation sentence-group leakage: {len(overlap)}"
        )

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit canonical processed ABSA splits."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/audits/canonical_split_audit.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = args.processed_root.resolve()
    repository_root = processed_root.parents[1]

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "domains": {},
        "summary": {
            "split_errors": 0,
            "partition_errors": 0,
            "sentence_id_overlaps": 0,
            "normalized_text_overlaps": 0,
            "target_signature_overlaps": 0,
        },
    }

    for domain in DOMAINS:
        domain_directory = processed_root / domain
        records_by_split: dict[str, list[dict[str, Any]]] = {}
        split_reports: dict[str, Any] = {}

        for split in (
            "train_full",
            "train",
            "validation",
            "test",
        ):
            path = domain_directory / f"{split}.jsonl"

            expected_internal_split = (
                "train"
                if split == "train_full"
                else split
            )

            audit, records = audit_split(
                domain=domain,
                split=split,
                path=path,
                expected_split=expected_internal_split,
                root=repository_root,
            )

            records_by_split[split] = records
            split_reports[split] = {
                **asdict(audit),
                "label_counts": dict(audit.label_counts),
                "error_count": len(audit.errors),
                "warning_count": len(audit.warnings),
            }

            report["summary"]["split_errors"] += len(
                audit.errors
            )

        partition_errors = verify_partition_reconstruction(
            records_by_split["train_full"],
            records_by_split["train"],
            records_by_split["validation"],
        )

        pair_overlaps: dict[str, Any] = {}

        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            key = f"{left}__{right}"
            overlap = overlap_report(
                records_by_split[left],
                records_by_split[right],
            )
            pair_overlaps[key] = overlap

            report["summary"]["sentence_id_overlaps"] += (
                overlap["sentence_id_overlap"]
            )
            report["summary"]["normalized_text_overlaps"] += (
                overlap["normalized_text_overlap"]
            )
            report["summary"]["target_signature_overlaps"] += (
                overlap["target_signature_overlap"]
            )

        report["summary"]["partition_errors"] += len(
            partition_errors
        )

        report["domains"][domain] = {
            "splits": split_reports,
            "partition_errors": partition_errors,
            "overlaps": pair_overlaps,
        }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    print("=" * 88)
    print("CANONICAL SPLIT AUDIT")
    print("=" * 88)

    for domain, domain_report in report["domains"].items():
        print(f"\n[{domain.upper()}]")

        for split, split_report in (
            domain_report["splits"].items()
        ):
            print(
                f"{split:11s} "
                f"instances={split_report['instances']:6d} "
                f"sentences={split_report['sentences']:6d} "
                f"errors={split_report['error_count']:6d}"
            )

            for error in split_report["errors"][:10]:
                print(f"  ERROR: {error}")

        for error in domain_report["partition_errors"]:
            print(f"  PARTITION ERROR: {error}")

        for pair, overlap in domain_report["overlaps"].items():
            print(
                f"{pair:24s} "
                f"id={overlap['sentence_id_overlap']:4d} "
                f"text={overlap['normalized_text_overlap']:4d} "
                f"target={overlap['target_signature_overlap']:4d}"
            )

    print("\n" + "-" * 88)

    for key, value in report["summary"].items():
        print(f"{key:27s}: {value}")

    print(f"Report:                     {args.output}")
    print("-" * 88)


if __name__ == "__main__":
    main()
