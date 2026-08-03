from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VALID_POLARITIES = {"positive", "negative", "neutral"}


@dataclass
class FileAudit:
    path: str
    records: int = 0
    aspects: int = 0
    tokens: int = 0
    polarity_counts: Counter[str] = field(default_factory=Counter)
    multi_aspect_records: int = 0
    records_with_short: int = 0
    errors: list[str] = field(default_factory=list)


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, list):
        raise TypeError(
            f"{path}: expected top-level JSON list, "
            f"found {type(value).__name__}"
        )

    return value


def audit_file(path: Path, root: Path) -> FileAudit:
    relative_path = str(path.relative_to(root))
    audit = FileAudit(path=relative_path)

    try:
        records = load_json(path)
    except Exception as error:
        audit.errors.append(f"Unable to load file: {error}")
        return audit

    audit.records = len(records)

    for record_index, record in enumerate(records):
        prefix = f"record={record_index}"

        if not isinstance(record, dict):
            audit.errors.append(
                f"{prefix}: expected object, found {type(record).__name__}"
            )
            continue

        tokens = record.get("token")
        pos_tags = record.get("pos")
        heads = record.get("head")
        relations = record.get("deprel")
        aspects = record.get("aspects")
        shortest_paths = record.get("short")

        if not isinstance(tokens, list):
            audit.errors.append(f"{prefix}: missing or invalid token list")
            continue

        sentence_length = len(tokens)
        audit.tokens += sentence_length

        aligned_fields = {
            "pos": pos_tags,
            "head": heads,
            "deprel": relations,
        }

        for field_name, field_value in aligned_fields.items():
            if not isinstance(field_value, list):
                audit.errors.append(
                    f"{prefix}: missing or invalid {field_name} list"
                )
            elif len(field_value) != sentence_length:
                audit.errors.append(
                    f"{prefix}: {field_name} length={len(field_value)} "
                    f"but token length={sentence_length}"
                )

        if not isinstance(aspects, list):
            audit.errors.append(f"{prefix}: missing or invalid aspects list")
            continue

        audit.aspects += len(aspects)

        if len(aspects) > 1:
            audit.multi_aspect_records += 1

        for aspect_index, aspect in enumerate(aspects):
            aspect_prefix = f"{prefix}, aspect={aspect_index}"

            if not isinstance(aspect, dict):
                audit.errors.append(
                    f"{aspect_prefix}: aspect must be an object"
                )
                continue

            term = aspect.get("term")
            start = aspect.get("from")
            end = aspect.get("to")
            polarity = aspect.get("polarity")

            if not isinstance(term, list):
                audit.errors.append(
                    f"{aspect_prefix}: invalid aspect term"
                )
                continue

            if not isinstance(start, int) or not isinstance(end, int):
                audit.errors.append(
                    f"{aspect_prefix}: invalid aspect boundaries"
                )
                continue

            if start < 0 or end > sentence_length or start >= end:
                audit.errors.append(
                    f"{aspect_prefix}: invalid span [{start}, {end}) "
                    f"for sentence length {sentence_length}"
                )
            elif tokens[start:end] != term:
                audit.errors.append(
                    f"{aspect_prefix}: target mismatch; "
                    f"tokens[{start}:{end}]={tokens[start:end]!r}, "
                    f"term={term!r}"
                )

            if polarity not in VALID_POLARITIES:
                audit.errors.append(
                    f"{aspect_prefix}: unsupported polarity {polarity!r}"
                )
            else:
                audit.polarity_counts[polarity] += 1

        if shortest_paths is not None:
            audit.records_with_short += 1

            if not isinstance(shortest_paths, list):
                audit.errors.append(
                    f"{prefix}: short must be a list"
                )
                continue

            if len(shortest_paths) != sentence_length:
                audit.errors.append(
                    f"{prefix}: short rows={len(shortest_paths)} "
                    f"but token length={sentence_length}"
                )
                continue

            for row_index, row in enumerate(shortest_paths):
                if not isinstance(row, list):
                    audit.errors.append(
                        f"{prefix}: short row {row_index} is not a list"
                    )
                    continue

                if len(row) != sentence_length:
                    audit.errors.append(
                        f"{prefix}: short row {row_index} "
                        f"length={len(row)}, expected={sentence_length}"
                    )
                    continue

                if row[row_index] != 0:
                    audit.errors.append(
                        f"{prefix}: short diagonal "
                        f"[{row_index},{row_index}]={row[row_index]}, expected 0"
                    )

            for row_index in range(sentence_length):
                row = shortest_paths[row_index]
                if not isinstance(row, list) or len(row) != sentence_length:
                    continue

                for column_index in range(row_index + 1, sentence_length):
                    other_row = shortest_paths[column_index]
                    if (
                        not isinstance(other_row, list)
                        or len(other_row) != sentence_length
                    ):
                        continue

                    if row[column_index] != other_row[row_index]:
                        audit.errors.append(
                            f"{prefix}: asymmetric short matrix at "
                            f"[{row_index},{column_index}]"
                        )
                        break

    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the raw CoreNLP ABSA JSON datasets."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root containing the three CoreNLP directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/audits/raw_dataset_audit.json"),
        help="Output path for the machine-readable audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    domains = {
        "laptops": root / "Laptops_corenlp",
        "restaurants": root / "Restaurants_corenlp",
        "tweets": root / "Tweets_corenlp",
    }

    required_names = {
        "train.json",
        "train_write.json",
        "test.json",
        "test_write.json",
    }

    report: dict[str, Any] = {
        "root": str(root),
        "domains": {},
        "summary": {
            "files": 0,
            "records": 0,
            "aspects": 0,
            "tokens": 0,
            "errors": 0,
        },
    }

    for domain, directory in domains.items():
        domain_report: dict[str, Any] = {
            "directory": str(directory),
            "missing_files": [],
            "files": {},
        }

        if not directory.exists():
            domain_report["missing_files"] = sorted(required_names)
            report["domains"][domain] = domain_report
            continue

        existing_names = {
            path.name
            for path in directory.glob("*.json")
        }

        domain_report["missing_files"] = sorted(
            required_names - existing_names
        )

        for filename in sorted(required_names & existing_names):
            path = directory / filename
            audit = audit_file(path, root)

            file_report = {
                "path": audit.path,
                "records": audit.records,
                "aspects": audit.aspects,
                "tokens": audit.tokens,
                "polarity_counts": dict(audit.polarity_counts),
                "multi_aspect_records": audit.multi_aspect_records,
                "records_with_short": audit.records_with_short,
                "error_count": len(audit.errors),
                "errors": audit.errors,
            }

            domain_report["files"][filename] = file_report

            report["summary"]["files"] += 1
            report["summary"]["records"] += audit.records
            report["summary"]["aspects"] += audit.aspects
            report["summary"]["tokens"] += audit.tokens
            report["summary"]["errors"] += len(audit.errors)

        report["domains"][domain] = domain_report

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("=" * 88)
    print("RAW DATASET AUDIT")
    print("=" * 88)

    for domain, domain_report in report["domains"].items():
        print(f"\n[{domain.upper()}]")

        missing = domain_report["missing_files"]
        if missing:
            print(f"Missing files: {', '.join(missing)}")

        for filename, file_report in domain_report["files"].items():
            print(
                f"{filename:18s} "
                f"records={file_report['records']:6d} "
                f"aspects={file_report['aspects']:6d} "
                f"tokens={file_report['tokens']:8d} "
                f"errors={file_report['error_count']:6d}"
            )
            print(
                " " * 20
                + f"labels={file_report['polarity_counts']} "
                + f"multi_aspect={file_report['multi_aspect_records']} "
                + f"with_short={file_report['records_with_short']}"
            )

    print("\n" + "-" * 88)
    print(f"Files:   {report['summary']['files']}")
    print(f"Records: {report['summary']['records']}")
    print(f"Aspects: {report['summary']['aspects']}")
    print(f"Tokens:  {report['summary']['tokens']}")
    print(f"Errors:  {report['summary']['errors']}")
    print(f"Report:  {args.output}")
    print("-" * 88)


if __name__ == "__main__":
    main()
