from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DOMAINS = ("laptops", "restaurants", "tweets")
PAIRS = (
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def normalize_tokens(tokens: list[str]) -> str:
    text = " ".join(str(token) for token in tokens)
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text


def normalized_target(record: dict[str, Any]) -> str:
    return normalize_tokens(record["aspect_tokens"])


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "instance_id": record["instance_id"],
        "sentence_id": record["sentence_id"],
        "split": record["split"],
        "text": normalize_tokens(record["tokens"]),
        "target": normalized_target(record),
        "polarity": record["polarity"],
        "aspect_start": record["aspect_start"],
        "aspect_end": record["aspect_end"],
    }


def index_records(
    records: list[dict[str, Any]],
    *,
    include_target: bool,
) -> dict[Any, list[dict[str, Any]]]:
    index: dict[Any, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        text = normalize_tokens(record["tokens"])

        key: Any

        if include_target:
            key = (
                text,
                normalized_target(record),
            )
        else:
            key = text

        index[key].append(record)

    return index


def collect_overlap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    include_target: bool,
) -> list[dict[str, Any]]:
    left_index = index_records(
        left,
        include_target=include_target,
    )
    right_index = index_records(
        right,
        include_target=include_target,
    )

    overlaps: list[dict[str, Any]] = []

    for key in sorted(
        left_index.keys() & right_index.keys(),
        key=str,
    ):
        left_records = left_index[key]
        right_records = right_index[key]

        overlaps.append(
            {
                "key": key,
                "left": [
                    compact_record(record)
                    for record in left_records
                ],
                "right": [
                    compact_record(record)
                    for record in right_records
                ],
                "polarity_conflict": (
                    {
                        record["polarity"]
                        for record in left_records
                    }
                    != {
                        record["polarity"]
                        for record in right_records
                    }
                ),
            }
        )

    return overlaps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/audits/split_overlap_details.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "domains": {},
        "summary": {
            "text_overlap_groups": 0,
            "target_overlap_groups": 0,
            "polarity_conflicts": 0,
        },
    }

    for domain in DOMAINS:
        directory = args.processed_root / domain

        records = {
            split: load_jsonl(
                directory / f"{split}.jsonl"
            )
            for split in (
                "train",
                "validation",
                "test",
            )
        }

        domain_report: dict[str, Any] = {}

        for left_name, right_name in PAIRS:
            text_overlaps = collect_overlap(
                records[left_name],
                records[right_name],
                include_target=False,
            )
            target_overlaps = collect_overlap(
                records[left_name],
                records[right_name],
                include_target=True,
            )

            key = f"{left_name}__{right_name}"

            domain_report[key] = {
                "text_overlaps": text_overlaps,
                "target_overlaps": target_overlaps,
            }

            report["summary"]["text_overlap_groups"] += len(
                text_overlaps
            )
            report["summary"]["target_overlap_groups"] += len(
                target_overlaps
            )
            report["summary"]["polarity_conflicts"] += sum(
                item["polarity_conflict"]
                for item in target_overlaps
            )

        report["domains"][domain] = domain_report

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

    print(json.dumps(report["summary"], indent=2))
    print("Report:", args.output)


if __name__ == "__main__":
    main()
