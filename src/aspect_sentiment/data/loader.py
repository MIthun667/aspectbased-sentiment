from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "instance_id",
    "sentence_id",
    "domain",
    "split",
    "tokens",
    "aspect_text",
    "aspect_start",
    "aspect_end",
    "polarity",
    "label_id",
}


def validate_canonical_record(
    record: dict[str, Any],
    *,
    path: Path | None = None,
    line_number: int | None = None,
) -> None:
    missing_fields = sorted(
        REQUIRED_FIELDS - set(record)
    )

    if missing_fields:
        location = ""

        if path is not None:
            location = f" in {path}"

        if line_number is not None:
            location += f" at line {line_number}"

        raise ValueError(
            f"Canonical record{location} is missing fields: "
            f"{missing_fields}"
        )

    if record["label_id"] not in {0, 1, 2}:
        raise ValueError(
            "label_id must be 0, 1, or 2"
        )

    expected_polarity = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }[record["label_id"]]

    if record["polarity"] != expected_polarity:
        raise ValueError(
            "polarity does not match label_id: "
            f"{record['polarity']!r} != "
            f"{expected_polarity!r}"
        )

    if not isinstance(record["tokens"], list):
        raise TypeError(
            "tokens must be a list"
        )

    aspect_start = record["aspect_start"]
    aspect_end = record["aspect_end"]

    if not isinstance(aspect_start, int):
        raise TypeError(
            "aspect_start must be an integer"
        )

    if not isinstance(aspect_end, int):
        raise TypeError(
            "aspect_end must be an integer"
        )

    if not (
        0 <= aspect_start < aspect_end
        <= len(record["tokens"])
    ):
        raise ValueError(
            "Invalid aspect span: "
            f"[{aspect_start}, {aspect_end}) "
            f"for {len(record['tokens'])} tokens"
        )


def load_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    source_path = Path(path)

    if not source_path.is_file():
        raise FileNotFoundError(
            f"JSONL file not found: {source_path}"
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
                    "Each JSONL row must be an object: "
                    f"{source_path} line {line_number}"
                )

            validate_canonical_record(
                value,
                path=source_path,
                line_number=line_number,
            )

            records.append(value)

    if not records:
        raise ValueError(
            f"JSONL file contains no records: {source_path}"
        )

    return records


def canonical_split_path(
    processed_root: str | Path,
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
        Path(processed_root)
        / domain
        / f"{split}.jsonl"
    )


def load_canonical_split(
    processed_root: str | Path,
    *,
    domain: str,
    split: str,
) -> list[dict[str, Any]]:
    path = canonical_split_path(
        processed_root,
        domain=domain,
        split=split,
    )

    records = load_jsonl(path)

    mismatched_domains = sorted(
        {
            record["domain"]
            for record in records
            if record["domain"] != domain
        }
    )

    if mismatched_domains:
        raise ValueError(
            f"Records in {path} contain unexpected domains: "
            f"{mismatched_domains}"
        )

    mismatched_splits = sorted(
        {
            record["split"]
            for record in records
            if record["split"] != split
        }
    )

    if mismatched_splits:
        raise ValueError(
            f"Records in {path} contain unexpected splits: "
            f"{mismatched_splits}"
        )

    return records
