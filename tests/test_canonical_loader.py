from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.aspect_sentiment.data import (
    canonical_split_path,
    load_canonical_split,
    load_jsonl,
)


def valid_record() -> dict[str, object]:
    return {
        "instance_id": "laptops:train:s1:a0",
        "sentence_id": "laptops:train:s1",
        "domain": "laptops",
        "split": "train",
        "tokens": [
            "battery",
            "life",
            "is",
            "excellent",
        ],
        "aspect_text": "battery life",
        "aspect_start": 0,
        "aspect_end": 2,
        "polarity": "positive",
        "label_id": 2,
    }


def write_jsonl(
    path: Path,
    records: list[object],
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
                json.dumps(record)
            )
            handle.write("\n")


def test_canonical_split_path() -> None:
    assert canonical_split_path(
        "data/processed",
        domain="laptops",
        split="train",
    ) == Path(
        "data/processed/laptops/train.jsonl"
    )


def test_load_jsonl_reads_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"

    write_jsonl(
        path,
        [valid_record()],
    )

    records = load_jsonl(path)

    assert len(records) == 1
    assert records[0]["label_id"] == 2


def test_load_jsonl_rejects_missing_fields(
    tmp_path: Path,
) -> None:
    record = valid_record()
    del record["aspect_text"]

    path = tmp_path / "records.jsonl"

    write_jsonl(path, [record])

    with pytest.raises(
        ValueError,
        match="missing fields",
    ):
        load_jsonl(path)


def test_load_jsonl_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "{broken json}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid JSON",
    ):
        load_jsonl(path)


def test_load_jsonl_rejects_empty_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="no records",
    ):
        load_jsonl(path)


def test_load_jsonl_rejects_label_mismatch(
    tmp_path: Path,
) -> None:
    record = valid_record()
    record["polarity"] = "negative"

    path = tmp_path / "records.jsonl"
    write_jsonl(path, [record])

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        load_jsonl(path)


def test_load_canonical_split_validates_metadata(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "laptops"
        / "train.jsonl"
    )

    write_jsonl(
        path,
        [valid_record()],
    )

    records = load_canonical_split(
        tmp_path,
        domain="laptops",
        split="train",
    )

    assert len(records) == 1


def test_load_canonical_split_rejects_wrong_split(
    tmp_path: Path,
) -> None:
    record = valid_record()
    record["split"] = "test"

    path = (
        tmp_path
        / "laptops"
        / "train.jsonl"
    )

    write_jsonl(path, [record])

    with pytest.raises(
        ValueError,
        match="unexpected splits",
    ):
        load_canonical_split(
            tmp_path,
            domain="laptops",
            split="train",
        )
