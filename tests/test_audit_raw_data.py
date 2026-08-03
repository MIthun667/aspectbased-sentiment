from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_raw_data import (
    audit_file,
    audit_file_pair,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value),
        encoding="utf-8",
    )


def valid_record() -> dict[str, object]:
    return {
        "token": [
            "The",
            "battery",
            "is",
            "good",
            ".",
        ],
        "pos": [
            "DT",
            "NN",
            "VBZ",
            "JJ",
            ".",
        ],
        "head": [2, 4, 4, 0, 4],
        "deprel": [
            "det",
            "nsubj",
            "cop",
            "ROOT",
            "punct",
        ],
        "aspects": [
            {
                "term": ["battery"],
                "from": 1,
                "to": 2,
                "polarity": "positive",
            }
        ],
        "short": [
            [0, 1, 2, 2, 3],
            [1, 0, 1, 1, 2],
            [2, 1, 0, 1, 2],
            [2, 1, 1, 0, 1],
            [3, 2, 2, 1, 0],
        ],
    }


def test_valid_record_has_no_errors(
    tmp_path: Path,
) -> None:
    path = tmp_path / "valid.json"
    write_json(path, [valid_record()])

    result = audit_file(path, tmp_path)

    assert result.records == 1
    assert result.aspects == 1
    assert result.tokens == 5
    assert result.records_with_short == 1
    assert result.polarity_counts["positive"] == 1
    assert result.errors == []


def test_detects_target_span_mismatch(
    tmp_path: Path,
) -> None:
    record = valid_record()
    record["aspects"][0]["term"] = ["screen"]

    path = tmp_path / "invalid_span.json"
    write_json(path, [record])

    result = audit_file(path, tmp_path)

    assert any(
        "target mismatch" in error
        for error in result.errors
    )


def test_detects_misaligned_pos_tags(
    tmp_path: Path,
) -> None:
    record = valid_record()
    record["pos"] = ["DT"]

    path = tmp_path / "invalid_pos.json"
    write_json(path, [record])

    result = audit_file(path, tmp_path)

    assert any(
        "pos length" in error
        for error in result.errors
    )


def test_detects_invalid_short_matrix(
    tmp_path: Path,
) -> None:
    record = valid_record()
    record["short"][0][1] = 9

    path = tmp_path / "invalid_short.json"
    write_json(path, [record])

    result = audit_file(path, tmp_path)

    assert any(
        "asymmetric short matrix" in error
        for error in result.errors
    )


def test_pair_audit_accepts_matching_files(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "train.json"
    write_path = tmp_path / "train_write.json"

    base_record = valid_record()
    base_record.pop("short")

    write_json(base_path, [base_record])
    write_json(write_path, [valid_record()])

    result = audit_file_pair(
        domain="laptops",
        split="train",
        base_path=base_path,
        write_path=write_path,
        root=tmp_path,
    )

    assert result.base_records == 1
    assert result.write_records == 1
    assert result.matched_records == 1
    assert result.errors == []


def test_pair_audit_detects_record_count_mismatch(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "train.json"
    write_path = tmp_path / "train_write.json"

    base_record = valid_record()
    base_record.pop("short")

    write_json(
        base_path,
        [base_record, base_record],
    )
    write_json(
        write_path,
        [valid_record()],
    )

    result = audit_file_pair(
        domain="restaurants",
        split="train",
        base_path=base_path,
        write_path=write_path,
        root=tmp_path,
    )

    assert result.base_records == 2
    assert result.write_records == 1

    assert any(
        "Record-count mismatch" in error
        for error in result.errors
    )


def test_pair_audit_detects_content_mismatch(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "test.json"
    write_path = tmp_path / "test_write.json"

    base_record = valid_record()
    base_record.pop("short")

    modified_write_record = valid_record()
    modified_write_record["token"][3] = "bad"

    write_json(base_path, [base_record])
    write_json(
        write_path,
        [modified_write_record],
    )

    result = audit_file_pair(
        domain="laptops",
        split="test",
        base_path=base_path,
        write_path=write_path,
        root=tmp_path,
    )

    assert any(
        "content mismatch" in error
        for error in result.errors
    )
