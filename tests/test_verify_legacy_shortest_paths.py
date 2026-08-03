from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_legacy_shortest_paths import verify_pair


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value),
        encoding="utf-8",
    )


def base_record() -> dict[str, object]:
    return {
        "token": ["The", "battery", "is", "good", "."],
        "pos": ["DT", "NN", "VBZ", "JJ", "."],
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
    }


def write_record() -> dict[str, object]:
    record = base_record()
    record["short"] = [
        [0, 1, 3, 2, 3],
        [1, 0, 2, 1, 2],
        [3, 2, 0, 1, 2],
        [2, 1, 1, 0, 1],
        [3, 2, 2, 1, 0],
    ]
    return record


def test_verify_pair_accepts_matching_matrices(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "train.json"
    write_path = tmp_path / "train_write.json"

    write_json(base_path, [base_record()])
    write_json(write_path, [write_record()])

    result = verify_pair(
        domain="laptops",
        split="train",
        base_path=base_path,
        write_path=write_path,
        maximum_distance=5,
    )

    assert result.status == "PASS"
    assert result.matching_records == 1
    assert result.mismatching_records == 0
    assert result.skipped is False


def test_verify_pair_detects_matrix_mismatch(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "test.json"
    write_path = tmp_path / "test_write.json"

    modified = write_record()
    modified["short"][0][1] = 5

    write_json(base_path, [base_record()])
    write_json(write_path, [modified])

    result = verify_pair(
        domain="laptops",
        split="test",
        base_path=base_path,
        write_path=write_path,
        maximum_distance=5,
    )

    assert result.status == "FAIL"
    assert result.matching_records == 0
    assert result.mismatching_records == 1


def test_verify_pair_marks_count_mismatch_unusable(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "train.json"
    write_path = tmp_path / "train_write.json"

    write_json(
        base_path,
        [base_record(), base_record()],
    )
    write_json(
        write_path,
        [write_record()],
    )

    result = verify_pair(
        domain="restaurants",
        split="train",
        base_path=base_path,
        write_path=write_path,
        maximum_distance=5,
    )

    assert result.status == "UNUSABLE"
    assert result.skipped is True
    assert result.compared_records == 0
    assert any(
        "record counts differ" in message
        for message in result.messages
    )


def test_verify_pair_detects_content_mismatch(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "test.json"
    write_path = tmp_path / "test_write.json"

    modified = write_record()
    modified["token"][3] = "bad"

    write_json(base_path, [base_record()])
    write_json(write_path, [modified])

    result = verify_pair(
        domain="tweets",
        split="test",
        base_path=base_path,
        write_path=write_path,
        maximum_distance=5,
    )

    assert result.status == "FAIL"
    assert result.mismatching_records == 1
    assert any(
        "content mismatch" in message
        for message in result.messages
    )
