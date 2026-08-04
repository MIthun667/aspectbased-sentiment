from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.merge_crossfit_harm_targets import (
    merge_crossfit_targets,
)


def write_json(
    path: Path,
    value,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(value) + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    records,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def make_record(
    instance_id: str,
    *,
    harmful: bool,
):
    return {
        "schema_version": 1,
        "instance_id": instance_id,
        "sentence_id": (
            instance_id + ":sentence"
        ),
        "domain": "laptops",
        "split": "train",
        "gold_label_id": 2,
        "harm_target": harmful,
        "probability_harm": harmful,
        "decisive_harm": False,
        "selected_causes_error": False,
        "harm_severity": (
            0.1 if harmful else 0.0
        ),
    }


def prepare_inputs(
    tmp_path: Path,
):
    manifest_path = (
        tmp_path / "fold_manifest.json"
    )

    write_json(
        manifest_path,
        {
            "global_statistics": {
                "instance_count": 4,
            },
            "folds": [
                {
                    "fold_index": 0,
                    "heldout_instance_ids": [
                        "one",
                        "two",
                    ],
                },
                {
                    "fold_index": 1,
                    "heldout_instance_ids": [
                        "three",
                        "four",
                    ],
                },
            ],
        },
    )

    fold_0 = tmp_path / "fold_0"
    fold_1 = tmp_path / "fold_1"

    write_jsonl(
        fold_0 / "harm_targets.jsonl",
        [
            make_record(
                "one",
                harmful=True,
            ),
            make_record(
                "two",
                harmful=False,
            ),
        ],
    )

    write_jsonl(
        fold_1 / "harm_targets.jsonl",
        [
            make_record(
                "three",
                harmful=False,
            ),
            make_record(
                "four",
                harmful=True,
            ),
        ],
    )

    write_json(
        fold_0 / "manifest.json",
        {"fold": 0},
    )

    write_json(
        fold_1 / "manifest.json",
        {"fold": 1},
    )

    return (
        manifest_path,
        fold_0,
        fold_1,
    )


def test_merges_crossfit_targets(
    tmp_path: Path,
) -> None:
    (
        manifest_path,
        fold_0,
        fold_1,
    ) = prepare_inputs(tmp_path)

    output = tmp_path / "output"

    result = merge_crossfit_targets(
        crossfit_manifest_path=(
            manifest_path
        ),
        fold_directories=[
            fold_0,
            fold_1,
        ],
        output_directory=output,
        overwrite=False,
    )

    assert result.is_file()

    summary = json.loads(
        (
            output / "summary.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert summary["record_count"] == 4
    assert (
        summary["harm_positive_count"]
        == 2
    )

    records = [
        json.loads(line)
        for line in (
            output / "harm_targets.jsonl"
        ).read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert {
        record["source_fold"]
        for record in records
    } == {0, 1}

    assert all(
        record[
            "target_generation_mode"
        ]
        == "cross_fitted_out_of_fold"
        for record in records
    )


def test_missing_fold_instance_rejected(
    tmp_path: Path,
) -> None:
    (
        manifest_path,
        fold_0,
        fold_1,
    ) = prepare_inputs(tmp_path)

    write_jsonl(
        fold_0 / "harm_targets.jsonl",
        [
            make_record(
                "one",
                harmful=True,
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        merge_crossfit_targets(
            crossfit_manifest_path=(
                manifest_path
            ),
            fold_directories=[
                fold_0,
                fold_1,
            ],
            output_directory=(
                tmp_path / "output"
            ),
            overwrite=False,
        )


def test_wrong_source_split_rejected(
    tmp_path: Path,
) -> None:
    (
        manifest_path,
        fold_0,
        fold_1,
    ) = prepare_inputs(tmp_path)

    record = make_record(
        "one",
        harmful=True,
    )

    record["split"] = "validation"

    write_jsonl(
        fold_0 / "harm_targets.jsonl",
        [
            record,
            make_record(
                "two",
                harmful=False,
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="split=train",
    ):
        merge_crossfit_targets(
            crossfit_manifest_path=(
                manifest_path
            ),
            fold_directories=[
                fold_0,
                fold_1,
            ],
            output_directory=(
                tmp_path / "output"
            ),
            overwrite=False,
        )
