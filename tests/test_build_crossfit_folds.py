from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_crossfit_folds import (
    assign_sentence_groups,
    build_crossfit_manifest,
    group_records_by_sentence,
    run_builder,
)


def make_records():
    records = []

    examples = [
        (
            "sentence-a",
            [0, 1],
        ),
        (
            "sentence-b",
            [2],
        ),
        (
            "sentence-c",
            [0, 2],
        ),
        (
            "sentence-d",
            [1],
        ),
        (
            "sentence-e",
            [2, 2],
        ),
        (
            "sentence-f",
            [0],
        ),
    ]

    for sentence_id, labels in examples:
        for aspect_index, label_id in enumerate(
            labels
        ):
            records.append(
                {
                    "instance_id": (
                        f"{sentence_id}:"
                        f"{aspect_index}"
                    ),
                    "sentence_id": (
                        sentence_id
                    ),
                    "domain": "laptops",
                    "split": "train",
                    "tokens": [
                        sentence_id,
                    ],
                    "aspect_text": (
                        sentence_id
                    ),
                    "aspect_start": 0,
                    "aspect_end": 1,
                    "polarity": {
                        0: "negative",
                        1: "neutral",
                        2: "positive",
                    }[label_id],
                    "label_id": label_id,
                }
            )

    return records


def test_grouping_preserves_sentence_instances() -> None:
    grouped = group_records_by_sentence(
        make_records()
    )

    assert len(
        grouped["sentence-a"]
    ) == 2

    assert {
        record["label_id"]
        for record in grouped[
            "sentence-a"
        ]
    } == {0, 1}


def test_assignment_is_deterministic() -> None:
    records = make_records()

    first = assign_sentence_groups(
        records,
        number_of_folds=2,
        seed=2026,
    )

    second = assign_sentence_groups(
        records,
        number_of_folds=2,
        seed=2026,
    )

    assert first == second


def test_folds_are_sentence_disjoint() -> None:
    manifest = build_crossfit_manifest(
        make_records(),
        domain="laptops",
        split="train",
        number_of_folds=2,
        seed=2026,
    )

    first = set(
        manifest["folds"][0][
            "heldout_sentence_ids"
        ]
    )

    second = set(
        manifest["folds"][1][
            "heldout_sentence_ids"
        ]
    )

    assert not (first & second)

    assert first | second == {
        "sentence-a",
        "sentence-b",
        "sentence-c",
        "sentence-d",
        "sentence-e",
        "sentence-f",
    }


def test_every_instance_held_out_once() -> None:
    records = make_records()

    manifest = build_crossfit_manifest(
        records,
        domain="laptops",
        split="train",
        number_of_folds=2,
        seed=2026,
    )

    heldout_ids = []

    for fold in manifest["folds"]:
        heldout_ids.extend(
            fold[
                "heldout_instance_ids"
            ]
        )

        assert not (
            set(
                fold[
                    "training_instance_ids"
                ]
            )
            & set(
                fold[
                    "heldout_instance_ids"
                ]
            )
        )

    assert sorted(heldout_ids) == sorted(
        record["instance_id"]
        for record in records
    )


def test_manifest_has_balanced_fold_sizes() -> None:
    manifest = build_crossfit_manifest(
        make_records(),
        domain="laptops",
        split="train",
        number_of_folds=2,
        seed=2026,
    )

    sizes = [
        fold[
            "heldout_statistics"
        ]["instance_count"]
        for fold in manifest["folds"]
    ]

    assert max(sizes) - min(sizes) <= 2


def test_duplicate_instance_rejected() -> None:
    records = make_records()
    records.append(
        dict(records[0])
    )

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        build_crossfit_manifest(
            records,
            domain="laptops",
            split="train",
            number_of_folds=2,
            seed=2026,
        )


def test_too_many_folds_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        assign_sentence_groups(
            make_records(),
            number_of_folds=20,
            seed=2026,
        )


def test_run_builder_writes_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    records = make_records()

    monkeypatch.setattr(
        "scripts.build_crossfit_folds."
        "load_canonical_split",
        lambda *args, **kwargs: (
            records
        ),
    )

    output_directory = (
        tmp_path / "crossfit"
    )

    manifest_path = run_builder(
        processed_root=(
            tmp_path / "processed"
        ),
        output_directory=(
            output_directory
        ),
        domain="laptops",
        split="train",
        number_of_folds=2,
        seed=2026,
        overwrite=False,
    )

    assert manifest_path.is_file()

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest["number_of_folds"]
        == 2
    )

    for fold_index in (0, 1):
        fold_directory = (
            output_directory
            / f"fold_{fold_index}"
        )

        assert (
            fold_directory
            / "train_instance_ids.json"
        ).is_file()

        assert (
            fold_directory
            / "heldout_instance_ids.json"
        ).is_file()
