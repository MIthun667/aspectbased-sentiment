from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_evidence_harm_targets import (
    build_harm_target_record,
    build_harm_target_records,
    summarize_records,
)
from src.aspect_sentiment.data import (
    EvidenceHarmTargetDataset,
    parse_harm_target_record,
)


def make_audit_record(
    *,
    utility_delta: float = -0.1,
    selected_correct: bool = False,
    empty_correct: bool = True,
):
    selected_probability = 0.4
    empty_probability = (
        selected_probability
        - utility_delta
    )

    return {
        "instance_id": "one",
        "sentence_id": "one:sentence",
        "domain": "laptops",
        "split": "train",
        "selected_gold_probability": (
            selected_probability
        ),
        "empty_gold_probability": (
            empty_probability
        ),
        "selected_prediction_id": 0,
        "empty_prediction_id": 1,
        "gold_label_id": 1,
        "selected_correct": (
            selected_correct
        ),
        "empty_correct": empty_correct,
        "original_compatibility_score": 1.2,
        "original_gate_value": 0.8,
    }


def build_record(
    **kwargs,
):
    return build_harm_target_record(
        make_audit_record(**kwargs),
        harm_threshold=0.02,
        source_experiment="/tmp/experiment",
        source_checkpoint="/tmp/best.pt",
        source_checkpoint_epoch=5,
        source_intervention_directory=(
            "/tmp/interventions"
        ),
        source_audit_file="/tmp/audit.jsonl",
    )


def test_probability_and_decisive_harm() -> None:
    record = build_record(
        utility_delta=-0.1,
        selected_correct=False,
        empty_correct=True,
    )

    assert record[
        "probability_harm"
    ] is True

    assert record[
        "decisive_harm"
    ] is True

    assert record[
        "harm_target"
    ] is True

    assert record[
        "harm_severity"
    ] == pytest.approx(0.1)


def test_decisive_harm_preserved_below_threshold() -> None:
    record = build_record(
        utility_delta=-0.01,
        selected_correct=False,
        empty_correct=True,
    )

    assert record[
        "probability_harm"
    ] is False

    assert record[
        "decisive_harm"
    ] is True

    assert record[
        "harm_target"
    ] is True


def test_non_harmful_record() -> None:
    record = build_record(
        utility_delta=0.1,
        selected_correct=True,
        empty_correct=True,
    )

    assert record[
        "probability_harm"
    ] is False

    assert record[
        "decisive_harm"
    ] is False

    assert record[
        "harm_target"
    ] is False

    assert record[
        "harm_severity"
    ] == pytest.approx(0.0)


def test_parse_harm_target_record() -> None:
    parsed = parse_harm_target_record(
        build_record()
    )

    assert parsed.instance_id == "one"
    assert parsed.harm_target is True
    assert parsed.harm_severity == (
        pytest.approx(0.1)
    )


def test_parser_rejects_inconsistent_target() -> None:
    record = build_record()
    record["harm_target"] = False

    with pytest.raises(
        ValueError,
        match="harm_target",
    ):
        parse_harm_target_record(
            record
        )


def test_dataset_from_records() -> None:
    dataset = EvidenceHarmTargetDataset(
        [
            build_record(),
            {
                **build_record(
                    utility_delta=0.1,
                    selected_correct=True,
                    empty_correct=True,
                ),
                "instance_id": "two",
                "sentence_id": (
                    "two:sentence"
                ),
            },
        ]
    )

    assert len(dataset) == 2
    assert dataset[0].instance_id == "one"
    assert dataset[1].instance_id == "two"


def test_dataset_from_jsonl(
    tmp_path: Path,
) -> None:
    path = tmp_path / "targets.jsonl"

    path.write_text(
        json.dumps(build_record())
        + "\n",
        encoding="utf-8",
    )

    dataset = (
        EvidenceHarmTargetDataset
        .from_jsonl(path)
    )

    assert len(dataset) == 1
    assert dataset[0].harm_target is True


def test_duplicate_targets_rejected() -> None:
    record = build_record()

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        EvidenceHarmTargetDataset(
            [record, record]
        )


def test_build_records_rejects_duplicates() -> None:
    audit = make_audit_record()

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        build_harm_target_records(
            [audit, audit],
            harm_threshold=0.02,
            source_experiment="/tmp/e",
            source_checkpoint="/tmp/c",
            source_checkpoint_epoch=5,
            source_intervention_directory="/tmp/i",
            source_audit_file="/tmp/a",
        )


def test_summary_counts() -> None:
    records = [
        build_record(),
        {
            **build_record(
                utility_delta=0.1,
                selected_correct=True,
                empty_correct=True,
            ),
            "instance_id": "two",
            "sentence_id": "two:sentence",
        },
    ]

    summary = summarize_records(
        records
    )

    assert summary[
        "record_count"
    ] == 2

    assert summary[
        "harm_positive_count"
    ] == 1

    assert summary[
        "harm_negative_count"
    ] == 1
