from __future__ import annotations

import json

import pytest

from scripts.audit_evidence_utility import (
    build_utility_audit,
    classify_utility,
    derive_harm_category,
    run_audit,
    summarize_audit,
)


def make_record(
    *,
    instance_id: str = "one",
    gold_label_id: int = 2,
    original_probabilities=(
        0.1,
        0.2,
        0.7,
    ),
    intervened_probabilities=(
        0.2,
        0.5,
        0.3,
    ),
    original_prediction_id: int = 2,
    intervened_prediction_id: int = 1,
    evidence_indices=(2,),
):
    labels = (
        "negative",
        "neutral",
        "positive",
    )

    return {
        "schema_version": "1.0",
        "intervention": "original",
        "instance_id": instance_id,
        "sentence_id": f"{instance_id}:sentence",
        "domain": "laptops",
        "split": "test",
        "tokens": [
            "battery",
            "is",
            "excellent",
        ],
        "aspect_text": "battery",
        "aspect_start": 0,
        "aspect_end": 1,
        "gold_label": labels[
            gold_label_id
        ],
        "gold_label_id": gold_label_id,
        "original_evidence_indices": list(
            evidence_indices
        ),
        "original_evidence_tokens": (
            ["excellent"]
            if evidence_indices
            else []
        ),
        "intervened_evidence_indices": list(
            evidence_indices
        ),
        "intervened_evidence_tokens": (
            ["excellent"]
            if evidence_indices
            else []
        ),
        "original_prediction": labels[
            original_prediction_id
        ],
        "original_prediction_id": (
            original_prediction_id
        ),
        "intervened_prediction": labels[
            intervened_prediction_id
        ],
        "intervened_prediction_id": (
            intervened_prediction_id
        ),
        "original_probabilities": list(
            original_probabilities
        ),
        "intervened_probabilities": list(
            intervened_probabilities
        ),
        "original_compatibility_score": 1.2,
        "intervened_compatibility_score": 0.0,
        "compatibility_score_change": -1.2,
        "original_gate_value": 0.8,
        "intervened_gate_value": 0.0,
        "gate_value_change": -0.8,
    }


def make_empty_record(**kwargs):
    record = make_record(**kwargs)
    record["intervention"] = "empty"
    record[
        "intervened_evidence_indices"
    ] = []
    record[
        "intervened_evidence_tokens"
    ] = []
    return record


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (0.05, "helpful"),
        (0.20, "helpful"),
        (0.049, "neutral"),
        (0.0, "neutral"),
        (-0.049, "neutral"),
        (-0.05, "harmful"),
        (-0.20, "harmful"),
    ],
)
def test_classify_utility(
    delta,
    expected,
) -> None:
    assert classify_utility(
        delta,
        threshold=0.05,
    ) == expected


def test_build_utility_audit_harmful_flip() -> None:
    original = make_record(
        original_probabilities=(
            0.1,
            0.2,
            0.7,
        ),
        original_prediction_id=1,
        intervened_prediction_id=1,
    )

    empty = make_empty_record(
        original_probabilities=(
            0.1,
            0.2,
            0.7,
        ),
        intervened_probabilities=(
            0.1,
            0.1,
            0.8,
        ),
        original_prediction_id=1,
        intervened_prediction_id=2,
    )

    audit = build_utility_audit(
        original_records=[original],
        empty_records=[empty],
        threshold=0.05,
    )

    record = audit[0]

    assert record[
        "utility_delta"
    ] == pytest.approx(-0.1)

    assert (
        record["utility_class"]
        == "harmful"
    )

    assert (
        record["harm_category"]
        == "selected_causes_error"
    )

    assert record[
        "original_compatibility_score"
    ] == pytest.approx(1.2)

    assert record[
        "original_gate_value"
    ] == pytest.approx(0.8)


def test_derive_harm_category_selected_prevents_error() -> None:
    category = derive_harm_category(
        selected_correct=True,
        empty_correct=False,
        prediction_flipped=True,
        utility_class="helpful",
    )

    assert (
        category
        == "selected_prevents_error"
    )


def test_summary_counts() -> None:
    records = [
        {
            "instance_id": "one",
            "gold_label": "positive",
            "selected_evidence_indices": [2],
            "selected_evidence_text": "excellent",
            "utility_delta": 0.2,
            "utility_class": "helpful",
            "harm_category": (
                "selected_prevents_error"
            ),
            "aspect_text": "battery",
        },
        {
            "instance_id": "two",
            "gold_label": "negative",
            "selected_evidence_indices": [3],
            "selected_evidence_text": "bad",
            "utility_delta": -0.3,
            "utility_class": "harmful",
            "harm_category": (
                "selected_causes_error"
            ),
            "aspect_text": "screen",
        },
        {
            "instance_id": "three",
            "gold_label": "neutral",
            "selected_evidence_indices": [],
            "selected_evidence_text": "",
            "utility_delta": 0.0,
            "utility_class": "neutral",
            "harm_category": (
                "neutral_effect"
            ),
            "aspect_text": "price",
        },
    ]

    summary = summarize_audit(
        records,
        threshold=0.05,
    )

    assert (
        summary[
            "utility_class_counts"
        ]
        == {
            "harmful": 1,
            "helpful": 1,
            "neutral": 1,
        }
    )

    assert (
        summary[
            "number_selected_causes_error"
        ]
        == 1
    )

    assert (
        summary[
            "number_selected_prevents_error"
        ]
        == 1
    )


def test_run_audit_writes_outputs(
    tmp_path,
) -> None:
    intervention_directory = (
        tmp_path / "interventions"
    )

    prediction_directory = (
        intervention_directory
        / "predictions"
    )

    prediction_directory.mkdir(
        parents=True
    )

    original = make_record()
    empty = make_empty_record()

    for path, record in (
        (
            prediction_directory
            / "original.jsonl",
            original,
        ),
        (
            prediction_directory
            / "empty.jsonl",
            empty,
        ),
    ):
        path.write_text(
            json.dumps(record) + "\n",
            encoding="utf-8",
        )

    paths = run_audit(
        intervention_directory=(
            intervention_directory
        ),
        output_directory=None,
        threshold=0.05,
        overwrite=False,
    )

    assert paths["jsonl"].is_file()
    assert paths["csv"].is_file()
    assert paths["summary"].is_file()

    summary = json.loads(
        paths["summary"].read_text(
            encoding="utf-8"
        )
    )

    assert (
        summary["number_of_instances"]
        == 1
    )
