from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from scripts.evaluate_evidence_interventions import (
    paired_prediction_records,
    validate_model_parameters,
)
from src.aspect_sentiment.data import (
    EvidenceAwareDataset,
)
from src.aspect_sentiment.evidence import (
    generate_evidence_intervention,
)


def make_instance():
    canonical = {
        "instance_id": "one",
        "sentence_id": "one:sentence",
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
        "polarity": "positive",
        "label_id": 2,
    }

    evidence = {
        "schema_version": 1,
        "instance_id": "one",
        "sentence_id": "one:sentence",
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
        "polarity": "positive",
        "selected_evidence_indices": [2],
        "selected_evidence_tokens": [
            "excellent"
        ],
        "evidence_is_empty": False,
        "candidate_count": 1,
        "selection_config": {
            "maximum_tokens": 3,
        },
    }

    dataset = EvidenceAwareDataset(
        [canonical],
        [evidence],
    )

    return dataset[0]


def make_result(
    probabilities,
):
    probability_array = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    return SimpleNamespace(
        predictions=(
            probability_array.argmax(
                axis=1
            )
        ),
        probabilities=probability_array,
        labels=np.asarray(
            [2],
            dtype=np.int64,
        ),
    )


def valid_parameters():
    return {
        "number_of_classes": 3,
        "dropout": 0.1,
        "maximum_length": 128,
        "mode": "cls_aspect_evidence",
        "evidence_root": (
            "data/derived/evidence"
        ),
        "reject_truncation": True,
        "dtype": "float32",
    }


def test_valid_model_parameters() -> None:
    parameters = valid_parameters()

    assert (
        validate_model_parameters(
            parameters
        )
        == parameters
    )


def test_non_evidence_mode_rejected() -> None:
    parameters = valid_parameters()
    parameters["mode"] = "cls_aspect"

    with pytest.raises(
        ValueError,
        match="cls_aspect_evidence",
    ):
        validate_model_parameters(
            parameters
        )


def test_missing_parameters_rejected() -> None:
    parameters = valid_parameters()
    del parameters["maximum_length"]

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        validate_model_parameters(
            parameters
        )


def test_paired_prediction_record() -> None:
    instance = make_instance()

    empty = generate_evidence_intervention(
        [instance],
        intervention="empty",
    ).instances[0]

    original = make_result(
        [[0.1, 0.2, 0.7]]
    )

    intervened = make_result(
        [[0.2, 0.5, 0.3]]
    )

    records = paired_prediction_records(
        original_instances=[instance],
        intervened_instances=[empty],
        original_result=original,
        intervened_result=intervened,
        intervention_name="empty",
    )

    assert len(records) == 1

    record = records[0]

    assert (
        record["instance_id"]
        == "one"
    )

    assert (
        record[
            "original_evidence_indices"
        ]
        == [2]
    )

    assert (
        record[
            "intervened_evidence_indices"
        ]
        == []
    )

    assert record[
        "prediction_flipped"
    ]

    assert (
        record["gold_probability_change"]
        == pytest.approx(-0.4)
    )


def test_identity_mismatch_rejected() -> None:
    instance = make_instance()

    other = make_instance()

    from dataclasses import replace

    other = replace(
        other,
        instance_id="other",
    )

    result = make_result(
        [[0.1, 0.2, 0.7]]
    )

    with pytest.raises(
        ValueError,
        match="identity",
    ):
        paired_prediction_records(
            original_instances=[instance],
            intervened_instances=[other],
            original_result=result,
            intervened_result=result,
            intervention_name="empty",
        )
