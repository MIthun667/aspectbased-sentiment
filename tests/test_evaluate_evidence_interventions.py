from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from scripts.evaluate_evidence_interventions import (
    normalized_evaluation_heads,
    paired_prediction_records,
    validate_binding_model_parameters,
    validate_checkpoint_parameters,
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


def valid_binding_parameters():
    return {
        "number_of_classes": 3,
        "dropout": 0.1,
        "gate_dimension": 256,
        "maximum_length": 128,
        "evidence_root": (
            "data/derived/evidence"
        ),
        "reject_truncation": True,
        "dtype": "float32",
    }


def test_valid_binding_parameters() -> None:
    parameters = (
        valid_binding_parameters()
    )

    assert (
        validate_binding_model_parameters(
            parameters
        )
        == parameters
    )


def test_binding_parameters_require_gate() -> None:
    parameters = (
        valid_binding_parameters()
    )

    del parameters["gate_dimension"]

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        validate_binding_model_parameters(
            parameters
        )


def test_checkpoint_parameter_dispatch() -> None:
    evidence = valid_parameters()
    binding = valid_binding_parameters()

    assert (
        validate_checkpoint_parameters(
            "evidence_transformer",
            evidence,
        )
        == evidence
    )

    assert (
        validate_checkpoint_parameters(
            "evidence_binding_transformer",
            binding,
        )
        == binding
    )


def test_unknown_checkpoint_type_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported checkpoint",
    ):
        validate_checkpoint_parameters(
            "unknown",
            {},
        )


def test_normalized_single_head_result() -> None:
    result = SimpleNamespace(
        metrics={
            "macro_f1": 0.7,
        },
        predictions=np.asarray(
            [2],
            dtype=np.int64,
        ),
        probabilities=np.asarray(
            [[0.1, 0.2, 0.7]],
            dtype=np.float64,
        ),
        labels=np.asarray(
            [2],
            dtype=np.int64,
        ),
    )

    heads = normalized_evaluation_heads(
        result,
        checkpoint_type=(
            "evidence_transformer"
        ),
    )

    assert set(heads) == {
        "combined"
    }

    assert (
        heads["combined"]
        .metrics["macro_f1"]
        == 0.7
    )


def test_normalized_binding_heads() -> None:
    labels = np.asarray(
        [2],
        dtype=np.int64,
    )

    result = SimpleNamespace(
        combined_metrics={
            "macro_f1": 0.8,
        },
        context_metrics={
            "macro_f1": 0.7,
        },
        evidence_metrics={
            "macro_f1": 0.6,
        },
        combined_predictions=np.asarray(
            [2],
            dtype=np.int64,
        ),
        context_predictions=np.asarray(
            [2],
            dtype=np.int64,
        ),
        evidence_predictions=np.asarray(
            [1],
            dtype=np.int64,
        ),
        combined_probabilities=np.asarray(
            [[0.1, 0.1, 0.8]],
            dtype=np.float64,
        ),
        context_probabilities=np.asarray(
            [[0.1, 0.2, 0.7]],
            dtype=np.float64,
        ),
        evidence_probabilities=np.asarray(
            [[0.1, 0.6, 0.3]],
            dtype=np.float64,
        ),
        labels=labels,
    )

    heads = normalized_evaluation_heads(
        result,
        checkpoint_type=(
            "evidence_binding_transformer"
        ),
    )

    assert set(heads) == {
        "combined",
        "context",
        "evidence",
    }

    assert (
        heads["combined"]
        .metrics["macro_f1"]
        == 0.8
    )

    assert (
        heads["context"]
        .metrics["macro_f1"]
        == 0.7
    )

    assert (
        heads["evidence"]
        .metrics["macro_f1"]
        == 0.6
    )


def test_classification_metrics_for_head() -> None:
    from scripts.evaluate_evidence_interventions import (
        InterventionHeadResult,
        classification_metrics_for_head,
    )

    result = InterventionHeadResult(
        metrics={},
        predictions=np.asarray(
            [0, 1, 2],
            dtype=np.int64,
        ),
        probabilities=np.asarray(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ],
            dtype=np.float64,
        ),
        labels=np.asarray(
            [0, 1, 2],
            dtype=np.int64,
        ),
    )

    metrics = (
        classification_metrics_for_head(
            result
        )
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_binding_gate_payload() -> None:
    from scripts.evaluate_evidence_interventions import (
        binding_gate_payload,
    )

    result = SimpleNamespace(
        mean_gate_value=0.4,
        gate_standard_deviation=0.2,
        mean_available_gate_value=0.5,
        available_gate_standard_deviation=0.05,
        instance_gate_mean_standard_deviation=0.01,
        minimum_available_gate_value=0.2,
        maximum_available_gate_value=0.8,
    )

    payload = binding_gate_payload(
        result
    )

    assert payload is not None
    assert payload["mean"] == 0.4

    assert (
        payload[
            "instance_mean_standard_deviation"
        ]
        == 0.01
    )


def test_gate_payload_absent_for_legacy_result() -> None:
    from scripts.evaluate_evidence_interventions import (
        binding_gate_payload,
    )

    assert (
        binding_gate_payload(
            SimpleNamespace()
        )
        is None
    )


def test_binding_head_supports_prediction_records() -> None:
    from scripts.evaluate_evidence_interventions import (
        InterventionHeadResult,
    )

    instance = make_instance()

    empty = generate_evidence_intervention(
        [instance],
        intervention="empty",
    ).instances[0]

    original = InterventionHeadResult(
        metrics={},
        predictions=np.asarray(
            [2],
            dtype=np.int64,
        ),
        probabilities=np.asarray(
            [[0.1, 0.2, 0.7]],
            dtype=np.float64,
        ),
        labels=np.asarray(
            [2],
            dtype=np.int64,
        ),
    )

    intervened = InterventionHeadResult(
        metrics={},
        predictions=np.asarray(
            [1],
            dtype=np.int64,
        ),
        probabilities=np.asarray(
            [[0.2, 0.5, 0.3]],
            dtype=np.float64,
        ),
        labels=np.asarray(
            [2],
            dtype=np.int64,
        ),
    )

    records = paired_prediction_records(
        original_instances=[instance],
        intervened_instances=[empty],
        original_result=original,
        intervened_result=intervened,
        intervention_name="empty",
    )

    assert len(records) == 1

    assert records[0][
        "prediction_flipped"
    ]

    assert records[0][
        "gold_probability_change"
    ] == pytest.approx(-0.4)
