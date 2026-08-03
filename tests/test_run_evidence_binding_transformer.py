from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_evidence_binding_transformer import (
    binding_metric_payload,
)
from src.aspect_sentiment.config import (
    load_experiment_config,
)


CONFIG_PATH = Path(
    "configs/evidence_binding/"
    "deberta_binding_laptops.yaml"
)


def make_result():
    metrics = {
        "number_of_instances": 2,
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "macro_f1": 0.5,
        "weighted_f1": 0.5,
        "negative_log_likelihood": 1.0,
        "brier_score": 0.6,
        "label_order": [
            "negative",
            "neutral",
            "positive",
        ],
        "label_ids": [0, 1, 2],
        "per_class": {},
        "confusion_matrix": [
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
        ],
    }

    return SimpleNamespace(
        combined_metrics=dict(metrics),
        context_metrics=dict(metrics),
        evidence_metrics=dict(metrics),
        loss=1.5,
        combined_loss=1.0,
        context_loss=1.1,
        evidence_loss=0.9,
        agreement_loss=0.2,
        mean_gate_value=0.3,
        gate_standard_deviation=0.1,
        mean_available_gate_value=0.4,
        available_gate_standard_deviation=0.05,
        instance_gate_mean_standard_deviation=0.02,
        minimum_available_gate_value=0.1,
        maximum_available_gate_value=0.8,
        number_of_instances=2,
        number_with_evidence=1,
        combined_predictions=np.asarray(
            [0, 1]
        ),
        combined_probabilities=np.asarray(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
            ]
        ),
    )


def test_binding_config_loads() -> None:
    config = load_experiment_config(
        CONFIG_PATH
    )

    assert (
        config.model.name
        == "evidence_binding_deberta"
    )

    assert (
        config.model.parameters[
            "model_name_or_path"
        ]
        == "microsoft/deberta-v3-base"
    )

    assert (
        config.model.parameters[
            "gate_dimension"
        ]
        == 256
    )

    assert (
        config.model.parameters[
            "combined_weight"
        ]
        == 1.0
    )

    assert (
        config.model.parameters[
            "context_weight"
        ]
        == 0.2
    )

    assert (
        config.model.parameters[
            "evidence_weight"
        ]
        == 0.5
    )

    assert (
        config.model.parameters[
            "agreement_weight"
        ]
        == 0.1
    )

    assert (
        config.output.experiment_name
        == "evidence_binding_deberta_laptops"
    )


def test_binding_metric_payload() -> None:
    payload = binding_metric_payload(
        make_result()
    )

    assert (
        payload["combined"]["macro_f1"]
        == 0.5
    )

    assert (
        payload["context"]["accuracy"]
        == 0.5
    )

    assert (
        payload["evidence"]["macro_f1"]
        == 0.5
    )

    assert (
        payload["losses"]["total"]
        == 1.5
    )

    assert (
        payload["gate"]["mean"]
        == 0.3
    )

    assert (
        payload["gate"][
            "available_standard_deviation"
        ]
        == 0.05
    )

    assert (
        payload["gate"][
            "instance_mean_standard_deviation"
        ]
        == 0.02
    )

    assert (
        payload["gate"][
            "minimum_available"
        ]
        == 0.1
    )

    assert (
        payload["gate"][
            "maximum_available"
        ]
        == 0.8
    )

    assert (
        payload["number_with_evidence"]
        == 1
    )


def test_metric_payload_allows_no_evidence() -> None:
    result = make_result()
    result.evidence_metrics = None
    result.mean_available_gate_value = None

    payload = binding_metric_payload(
        result
    )

    assert payload["evidence"] is None

    assert (
        payload["gate"]["mean_available"]
        is None
    )


def test_deterministic_environment_is_set() -> None:
    assert os.environ[
        "CUBLAS_WORKSPACE_CONFIG"
    ] in {
        ":4096:8",
        ":16:8",
    }


def test_huggingface_progress_disabled() -> None:
    assert (
        os.environ[
            "HF_HUB_DISABLE_PROGRESS_BARS"
        ]
        == "1"
    )


def test_configuration_uses_fp32() -> None:
    config = load_experiment_config(
        CONFIG_PATH
    )

    assert (
        config.model.parameters[
            "use_bfloat16"
        ]
        is False
    )


@pytest.mark.parametrize(
    "parameter",
    [
        "maximum_length",
        "gate_dimension",
        "gradient_clip_norm",
        "warmup_ratio",
        "combined_weight",
        "context_weight",
        "evidence_weight",
        "agreement_weight",
        "agreement_temperature",
    ],
)
def test_required_binding_parameters_present(
    parameter: str,
) -> None:
    config = load_experiment_config(
        CONFIG_PATH
    )

    assert parameter in (
        config.model.parameters
    )
