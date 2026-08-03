from __future__ import annotations

import json
from pathlib import Path

from scripts.run_neural_baseline import (
    metric_improved,
    prediction_records,
)
from src.aspect_sentiment.config import (
    experiment_config_from_dict,
)


def make_record(
    *,
    instance_id: str,
    label_id: int,
) -> dict[str, object]:
    polarity = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }[label_id]

    return {
        "instance_id": instance_id,
        "sentence_id": f"{instance_id}:sentence",
        "domain": "laptops",
        "split": "test",
        "tokens": [
            "battery",
            "is",
            instance_id,
        ],
        "aspect_text": "battery",
        "aspect_start": 0,
        "aspect_end": 1,
        "polarity": polarity,
        "label_id": label_id,
    }


def test_metric_improved_for_maximization() -> None:
    assert metric_improved(
        current=0.6,
        best=0.5,
        maximize=True,
    )

    assert not metric_improved(
        current=0.4,
        best=0.5,
        maximize=True,
    )


def test_metric_improved_for_minimization() -> None:
    assert metric_improved(
        current=0.4,
        best=0.5,
        maximize=False,
    )

    assert not metric_improved(
        current=0.6,
        best=0.5,
        maximize=False,
    )


def test_prediction_records_use_requested_split() -> None:
    records = [
        make_record(
            instance_id="example",
            label_id=2,
        )
    ]

    output = prediction_records(
        records,
        predictions=[2],
        probabilities=[
            [0.1, 0.2, 0.7]
        ],
        split_name="test_duplicate_excluded",
    )

    assert len(output) == 1
    assert (
        output[0].split
        == "test_duplicate_excluded"
    )
    assert (
        output[0].predicted_label
        == "positive"
    )


def test_bilstm_configuration_loads() -> None:
    config = experiment_config_from_dict(
        {
            "schema_version": "1.0",
            "model": {
                "name": "target_aware_bilstm",
                "parameters": {
                    "minimum_frequency": 2,
                    "maximum_length": 128,
                    "embedding_dimension": 128,
                    "hidden_dimension": 128,
                    "number_of_layers": 1,
                    "dropout": 0.3,
                    "gradient_clip_norm": 1.0,
                    "class_weighting": False,
                    "num_workers": 0,
                },
            },
            "training": {
                "seed": 2026,
                "epochs": 5,
                "batch_size": 16,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "early_stopping_patience": 2,
            },
            "output": {
                "root": "artifacts/experiments",
                "experiment_name": "test_bilstm",
            },
        }
    )

    assert (
        config.model.name
        == "target_aware_bilstm"
    )
    assert config.training.epochs == 5
