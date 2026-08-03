from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.run_transformer_baseline import (
    class_weights_from_records,
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
            "good",
        ],
        "aspect_text": "battery",
        "aspect_start": 0,
        "aspect_end": 1,
        "polarity": polarity,
        "label_id": label_id,
    }


def test_metric_improved_maximization() -> None:
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


def test_metric_improved_minimization() -> None:
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


def test_prediction_records() -> None:
    records = [
        make_record(
            instance_id="example",
            label_id=2,
        )
    ]

    output = prediction_records(
        records,
        predictions=np.asarray([2]),
        probabilities=np.asarray(
            [[0.1, 0.2, 0.7]]
        ),
        split_name="test",
    )

    assert len(output) == 1
    assert output[0].predicted_label == "positive"
    assert output[0].split == "test"


def test_class_weights_are_finite() -> None:
    records = [
        make_record(
            instance_id="negative",
            label_id=0,
        ),
        make_record(
            instance_id="neutral",
            label_id=1,
        ),
        make_record(
            instance_id="positive",
            label_id=2,
        ),
    ]

    weights = class_weights_from_records(
        records,
        device=torch.device("cpu"),
    )

    assert weights.shape == (3,)
    assert torch.isfinite(weights).all()


def test_class_weights_require_all_classes() -> None:
    records = [
        make_record(
            instance_id="negative",
            label_id=0,
        ),
        make_record(
            instance_id="positive",
            label_id=2,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="All classes",
    ):
        class_weights_from_records(
            records,
            device=torch.device("cpu"),
        )


def test_transformer_configuration_loads() -> None:
    config = experiment_config_from_dict(
        {
            "schema_version": "1.0",
            "model": {
                "name": "deberta_pair_classifier",
                "parameters": {
                    "model_name_or_path": (
                        "microsoft/deberta-v3-base"
                    ),
                    "maximum_length": 128,
                    "dropout": 0.1,
                    "gradient_clip_norm": 1.0,
                    "warmup_ratio": 0.1,
                    "class_weighting": False,
                    "use_bfloat16": False,
                    "local_files_only": True,
                    "num_workers": 0,
                },
            },
            "training": {
                "seed": 2026,
                "epochs": 3,
                "batch_size": 8,
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "early_stopping_patience": 2,
                "selection_metric": "macro_f1",
                "maximize_selection_metric": True,
            },
            "output": {
                "root": "artifacts/experiments",
                "experiment_name": "test_transformer",
            },
        }
    )

    assert (
        config.model.name
        == "deberta_pair_classifier"
    )
    assert config.training.epochs == 3


def test_transformer_runner_sets_deterministic_cublas() -> None:
    import os

    assert os.environ[
        "CUBLAS_WORKSPACE_CONFIG"
    ] in {
        ":4096:8",
        ":16:8",
    }


def test_transformer_runner_disables_hf_progress() -> None:
    import os

    assert (
        os.environ[
            "HF_HUB_DISABLE_PROGRESS_BARS"
        ]
        == "1"
    )
