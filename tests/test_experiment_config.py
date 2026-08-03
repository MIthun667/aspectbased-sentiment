from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.aspect_sentiment.config import (
    experiment_config_from_dict,
    load_experiment_config,
)


def minimal_config() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "model": {
            "name": "majority_class",
        },
    }


def test_minimal_config_uses_defaults() -> None:
    config = experiment_config_from_dict(
        minimal_config()
    )

    assert config.model.name == "majority_class"
    assert config.data.train_domain == "laptops"
    assert config.training.seed == 2026
    assert config.training.selection_metric == (
        "macro_f1"
    )
    assert config.evaluation.save_predictions
    assert config.output.experiment_name == "baseline"


def test_full_yaml_config_loads() -> None:
    path = Path(
        "configs/baselines/majority_laptops.yaml"
    )

    config = load_experiment_config(path)

    assert config.schema_version == "1.0"
    assert config.model.name == "majority_class"
    assert config.data.evaluation_domains == (
        "laptops",
    )
    assert config.output.experiment_name == (
        "majority_laptops"
    )
    assert config.output_directory() == Path(
        "artifacts/experiments/"
        "majority_laptops/seed_2026"
    )


def test_resolved_config_is_json_serializable() -> None:
    config = experiment_config_from_dict(
        minimal_config()
    )

    serialized = json.dumps(
        config.to_dict(),
        sort_keys=True,
    )

    assert "majority_class" in serialized


def test_unknown_root_field_is_rejected() -> None:
    value = minimal_config()
    value["unexpected"] = True

    with pytest.raises(
        ValueError,
        match="unknown fields",
    ):
        experiment_config_from_dict(value)


def test_unknown_nested_field_is_rejected() -> None:
    value = minimal_config()
    value["training"] = {
        "seed": 2026,
        "mystery_option": 10,
    }

    with pytest.raises(
        ValueError,
        match="unknown fields",
    ):
        experiment_config_from_dict(value)


def test_invalid_domain_is_rejected() -> None:
    value = minimal_config()
    value["data"] = {
        "train_domain": "unknown-domain",
    }

    with pytest.raises(
        ValueError,
        match="Unsupported train_domain",
    ):
        experiment_config_from_dict(value)


def test_invalid_seed_is_rejected() -> None:
    value = minimal_config()
    value["training"] = {
        "seed": -1,
    }

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        experiment_config_from_dict(value)


def test_missing_model_is_rejected() -> None:
    value = {
        "schema_version": "1.0",
    }

    with pytest.raises(
        ValueError,
        match="Missing required field: model",
    ):
        experiment_config_from_dict(value)


def test_empty_yaml_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        load_experiment_config(path)


def test_config_round_trip_through_yaml(
    tmp_path: Path,
) -> None:
    original = minimal_config()
    path = tmp_path / "config.yaml"

    path.write_text(
        yaml.safe_dump(
            original,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    loaded = load_experiment_config(path)

    assert loaded.model.name == "majority_class"
    assert loaded.schema_version == "1.0"
