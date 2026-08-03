from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .schema import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)


T = TypeVar("T")


def _require_mapping(
    value: Any,
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(
            f"{context} must be a mapping"
        )

    return value


def _construct_dataclass(
    cls: type[T],
    values: dict[str, Any],
    *,
    context: str,
) -> T:
    valid_fields = {
        field.name
        for field in fields(cls)
    }

    unknown_fields = sorted(
        set(values) - valid_fields
    )

    if unknown_fields:
        raise ValueError(
            f"{context} contains unknown fields: "
            f"{unknown_fields}"
        )

    try:
        return cls(**values)
    except TypeError as error:
        raise ValueError(
            f"Invalid {context} configuration: {error}"
        ) from error


def experiment_config_from_dict(
    value: dict[str, Any],
) -> ExperimentConfig:
    root = _require_mapping(
        value,
        context="experiment configuration",
    )

    valid_root_fields = {
        "schema_version",
        "description",
        "model",
        "data",
        "training",
        "evaluation",
        "output",
    }

    unknown_root_fields = sorted(
        set(root) - valid_root_fields
    )

    if unknown_root_fields:
        raise ValueError(
            "Experiment configuration contains "
            f"unknown fields: {unknown_root_fields}"
        )

    if "schema_version" not in root:
        raise ValueError(
            "Missing required field: schema_version"
        )

    if "model" not in root:
        raise ValueError(
            "Missing required field: model"
        )

    model_values = _require_mapping(
        root["model"],
        context="model",
    )

    data_values = _require_mapping(
        root.get("data", {}),
        context="data",
    )

    training_values = _require_mapping(
        root.get("training", {}),
        context="training",
    )

    evaluation_values = _require_mapping(
        root.get("evaluation", {}),
        context="evaluation",
    )

    output_values = _require_mapping(
        root.get("output", {}),
        context="output",
    )

    if "evaluation_domains" in data_values:
        evaluation_domains = data_values[
            "evaluation_domains"
        ]

        if not isinstance(
            evaluation_domains,
            (list, tuple),
        ):
            raise TypeError(
                "data.evaluation_domains must be "
                "a sequence"
            )

        data_values = dict(data_values)
        data_values["evaluation_domains"] = tuple(
            evaluation_domains
        )

    config = ExperimentConfig(
        schema_version=str(
            root["schema_version"]
        ),
        description=str(
            root.get("description", "")
        ),
        model=_construct_dataclass(
            ModelConfig,
            model_values,
            context="model",
        ),
        data=_construct_dataclass(
            DataConfig,
            data_values,
            context="data",
        ),
        training=_construct_dataclass(
            TrainingConfig,
            training_values,
            context="training",
        ),
        evaluation=_construct_dataclass(
            EvaluationConfig,
            evaluation_values,
            context="evaluation",
        ),
        output=_construct_dataclass(
            OutputConfig,
            output_values,
            context="output",
        ),
    )

    config.validate()
    return config


def load_experiment_config(
    path: str | Path,
) -> ExperimentConfig:
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        value = yaml.safe_load(handle)

    if value is None:
        raise ValueError(
            f"Configuration file is empty: {config_path}"
        )

    root = _require_mapping(
        value,
        context="experiment configuration",
    )

    return experiment_config_from_dict(root)
