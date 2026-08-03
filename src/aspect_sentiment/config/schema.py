from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


VALID_DOMAINS = {
    "laptops",
    "restaurants",
    "tweets",
}

VALID_SPLITS = {
    "train",
    "validation",
    "test",
}

VALID_SELECTION_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "negative_log_likelihood",
    "brier_score",
}


@dataclass(frozen=True, slots=True)
class DataConfig:
    processed_root: str = "data/processed"
    train_domain: str = "laptops"
    evaluation_domains: tuple[str, ...] = (
        "laptops",
    )
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"

    def validate(self) -> None:
        if self.train_domain not in VALID_DOMAINS:
            raise ValueError(
                "Unsupported train_domain: "
                f"{self.train_domain!r}"
            )

        if not self.evaluation_domains:
            raise ValueError(
                "evaluation_domains must not be empty"
            )

        invalid_domains = sorted(
            set(self.evaluation_domains)
            - VALID_DOMAINS
        )

        if invalid_domains:
            raise ValueError(
                "Unsupported evaluation domains: "
                f"{invalid_domains}"
            )

        configured_splits = {
            self.train_split,
            self.validation_split,
            self.test_split,
        }

        invalid_splits = sorted(
            configured_splits - VALID_SPLITS
        )

        if invalid_splits:
            raise ValueError(
                f"Unsupported splits: {invalid_splits}"
            )

        if len(configured_splits) != 3:
            raise ValueError(
                "train, validation, and test split "
                "names must be distinct"
            )

        if not self.processed_root.strip():
            raise ValueError(
                "processed_root must not be empty"
            )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    parameters: dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "model.name must not be empty"
            )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 2026
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    early_stopping_patience: int = 5
    selection_metric: str = "macro_f1"
    maximize_selection_metric: bool = True

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError(
                "training.seed must be non-negative"
            )

        if self.epochs <= 0:
            raise ValueError(
                "training.epochs must be positive"
            )

        if self.batch_size <= 0:
            raise ValueError(
                "training.batch_size must be positive"
            )

        if self.learning_rate <= 0.0:
            raise ValueError(
                "training.learning_rate must be positive"
            )

        if self.weight_decay < 0.0:
            raise ValueError(
                "training.weight_decay must be "
                "non-negative"
            )

        if self.early_stopping_patience < 0:
            raise ValueError(
                "training.early_stopping_patience must "
                "be non-negative"
            )

        if (
            self.selection_metric
            not in VALID_SELECTION_METRICS
        ):
            raise ValueError(
                "Unsupported selection metric: "
                f"{self.selection_metric!r}"
            )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    batch_size: int = 64
    save_probabilities: bool = True
    save_predictions: bool = True
    duplicate_excluded_sensitivity: bool = True

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(
                "evaluation.batch_size must be positive"
            )


@dataclass(frozen=True, slots=True)
class OutputConfig:
    root: str = "artifacts/experiments"
    experiment_name: str = "baseline"
    overwrite: bool = False

    def validate(self) -> None:
        if not self.root.strip():
            raise ValueError(
                "output.root must not be empty"
            )

        if not self.experiment_name.strip():
            raise ValueError(
                "output.experiment_name must not be empty"
            )

        invalid_characters = {
            "/",
            "\\",
        }

        if any(
            character in self.experiment_name
            for character in invalid_characters
        ):
            raise ValueError(
                "output.experiment_name must not "
                "contain path separators"
            )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    schema_version: str
    model: ModelConfig
    data: DataConfig = field(
        default_factory=DataConfig
    )
    training: TrainingConfig = field(
        default_factory=TrainingConfig
    )
    evaluation: EvaluationConfig = field(
        default_factory=EvaluationConfig
    )
    output: OutputConfig = field(
        default_factory=OutputConfig
    )
    description: str = ""

    def validate(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(
                "Unsupported configuration schema_version: "
                f"{self.schema_version!r}"
            )

        self.model.validate()
        self.data.validate()
        self.training.validate()
        self.evaluation.validate()
        self.output.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def output_directory(self) -> Path:
        return (
            Path(self.output.root)
            / self.output.experiment_name
            / f"seed_{self.training.seed}"
        )
