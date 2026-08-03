from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    instance_id: str
    sentence_id: str
    domain: str
    split: str

    target: str
    gold_label: str
    gold_label_id: int
    predicted_label: str
    predicted_label_id: int

    probabilities: tuple[float, float, float] | None = None

    def validate(self) -> None:
        valid_labels = {
            "negative",
            "neutral",
            "positive",
        }

        if self.gold_label not in valid_labels:
            raise ValueError(
                f"Invalid gold label: {self.gold_label!r}"
            )

        if self.predicted_label not in valid_labels:
            raise ValueError(
                "Invalid predicted label: "
                f"{self.predicted_label!r}"
            )

        if self.gold_label_id not in {0, 1, 2}:
            raise ValueError(
                "gold_label_id must be 0, 1, or 2"
            )

        if self.predicted_label_id not in {0, 1, 2}:
            raise ValueError(
                "predicted_label_id must be 0, 1, or 2"
            )

        if self.probabilities is not None:
            if len(self.probabilities) != 3:
                raise ValueError(
                    "probabilities must contain three values"
                )

            if any(
                probability < 0.0 or probability > 1.0
                for probability in self.probabilities
            ):
                raise ValueError(
                    "probabilities must be within [0, 1]"
                )

            if abs(sum(self.probabilities) - 1.0) > 1e-6:
                raise ValueError(
                    "probabilities must sum to 1"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RunMetadata:
    schema_version: str
    experiment_name: str
    model_name: str
    seed: int
    status: str

    started_at_utc: str
    completed_at_utc: str | None

    git_commit: str | None
    git_branch: str | None
    git_dirty: bool | None

    python_version: str
    platform: str
    torch_version: str
    numpy_version: str

    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None

    command: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
