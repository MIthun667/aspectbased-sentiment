from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.aspect_sentiment.artifacts.schema import (
    PredictionRecord,
    RunMetadata,
)
from src.aspect_sentiment.config import (
    ExperimentConfig,
)


class ExperimentArtifactWriter:
    def __init__(
        self,
        config: ExperimentConfig,
    ) -> None:
        self.config = config
        self.root = config.output_directory()

    def prepare(self) -> Path:
        if self.root.exists():
            if not self.config.output.overwrite:
                raise FileExistsError(
                    "Experiment output directory already "
                    f"exists: {self.root}"
                )

            shutil.rmtree(self.root)

        (self.root / "metrics").mkdir(
            parents=True,
            exist_ok=True,
        )
        (self.root / "predictions").mkdir(
            parents=True,
            exist_ok=True,
        )

        return self.root

    def write_resolved_config(self) -> Path:
        return self._write_json(
            self.root / "resolved_config.json",
            self.config.to_dict(),
        )

    def write_run_metadata(
        self,
        metadata: RunMetadata,
    ) -> Path:
        return self._write_json(
            self.root / "run_metadata.json",
            metadata.to_dict(),
        )

    def write_metrics(
        self,
        *,
        domain: str,
        split: str,
        metrics: Mapping[str, Any],
    ) -> Path:
        payload = {
            "schema_version": "1.0",
            "experiment_name": (
                self.config.output.experiment_name
            ),
            "model_name": self.config.model.name,
            "seed": self.config.training.seed,
            "domain": domain,
            "split": split,
            "metrics": dict(metrics),
        }

        filename = (
            f"{domain}__{split}.json"
        )

        return self._write_json(
            self.root / "metrics" / filename,
            payload,
        )

    def write_predictions(
        self,
        *,
        domain: str,
        split: str,
        predictions: Iterable[PredictionRecord],
    ) -> Path:
        filename = (
            f"{domain}__{split}.jsonl"
        )
        path = self.root / "predictions" / filename

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for prediction in predictions:
                handle.write(
                    json.dumps(
                        prediction.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                handle.write("\n")

        return path

    @staticmethod
    def _write_json(
        path: Path,
        value: Mapping[str, Any],
    ) -> Path:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                value,
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.write("\n")

        return path
