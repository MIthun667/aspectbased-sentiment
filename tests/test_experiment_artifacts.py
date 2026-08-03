from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.aspect_sentiment.artifacts import (
    ExperimentArtifactWriter,
    PredictionRecord,
    collect_git_metadata,
    collect_run_metadata,
    utc_now_iso,
)
from src.aspect_sentiment.config import (
    experiment_config_from_dict,
)


def make_config(
    tmp_path: Path,
    *,
    overwrite: bool = False,
):
    return experiment_config_from_dict(
        {
            "schema_version": "1.0",
            "model": {
                "name": "majority_class",
            },
            "training": {
                "seed": 2026,
            },
            "output": {
                "root": str(tmp_path),
                "experiment_name": "test_run",
                "overwrite": overwrite,
            },
        }
    )


def test_prediction_record_serializes() -> None:
    record = PredictionRecord(
        instance_id="instance-1",
        sentence_id="sentence-1",
        domain="laptops",
        split="test",
        target="battery",
        gold_label="positive",
        gold_label_id=2,
        predicted_label="positive",
        predicted_label_id=2,
        probabilities=(0.1, 0.2, 0.7),
    )

    value = record.to_dict()

    assert value["target"] == "battery"
    assert value["predicted_label_id"] == 2
    assert value["probabilities"] == (
        0.1,
        0.2,
        0.7,
    )


def test_invalid_probabilities_are_rejected() -> None:
    record = PredictionRecord(
        instance_id="instance-1",
        sentence_id="sentence-1",
        domain="laptops",
        split="test",
        target="battery",
        gold_label="positive",
        gold_label_id=2,
        predicted_label="positive",
        predicted_label_id=2,
        probabilities=(0.2, 0.2, 0.2),
    )

    with pytest.raises(
        ValueError,
        match="sum to 1",
    ):
        record.to_dict()


def test_writer_creates_standard_layout(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    writer = ExperimentArtifactWriter(config)

    root = writer.prepare()

    assert root.is_dir()
    assert (root / "metrics").is_dir()
    assert (root / "predictions").is_dir()


def test_writer_rejects_existing_directory(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    writer = ExperimentArtifactWriter(config)

    writer.prepare()

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        writer.prepare()


def test_writer_overwrites_when_enabled(
    tmp_path: Path,
) -> None:
    first_config = make_config(tmp_path)
    first_writer = ExperimentArtifactWriter(
        first_config
    )
    root = first_writer.prepare()

    temporary_file = root / "temporary.txt"
    temporary_file.write_text(
        "remove me",
        encoding="utf-8",
    )

    second_config = make_config(
        tmp_path,
        overwrite=True,
    )
    second_writer = ExperimentArtifactWriter(
        second_config
    )
    second_writer.prepare()

    assert not temporary_file.exists()


def test_writer_saves_all_artifacts(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    writer = ExperimentArtifactWriter(config)
    writer.prepare()

    config_path = writer.write_resolved_config()

    started_at = utc_now_iso()

    metadata = collect_run_metadata(
        experiment_name=(
            config.output.experiment_name
        ),
        model_name=config.model.name,
        seed=config.training.seed,
        status="COMPLETE",
        started_at_utc=started_at,
        completed_at_utc=utc_now_iso(),
        repository_root=".",
    )

    metadata_path = writer.write_run_metadata(
        metadata
    )

    metrics_path = writer.write_metrics(
        domain="laptops",
        split="test",
        metrics={
            "accuracy": 0.75,
            "macro_f1": 0.70,
        },
    )

    predictions_path = writer.write_predictions(
        domain="laptops",
        split="test",
        predictions=[
            PredictionRecord(
                instance_id="instance-1",
                sentence_id="sentence-1",
                domain="laptops",
                split="test",
                target="battery",
                gold_label="positive",
                gold_label_id=2,
                predicted_label="positive",
                predicted_label_id=2,
                probabilities=(0.1, 0.2, 0.7),
            )
        ],
    )

    assert config_path.is_file()
    assert metadata_path.is_file()
    assert metrics_path.is_file()
    assert predictions_path.is_file()

    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metrics = json.load(handle)

    assert metrics["domain"] == "laptops"
    assert metrics["metrics"]["accuracy"] == 0.75

    prediction_lines = (
        predictions_path.read_text(
            encoding="utf-8"
        )
        .strip()
        .splitlines()
    )

    assert len(prediction_lines) == 1

    prediction = json.loads(
        prediction_lines[0]
    )

    assert prediction["target"] == "battery"


def test_git_metadata_has_expected_keys() -> None:
    metadata = collect_git_metadata(".")

    assert set(metadata) == {
        "git_commit",
        "git_branch",
        "git_dirty",
    }
