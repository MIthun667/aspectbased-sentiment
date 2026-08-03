from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_baseline import (
    labels_from_records,
    prediction_records,
    run_experiment,
)
from src.aspect_sentiment.config import (
    experiment_config_from_dict,
)
from src.aspect_sentiment.models.baselines import (
    MajorityClassBaseline,
)


def make_record(
    *,
    instance_id: str,
    split: str,
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
        "split": split,
        "tokens": [instance_id, "is", "good"],
        "aspect_text": instance_id,
        "aspect_start": 0,
        "aspect_end": 1,
        "polarity": polarity,
        "label_id": label_id,
    }


def write_split(
    root: Path,
    split: str,
    labels: list[int],
) -> None:
    path = (
        root
        / "laptops"
        / f"{split}.jsonl"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for index, label_id in enumerate(labels):
            record = make_record(
                instance_id=(
                    f"laptops:{split}:{index}"
                ),
                split=split,
                label_id=label_id,
            )

            handle.write(
                json.dumps(record)
            )
            handle.write("\n")


def test_labels_from_records() -> None:
    records = [
        make_record(
            instance_id="a",
            split="train",
            label_id=0,
        ),
        make_record(
            instance_id="b",
            split="train",
            label_id=2,
        ),
    ]

    assert labels_from_records(records) == [0, 2]


def test_prediction_records() -> None:
    records = [
        make_record(
            instance_id="a",
            split="test",
            label_id=2,
        )
    ]

    model = MajorityClassBaseline.fit(
        [0, 2, 2]
    )

    predictions = model.predict(1)
    probabilities = model.predict_proba(1)

    output = prediction_records(
        records,
        predictions,
        probabilities,
    )

    assert len(output) == 1
    assert output[0].predicted_label == "positive"
    assert output[0].gold_label == "positive"


def test_run_experiment_creates_artifacts(
    tmp_path: Path,
) -> None:
    processed_root = tmp_path / "processed"

    write_split(
        processed_root,
        "train",
        [0, 2, 2, 2],
    )
    write_split(
        processed_root,
        "validation",
        [0, 2],
    )
    write_split(
        processed_root,
        "test",
        [1, 2],
    )

    config = experiment_config_from_dict(
        {
            "schema_version": "1.0",
            "model": {
                "name": "majority_class",
            },
            "data": {
                "processed_root": str(
                    processed_root
                ),
                "train_domain": "laptops",
                "evaluation_domains": [
                    "laptops"
                ],
            },
            "training": {
                "seed": 2026,
                "epochs": 1,
            },
            "output": {
                "root": str(
                    tmp_path / "artifacts"
                ),
                "experiment_name": "test_majority",
            },
        }
    )

    output_directory = run_experiment(
        config,
        command="pytest",
    )

    assert (
        output_directory
        / "resolved_config.json"
    ).is_file()

    assert (
        output_directory
        / "run_metadata.json"
    ).is_file()

    assert (
        output_directory
        / "metrics"
        / "laptops__validation.json"
    ).is_file()

    assert (
        output_directory
        / "metrics"
        / "laptops__test.json"
    ).is_file()

    assert (
        output_directory
        / "predictions"
        / "laptops__validation.jsonl"
    ).is_file()

    assert (
        output_directory
        / "metrics"
        / "laptops__test_duplicate_excluded.json"
    ).is_file()

    assert (
        output_directory
        / "predictions"
        / "laptops__test_duplicate_excluded.jsonl"
    ).is_file()

    metadata = json.loads(
        (
            output_directory
            / "run_metadata.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert metadata["status"] == "COMPLETE"


def test_unsupported_model_is_rejected(
    tmp_path: Path,
) -> None:
    config = experiment_config_from_dict(
        {
            "schema_version": "1.0",
            "model": {
                "name": "unsupported",
            },
            "output": {
                "root": str(tmp_path),
                "experiment_name": "unsupported",
            },
        }
    )

    with pytest.raises(
        ValueError,
        match="Unsupported baseline model",
    ):
        run_experiment(config)


@pytest.mark.parametrize(
    "representation",
    [
        "sentence",
        "target_marked",
    ],
)
def test_run_tfidf_experiment_creates_artifacts(
    tmp_path: Path,
    representation: str,
) -> None:
    processed_root = tmp_path / "processed"

    write_split(
        processed_root,
        "train",
        [0, 1, 2, 0, 1, 2],
    )
    write_split(
        processed_root,
        "validation",
        [0, 1, 2],
    )
    write_split(
        processed_root,
        "test",
        [0, 1, 2],
    )

    config = experiment_config_from_dict(
        {
            "schema_version": "1.0",
            "model": {
                "name": "tfidf_logistic_regression",
                "parameters": {
                    "representation": representation,
                    "ngram_range": [1, 2],
                    "min_df": 1,
                    "max_features": None,
                    "sublinear_tf": True,
                    "C": 1.0,
                    "max_iter": 200,
                },
            },
            "data": {
                "processed_root": str(
                    processed_root
                ),
                "train_domain": "laptops",
                "evaluation_domains": [
                    "laptops"
                ],
            },
            "training": {
                "seed": 2026,
                "epochs": 1,
            },
            "evaluation": {
                "duplicate_excluded_sensitivity": False,
            },
            "output": {
                "root": str(
                    tmp_path / "artifacts"
                ),
                "experiment_name": (
                    f"tfidf_{representation}"
                ),
            },
        }
    )

    output_directory = run_experiment(
        config,
        command="pytest",
    )

    assert (
        output_directory
        / "metrics"
        / "laptops__validation.json"
    ).is_file()

    assert (
        output_directory
        / "metrics"
        / "laptops__test.json"
    ).is_file()

    predictions_path = (
        output_directory
        / "predictions"
        / "laptops__test.jsonl"
    )

    assert predictions_path.is_file()

    rows = [
        json.loads(line)
        for line in predictions_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(rows) == 3
    assert all(
        len(row["probabilities"]) == 3
        for row in rows
    )


@pytest.mark.parametrize(
    "representation",
    [
        "sentence",
        "target_marked",
    ],
)
def test_run_tfidf_experiment_creates_artifacts(
    tmp_path: Path,
    representation: str,
) -> None:
    processed_root = tmp_path / "processed"

    write_split(
        processed_root,
        "train",
        [0, 1, 2, 0, 1, 2],
    )
    write_split(
        processed_root,
        "validation",
        [0, 1, 2],
    )
    write_split(
        processed_root,
        "test",
        [0, 1, 2],
    )

    config = experiment_config_from_dict(
        {
            "schema_version": "1.0",
            "model": {
                "name": "tfidf_logistic_regression",
                "parameters": {
                    "representation": representation,
                    "ngram_range": [1, 2],
                    "min_df": 1,
                    "max_features": None,
                    "sublinear_tf": True,
                    "C": 1.0,
                    "max_iter": 200,
                },
            },
            "data": {
                "processed_root": str(
                    processed_root
                ),
                "train_domain": "laptops",
                "evaluation_domains": [
                    "laptops"
                ],
            },
            "training": {
                "seed": 2026,
                "epochs": 1,
            },
            "evaluation": {
                "duplicate_excluded_sensitivity": False,
            },
            "output": {
                "root": str(
                    tmp_path / "artifacts"
                ),
                "experiment_name": (
                    f"tfidf_{representation}"
                ),
            },
        }
    )

    output_directory = run_experiment(
        config,
        command="pytest",
    )

    assert (
        output_directory
        / "metrics"
        / "laptops__validation.json"
    ).is_file()

    assert (
        output_directory
        / "metrics"
        / "laptops__test.json"
    ).is_file()

    predictions_path = (
        output_directory
        / "predictions"
        / "laptops__test.jsonl"
    )

    assert predictions_path.is_file()

    rows = [
        json.loads(line)
        for line in predictions_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(rows) == 3
    assert all(
        len(row["probabilities"]) == 3
        for row in rows
    )
