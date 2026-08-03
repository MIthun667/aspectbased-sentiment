from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.aspect_sentiment.evidence import (
    WeakEvidenceConfig,
    load_weak_evidence_config,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


def write_source_records(
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = [
        {
            "instance_id": "s1:a0",
            "sentence_id": "s1",
            "domain": "laptops",
            "split": "train",
            "tokens": [
                "Battery",
                "life",
                "is",
                "good",
                ".",
            ],
            "pos_tags": [
                "NN",
                "NN",
                "VBZ",
                "JJ",
                ".",
            ],
            "dependency_heads": [
                1,
                3,
                3,
                -1,
                3,
            ],
            "dependency_relations": [
                "compound",
                "nsubj",
                "cop",
                "ROOT",
                "punct",
            ],
            "aspect_distances": [
                0,
                0,
                1,
                1,
                2,
            ],
            "aspect_text": "Battery life",
            "aspect_start": 0,
            "aspect_end": 2,
            "polarity": "positive",
            "is_multi_aspect": False,
        }
    ]

    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )


def write_config(
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "candidate_generation:",
                "  maximum_dependency_distance: 3",
                "selection:",
                "  minimum_score: 2.5",
                "  maximum_tokens: 3",
                "  allow_ties: false",
                (
                    "  exclude_competing_"
                    "predicate_clauses: true"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_loads_yaml_config(
    tmp_path: Path,
) -> None:
    config_path = (
        tmp_path / "config.yaml"
    )

    write_config(config_path)

    config = load_weak_evidence_config(
        config_path
    )

    assert config == WeakEvidenceConfig()


def test_rejects_invalid_yaml_structure(
    tmp_path: Path,
) -> None:
    config_path = (
        tmp_path / "config.yaml"
    )

    config_path.write_text(
        "- invalid\n- configuration\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="mapping",
    ):
        load_weak_evidence_config(
            config_path
        )


def test_cli_builds_artifacts(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "processed"
    output_root = tmp_path / "derived"
    config_path = tmp_path / "config.yaml"

    write_config(config_path)

    write_source_records(
        input_root
        / "laptops"
        / "train.jsonl"
    )

    command = [
        sys.executable,
        str(
            PROJECT_ROOT
            / "scripts"
            / "build_weak_evidence.py"
        ),
        "--config",
        str(config_path),
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
        "--domains",
        "laptops",
        "--splits",
        "train",
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        result.stdout + result.stderr
    )

    evidence_path = (
        output_root
        / "laptops"
        / "train.jsonl"
    )

    split_manifest_path = (
        output_root
        / "laptops"
        / "train.manifest.json"
    )

    domain_manifest_path = (
        output_root
        / "laptops"
        / "manifest.json"
    )

    assert evidence_path.is_file()
    assert split_manifest_path.is_file()
    assert domain_manifest_path.is_file()

    record = json.loads(
        evidence_path.read_text(
            encoding="utf-8"
        ).splitlines()[0]
    )

    assert record["instance_id"] == "s1:a0"
    assert (
        record["selected_evidence_tokens"]
        == ["good"]
    )


def test_cli_requires_overwrite(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "processed"
    output_root = tmp_path / "derived"
    config_path = tmp_path / "config.yaml"

    write_config(config_path)

    write_source_records(
        input_root
        / "laptops"
        / "train.jsonl"
    )

    command = [
        sys.executable,
        str(
            PROJECT_ROOT
            / "scripts"
            / "build_weak_evidence.py"
        ),
        "--config",
        str(config_path),
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
        "--domains",
        "laptops",
        "--splits",
        "train",
    ]

    first = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert first.returncode == 0

    second = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert second.returncode != 0
    assert "Use --overwrite" in (
        second.stdout + second.stderr
    )

    third = subprocess.run(
        command + ["--overwrite"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert third.returncode == 0
