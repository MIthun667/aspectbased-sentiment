from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.aspect_sentiment.evidence import (
    WeakEvidenceConfig,
    build_split_manifest,
    build_weak_evidence_record,
    build_weak_evidence_records,
    canonical_json_dumps,
    group_records_by_sentence,
    read_jsonl,
    sha256_file,
    summarize_weak_evidence_records,
    write_jsonl,
)


def make_records() -> list[
    dict[str, object]
]:
    tokens = [
        "The",
        "cord",
        "is",
        "heavy",
        "but",
        "battery",
        "life",
        "is",
        "good",
        ".",
    ]

    pos_tags = [
        "DT",
        "NN",
        "VBZ",
        "JJ",
        "CC",
        "NN",
        "NN",
        "VBZ",
        "JJ",
        ".",
    ]

    dependency_heads = [
        1,
        3,
        3,
        -1,
        8,
        6,
        8,
        8,
        3,
        3,
    ]

    dependency_relations = [
        "det",
        "nsubj",
        "cop",
        "ROOT",
        "cc",
        "compound",
        "nsubj",
        "cop",
        "conj",
        "punct",
    ]

    common = {
        "sentence_id": "sentence-1",
        "domain": "laptops",
        "split": "train",
        "tokens": tokens,
        "pos_tags": pos_tags,
        "dependency_heads": (
            dependency_heads
        ),
        "dependency_relations": (
            dependency_relations
        ),
        "is_multi_aspect": True,
    }

    return [
        {
            **common,
            "instance_id": "sentence-1:a0",
            "aspect_text": "cord",
            "aspect_start": 1,
            "aspect_end": 2,
            "aspect_distances": [
                1,
                0,
                1,
                1,
                2,
                3,
                4,
                4,
                3,
                2,
            ],
            "polarity": "negative",
        },
        {
            **common,
            "instance_id": "sentence-1:a1",
            "aspect_text": "battery life",
            "aspect_start": 5,
            "aspect_end": 7,
            "aspect_distances": [
                4,
                3,
                3,
                2,
                2,
                0,
                0,
                1,
                1,
                2,
            ],
            "polarity": "positive",
        },
    ]


def test_config_validation() -> None:
    config = WeakEvidenceConfig()
    config.validate()

    assert config.maximum_tokens == 3
    assert config.minimum_score == 2.5


def test_invalid_configuration_rejected() -> None:
    config = WeakEvidenceConfig(
        maximum_tokens=0
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        config.validate()


def test_groups_records_by_sentence() -> None:
    records = make_records()

    grouped = group_records_by_sentence(
        records
    )

    assert list(grouped) == [
        "sentence-1"
    ]
    assert len(
        grouped["sentence-1"]
    ) == 2


def test_builds_one_artifact_record() -> None:
    records = make_records()
    config = WeakEvidenceConfig()

    artifact = build_weak_evidence_record(
        records[1],
        sentence_records=records,
        config=config,
    )

    assert artifact["schema_version"] == 1
    assert (
        artifact["instance_id"]
        == "sentence-1:a1"
    )
    assert artifact["aspect_text"] == (
        "battery life"
    )
    assert artifact["candidate_count"] >= 1
    assert isinstance(
        artifact["candidates"],
        list,
    )


def test_build_preserves_record_order() -> None:
    records = make_records()

    artifacts = (
        build_weak_evidence_records(
            records,
            config=WeakEvidenceConfig(),
        )
    )

    assert [
        record["instance_id"]
        for record in artifacts
    ] == [
        "sentence-1:a0",
        "sentence-1:a1",
    ]


def test_artifact_is_deterministic() -> None:
    records = make_records()
    config = WeakEvidenceConfig()

    first = build_weak_evidence_records(
        records,
        config=config,
    )

    second = build_weak_evidence_records(
        records,
        config=config,
    )

    assert canonical_json_dumps(
        first
    ) == canonical_json_dumps(second)


def test_summary_statistics() -> None:
    artifacts = (
        build_weak_evidence_records(
            make_records(),
            config=WeakEvidenceConfig(),
        )
    )

    summary = (
        summarize_weak_evidence_records(
            artifacts
        )
    )

    assert summary["record_count"] == 2
    assert 0.0 <= summary["coverage"] <= 1.0
    assert (
        summary["maximum_selected_tokens"]
        <= 3
    )


def test_jsonl_round_trip(
    tmp_path: Path,
) -> None:
    artifacts = (
        build_weak_evidence_records(
            make_records(),
            config=WeakEvidenceConfig(),
        )
    )

    output_path = (
        tmp_path / "evidence.jsonl"
    )

    write_jsonl(
        artifacts,
        output_path=output_path,
    )

    loaded = read_jsonl(output_path)

    assert loaded == artifacts


def test_jsonl_is_stable(
    tmp_path: Path,
) -> None:
    artifacts = (
        build_weak_evidence_records(
            make_records(),
            config=WeakEvidenceConfig(),
        )
    )

    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    write_jsonl(
        artifacts,
        output_path=first_path,
    )

    write_jsonl(
        artifacts,
        output_path=second_path,
    )

    assert (
        first_path.read_bytes()
        == second_path.read_bytes()
    )


def test_builds_manifest(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.jsonl"

    source_records = make_records()

    source_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in source_records
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = (
        build_weak_evidence_records(
            source_records,
            config=WeakEvidenceConfig(),
        )
    )

    output_path = tmp_path / "output.jsonl"

    write_jsonl(
        artifacts,
        output_path=output_path,
    )

    manifest = build_split_manifest(
        source_path=source_path,
        output_path=output_path,
        records=artifacts,
        config=WeakEvidenceConfig(),
        generated_at_utc=(
            "2026-08-03T10:00:00+00:00"
        ),
    )

    assert (
        manifest["generated_at_utc"]
        == "2026-08-03T10:00:00+00:00"
    )
    assert (
        manifest["source_sha256"]
        == sha256_file(source_path)
    )
    assert (
        manifest["output_sha256"]
        == sha256_file(output_path)
    )
    assert (
        manifest["statistics"][
            "record_count"
        ]
        == 2
    )
