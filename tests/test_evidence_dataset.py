from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.aspect_sentiment.data import (
    EvidenceAwareDataset,
    evidence_split_path,
    join_canonical_and_evidence_records,
)


def canonical_record(
    *,
    instance_id: str = "laptops:train:s1:a0",
) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "sentence_id": "laptops:train:s1",
        "domain": "laptops",
        "split": "train",
        "tokens": [
            "battery",
            "life",
            "is",
            "excellent",
        ],
        "aspect_text": "battery life",
        "aspect_start": 0,
        "aspect_end": 2,
        "polarity": "positive",
        "label_id": 2,
    }


def evidence_record(
    *,
    instance_id: str = "laptops:train:s1:a0",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "instance_id": instance_id,
        "sentence_id": "laptops:train:s1",
        "domain": "laptops",
        "split": "train",
        "tokens": [
            "battery",
            "life",
            "is",
            "excellent",
        ],
        "aspect_text": "battery life",
        "aspect_start": 0,
        "aspect_end": 2,
        "polarity": "positive",
        "selected_evidence_indices": [
            3
        ],
        "selected_evidence_tokens": [
            "excellent"
        ],
        "evidence_is_empty": False,
        "candidate_count": 1,
        "selection_config": {
            "minimum_score": 2.5,
            "maximum_tokens": 3,
        },
    }


def write_jsonl(
    path: Path,
    records: list[
        dict[str, object]
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )


def test_evidence_split_path() -> None:
    assert evidence_split_path(
        "data/derived/evidence",
        domain="laptops",
        split="train",
    ) == Path(
        "data/derived/evidence/"
        "laptops/train.jsonl"
    )


def test_join_builds_instance() -> None:
    instances = (
        join_canonical_and_evidence_records(
            [canonical_record()],
            [evidence_record()],
        )
    )

    assert len(instances) == 1

    instance = instances[0]

    assert instance.instance_id == (
        "laptops:train:s1:a0"
    )
    assert instance.label_id == 2
    assert (
        instance.selected_evidence_indices
        == (3,)
    )
    assert (
        instance.selected_evidence_tokens
        == ("excellent",)
    )
    assert instance.evidence_mask == (
        False,
        False,
        False,
        True,
    )
    assert not instance.evidence_is_empty


def test_dataset_preserves_canonical_order() -> None:
    canonical_records = [
        canonical_record(
            instance_id="second"
        ),
        canonical_record(
            instance_id="first"
        ),
    ]

    evidence_records = [
        evidence_record(
            instance_id="first"
        ),
        evidence_record(
            instance_id="second"
        ),
    ]

    dataset = EvidenceAwareDataset(
        canonical_records,
        evidence_records,
    )

    assert [
        dataset[index].instance_id
        for index in range(len(dataset))
    ] == [
        "second",
        "first",
    ]


def test_empty_evidence_builds_empty_mask() -> None:
    evidence = evidence_record()
    evidence[
        "selected_evidence_indices"
    ] = []
    evidence[
        "selected_evidence_tokens"
    ] = []
    evidence["evidence_is_empty"] = True
    evidence["candidate_count"] = 0

    dataset = EvidenceAwareDataset(
        [canonical_record()],
        [evidence],
    )

    instance = dataset[0]

    assert (
        instance.selected_evidence_indices
        == ()
    )
    assert instance.evidence_mask == (
        False,
        False,
        False,
        False,
    )
    assert instance.evidence_is_empty


def test_duplicate_canonical_id_rejected() -> None:
    record = canonical_record()

    with pytest.raises(
        ValueError,
        match="Duplicate canonical",
    ):
        EvidenceAwareDataset(
            [record, dict(record)],
            [evidence_record()],
        )


def test_duplicate_evidence_id_rejected() -> None:
    record = evidence_record()

    with pytest.raises(
        ValueError,
        match="Duplicate evidence",
    ):
        EvidenceAwareDataset(
            [canonical_record()],
            [record, dict(record)],
        )


def test_missing_evidence_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Missing evidence",
    ):
        EvidenceAwareDataset(
            [canonical_record()],
            [
                evidence_record(
                    instance_id="unknown"
                )
            ],
        )


def test_unknown_evidence_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="unknown instance_ids",
    ):
        EvidenceAwareDataset(
            [canonical_record()],
            [
                evidence_record(),
                evidence_record(
                    instance_id="extra"
                ),
            ],
        )


def test_token_mismatch_rejected() -> None:
    evidence = evidence_record()
    evidence["tokens"] = [
        "different",
        "tokens",
    ]

    with pytest.raises(
        ValueError,
        match="tokens mismatch",
    ):
        EvidenceAwareDataset(
            [canonical_record()],
            [evidence],
        )


def test_aspect_span_mismatch_rejected() -> None:
    evidence = evidence_record()
    evidence["aspect_end"] = 1

    with pytest.raises(
        ValueError,
        match="aspect_end mismatch",
    ):
        EvidenceAwareDataset(
            [canonical_record()],
            [evidence],
        )


def test_polarity_mismatch_rejected() -> None:
    evidence = evidence_record()
    evidence["polarity"] = "negative"

    with pytest.raises(
        ValueError,
        match="polarity mismatch",
    ):
        EvidenceAwareDataset(
            [canonical_record()],
            [evidence],
        )


def test_out_of_range_index_rejected() -> None:
    evidence = evidence_record()
    evidence[
        "selected_evidence_indices"
    ] = [9]
    evidence[
        "selected_evidence_tokens"
    ] = ["missing"]

    with pytest.raises(
        ValueError,
        match="out of range",
    ):
        EvidenceAwareDataset(
            [canonical_record()],
            [evidence],
        )


def test_selected_token_mismatch_rejected() -> None:
    evidence = evidence_record()
    evidence[
        "selected_evidence_tokens"
    ] = ["wrong"]

    with pytest.raises(
        ValueError,
        match="do not match indices",
    ):
        EvidenceAwareDataset(
            [canonical_record()],
            [evidence],
        )


def test_unsorted_indices_rejected() -> None:
    canonical = canonical_record()
    canonical["tokens"] = [
        "very",
        "good",
        "battery",
        "life",
    ]
    canonical["aspect_start"] = 2
    canonical["aspect_end"] = 4

    evidence = evidence_record()
    evidence["tokens"] = list(
        canonical["tokens"]
    )
    evidence["aspect_start"] = 2
    evidence["aspect_end"] = 4
    evidence[
        "selected_evidence_indices"
    ] = [1, 0]
    evidence[
        "selected_evidence_tokens"
    ] = ["good", "very"]
    evidence["candidate_count"] = 2

    with pytest.raises(
        ValueError,
        match="must be sorted",
    ):
        EvidenceAwareDataset(
            [canonical],
            [evidence],
        )


def test_from_split_loads_realistic_files(
    tmp_path: Path,
) -> None:
    processed_root = (
        tmp_path / "processed"
    )

    evidence_root = (
        tmp_path / "evidence"
    )

    write_jsonl(
        processed_root
        / "laptops"
        / "train.jsonl",
        [canonical_record()],
    )

    write_jsonl(
        evidence_root
        / "laptops"
        / "train.jsonl",
        [evidence_record()],
    )

    dataset = (
        EvidenceAwareDataset.from_split(
            processed_root=processed_root,
            evidence_root=evidence_root,
            domain="laptops",
            split="train",
        )
    )

    assert len(dataset) == 1
    assert dataset[0].aspect_text == (
        "battery life"
    )


def test_load_evidence_jsonl_reads_artifacts(
    tmp_path: Path,
) -> None:
    from src.aspect_sentiment.data import (
        load_evidence_jsonl,
    )

    path = tmp_path / "evidence.jsonl"

    write_jsonl(
        path,
        [evidence_record()],
    )

    records = load_evidence_jsonl(path)

    assert len(records) == 1
    assert (
        records[0]["instance_id"]
        == "laptops:train:s1:a0"
    )


def test_load_evidence_jsonl_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    from src.aspect_sentiment.data import (
        load_evidence_jsonl,
    )

    path = tmp_path / "evidence.jsonl"

    path.write_text(
        "{broken json}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid JSON",
    ):
        load_evidence_jsonl(path)


def test_load_evidence_jsonl_rejects_empty_file(
    tmp_path: Path,
) -> None:
    from src.aspect_sentiment.data import (
        load_evidence_jsonl,
    )

    path = tmp_path / "evidence.jsonl"

    path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="no records",
    ):
        load_evidence_jsonl(path)
