from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from src.aspect_sentiment.data import (
    EvidenceAwareDataset,
)
from src.aspect_sentiment.models.transformer import (
    EvidenceTransformerBatchCollator,
    EvidenceTransformerDataset,
)


MODEL_NAME = "microsoft/deberta-v3-base"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        local_files_only=True,
    )


def canonical_record(
    *,
    instance_id: str,
    tokens: list[str],
    aspect_start: int,
    aspect_end: int,
    label_id: int,
) -> dict[str, object]:
    polarity = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }[label_id]

    return {
        "instance_id": instance_id,
        "sentence_id": (
            f"{instance_id}:sentence"
        ),
        "domain": "laptops",
        "split": "train",
        "tokens": tokens,
        "aspect_text": " ".join(
            tokens[
                aspect_start:aspect_end
            ]
        ),
        "aspect_start": aspect_start,
        "aspect_end": aspect_end,
        "polarity": polarity,
        "label_id": label_id,
    }


def evidence_record(
    canonical: dict[str, object],
    *,
    selected_indices: list[int],
) -> dict[str, object]:
    tokens = canonical["tokens"]

    assert isinstance(tokens, list)

    return {
        "schema_version": 1,
        "instance_id": canonical[
            "instance_id"
        ],
        "sentence_id": canonical[
            "sentence_id"
        ],
        "domain": canonical["domain"],
        "split": canonical["split"],
        "tokens": tokens,
        "aspect_text": canonical[
            "aspect_text"
        ],
        "aspect_start": canonical[
            "aspect_start"
        ],
        "aspect_end": canonical[
            "aspect_end"
        ],
        "polarity": canonical[
            "polarity"
        ],
        "selected_evidence_indices": (
            selected_indices
        ),
        "selected_evidence_tokens": [
            tokens[index]
            for index in selected_indices
        ],
        "evidence_is_empty": (
            len(selected_indices) == 0
        ),
        "candidate_count": len(
            selected_indices
        ),
        "selection_config": {
            "maximum_tokens": 3,
        },
    }


def joined_instances():
    first = canonical_record(
        instance_id="one",
        tokens=[
            "The",
            "battery",
            "life",
            "is",
            "excellent",
        ],
        aspect_start=1,
        aspect_end=3,
        label_id=2,
    )

    second = canonical_record(
        instance_id="two",
        tokens=[
            "The",
            "screen",
            "is",
            "acceptable",
        ],
        aspect_start=1,
        aspect_end=2,
        label_id=1,
    )

    dataset = EvidenceAwareDataset(
        [first, second],
        [
            evidence_record(
                first,
                selected_indices=[4],
            ),
            evidence_record(
                second,
                selected_indices=[],
            ),
        ],
    )

    return [
        dataset[0],
        dataset[1],
    ]


def test_dataset_encodes_evidence_instances(
    tokenizer,
) -> None:
    dataset = EvidenceTransformerDataset(
        joined_instances(),
        tokenizer=tokenizer,
        maximum_length=128,
    )

    assert len(dataset) == 2

    first = dataset[0]

    assert first.instance_id == "one"
    assert first.label_id == 2
    assert any(
        first.aspect_subword_mask
    )
    assert any(
        first.evidence_subword_mask
    )
    assert not first.evidence_is_empty


def test_empty_evidence_produces_empty_mask(
    tokenizer,
) -> None:
    dataset = EvidenceTransformerDataset(
        joined_instances(),
        tokenizer=tokenizer,
        maximum_length=128,
    )

    second = dataset[1]

    assert second.evidence_is_empty
    assert not any(
        second.evidence_subword_mask
    )


def test_collator_pads_all_aligned_masks(
    tokenizer,
) -> None:
    dataset = EvidenceTransformerDataset(
        joined_instances(),
        tokenizer=tokenizer,
        maximum_length=128,
    )

    batch = (
        EvidenceTransformerBatchCollator(
            tokenizer=tokenizer
        )(
            [
                dataset[0],
                dataset[1],
            ]
        )
    )

    expected_shape = (
        batch.input_ids.shape
    )

    assert (
        batch.attention_mask.shape
        == expected_shape
    )
    assert (
        batch.aspect_subword_mask.shape
        == expected_shape
    )
    assert (
        batch.evidence_subword_mask.shape
        == expected_shape
    )
    assert (
        batch.sentence_subword_mask.shape
        == expected_shape
    )
    assert (
        batch.pair_aspect_subword_mask.shape
        == expected_shape
    )

    assert batch.labels.tolist() == [2, 1]

    assert (
        batch.evidence_is_empty.tolist()
        == [False, True]
    )

    assert batch.instance_ids == (
        "one",
        "two",
    )


def test_batch_dtypes(
    tokenizer,
) -> None:
    dataset = EvidenceTransformerDataset(
        joined_instances(),
        tokenizer=tokenizer,
        maximum_length=128,
    )

    batch = (
        EvidenceTransformerBatchCollator(
            tokenizer=tokenizer
        )(
            [
                dataset[0],
                dataset[1],
            ]
        )
    )

    assert batch.input_ids.dtype == (
        torch.long
    )
    assert batch.attention_mask.dtype == (
        torch.long
    )
    assert batch.labels.dtype == torch.long

    assert (
        batch.aspect_subword_mask.dtype
        == torch.bool
    )
    assert (
        batch.evidence_subword_mask.dtype
        == torch.bool
    )
    assert (
        batch.sentence_subword_mask.dtype
        == torch.bool
    )
    assert (
        batch.pair_aspect_subword_mask.dtype
        == torch.bool
    )
    assert (
        batch.evidence_is_empty.dtype
        == torch.bool
    )


def test_rejects_truncated_aspect(
    tokenizer,
) -> None:
    canonical = canonical_record(
        instance_id="long",
        tokens=[
            f"token{index}"
            for index in range(100)
        ],
        aspect_start=95,
        aspect_end=97,
        label_id=1,
    )

    joined = EvidenceAwareDataset(
        [canonical],
        [
            evidence_record(
                canonical,
                selected_indices=[90],
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="Aspect was truncated",
    ):
        EvidenceTransformerDataset(
            [joined[0]],
            tokenizer=tokenizer,
            maximum_length=24,
            reject_truncation=True,
        )


def test_can_allow_truncation(
    tokenizer,
) -> None:
    canonical = canonical_record(
        instance_id="long",
        tokens=[
            f"token{index}"
            for index in range(100)
        ],
        aspect_start=95,
        aspect_end=97,
        label_id=1,
    )

    joined = EvidenceAwareDataset(
        [canonical],
        [
            evidence_record(
                canonical,
                selected_indices=[90],
            )
        ],
    )

    dataset = EvidenceTransformerDataset(
        [joined[0]],
        tokenizer=tokenizer,
        maximum_length=24,
        reject_truncation=False,
    )

    assert len(dataset) == 1


def test_empty_dataset_rejected(
    tokenizer,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        EvidenceTransformerDataset(
            [],
            tokenizer=tokenizer,
            maximum_length=128,
        )


def test_empty_batch_rejected(
    tokenizer,
) -> None:
    collator = (
        EvidenceTransformerBatchCollator(
            tokenizer=tokenizer
        )
    )

    with pytest.raises(
        ValueError,
        match="empty batch",
    ):
        collator([])
