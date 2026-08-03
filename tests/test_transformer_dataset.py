from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from src.aspect_sentiment.models.transformer import (
    TransformerAspectDataset,
    TransformerBatchCollator,
    aspect_from_record,
    sentence_from_record,
)


MODEL_NAME = "microsoft/deberta-v3-base"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        local_files_only=True,
    )


def make_record(
    *,
    instance_id: str = "example-1",
    tokens: list[str] | None = None,
    aspect_text: str = "battery life",
    label_id: int = 2,
) -> dict[str, object]:
    if tokens is None:
        tokens = [
            "The",
            "battery",
            "life",
            "is",
            "excellent",
        ]

    return {
        "instance_id": instance_id,
        "sentence_id": (
            f"{instance_id}:sentence"
        ),
        "domain": "laptops",
        "split": "train",
        "tokens": tokens,
        "aspect_text": aspect_text,
        "aspect_start": 1,
        "aspect_end": 3,
        "polarity": {
            0: "negative",
            1: "neutral",
            2: "positive",
        }[label_id],
        "label_id": label_id,
    }


def test_sentence_from_record() -> None:
    record = make_record()

    assert sentence_from_record(record) == (
        "The battery life is excellent"
    )


def test_aspect_from_record() -> None:
    record = make_record()

    assert aspect_from_record(record) == (
        "battery life"
    )


def test_dataset_encodes_sentence_aspect_pair(
    tokenizer,
) -> None:
    dataset = TransformerAspectDataset(
        [make_record()],
        tokenizer=tokenizer,
        maximum_length=128,
    )

    instance = dataset[0]

    assert len(dataset) == 1
    assert instance.instance_id == "example-1"
    assert instance.label_id == 2
    assert len(instance.input_ids) > 0
    assert (
        len(instance.input_ids)
        == len(instance.attention_mask)
    )


def test_encoded_pair_contains_aspect_tokens(
    tokenizer,
) -> None:
    dataset = TransformerAspectDataset(
        [make_record()],
        tokenizer=tokenizer,
        maximum_length=128,
    )

    decoded = tokenizer.decode(
        dataset[0].input_ids,
        skip_special_tokens=False,
    )

    assert "battery" in decoded.lower()
    assert "life" in decoded.lower()


def test_collator_pads_batch(
    tokenizer,
) -> None:
    records = [
        make_record(
            instance_id="short",
            tokens=[
                "battery",
                "is",
                "good",
            ],
            aspect_text="battery",
        ),
        make_record(
            instance_id="long",
            tokens=[
                "the",
                "battery",
                "life",
                "of",
                "this",
                "laptop",
                "is",
                "excellent",
            ],
            aspect_text="battery life",
        ),
    ]

    dataset = TransformerAspectDataset(
        records,
        tokenizer=tokenizer,
        maximum_length=128,
    )

    batch = TransformerBatchCollator(
        tokenizer=tokenizer
    )(
        [
            dataset[0],
            dataset[1],
        ]
    )

    assert batch.input_ids.shape[0] == 2
    assert (
        batch.input_ids.shape
        == batch.attention_mask.shape
    )
    assert batch.labels.tolist() == [2, 2]
    assert batch.instance_ids == (
        "short",
        "long",
    )

    assert batch.input_ids.dtype == torch.long
    assert (
        batch.attention_mask.dtype
        == torch.long
    )


def test_maximum_length_is_respected(
    tokenizer,
) -> None:
    record = make_record(
        tokens=["word"] * 300,
        aspect_text="word",
    )

    dataset = TransformerAspectDataset(
        [record],
        tokenizer=tokenizer,
        maximum_length=32,
    )

    assert len(
        dataset[0].input_ids
    ) <= 32


def test_invalid_label_is_rejected(
    tokenizer,
) -> None:
    record = make_record()
    record["label_id"] = 4

    with pytest.raises(
        ValueError,
        match="label_id",
    ):
        TransformerAspectDataset(
            [record],
            tokenizer=tokenizer,
            maximum_length=128,
        )


def test_empty_dataset_is_rejected(
    tokenizer,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        TransformerAspectDataset(
            [],
            tokenizer=tokenizer,
            maximum_length=128,
        )
