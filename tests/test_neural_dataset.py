from __future__ import annotations

import torch

from src.aspect_sentiment.models.neural import (
    AspectSentimentDataset,
    NeuralBatchCollator,
    Vocabulary,
)


def make_record(
    *,
    instance_id: str,
    tokens: list[str],
    aspect_start: int,
    aspect_end: int,
    label_id: int,
) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "tokens": tokens,
        "aspect_start": aspect_start,
        "aspect_end": aspect_end,
        "label_id": label_id,
    }


def records() -> list[dict[str, object]]:
    return [
        make_record(
            instance_id="one",
            tokens=[
                "battery",
                "is",
                "excellent",
            ],
            aspect_start=0,
            aspect_end=1,
            label_id=2,
        ),
        make_record(
            instance_id="two",
            tokens=[
                "the",
                "screen",
                "quality",
                "is",
                "poor",
            ],
            aspect_start=1,
            aspect_end=3,
            label_id=0,
        ),
    ]


def test_dataset_encodes_records() -> None:
    source_records = records()
    vocabulary = Vocabulary.build(
        source_records
    )

    dataset = AspectSentimentDataset(
        source_records,
        vocabulary=vocabulary,
        maximum_length=32,
    )

    assert len(dataset) == 2
    assert dataset[0].instance_id == "one"
    assert dataset[0].label_id == 2
    assert len(dataset[0].token_ids) > 0


def test_collator_pads_variable_lengths() -> None:
    source_records = records()
    vocabulary = Vocabulary.build(
        source_records
    )

    dataset = AspectSentimentDataset(
        source_records,
        vocabulary=vocabulary,
        maximum_length=32,
    )

    batch = NeuralBatchCollator(
        pad_id=vocabulary.pad_id
    )(
        [
            dataset[0],
            dataset[1],
        ]
    )

    assert batch.input_ids.shape[0] == 2
    assert batch.input_ids.shape == (
        batch.attention_mask.shape
    )
    assert batch.labels.tolist() == [2, 0]
    assert batch.instance_ids == (
        "one",
        "two",
    )

    assert batch.input_ids.dtype == torch.long
    assert batch.attention_mask.dtype == torch.bool
    assert batch.labels.dtype == torch.long


def test_collator_attention_mask_matches_lengths() -> None:
    source_records = records()
    vocabulary = Vocabulary.build(
        source_records
    )

    dataset = AspectSentimentDataset(
        source_records,
        vocabulary=vocabulary,
        maximum_length=32,
    )

    batch = NeuralBatchCollator(
        pad_id=vocabulary.pad_id
    )(
        [
            dataset[0],
            dataset[1],
        ]
    )

    expected_lengths = [
        len(dataset[0].token_ids),
        len(dataset[1].token_ids),
    ]

    actual_lengths = (
        batch.attention_mask.sum(
            dim=1
        ).tolist()
    )

    assert actual_lengths == expected_lengths
