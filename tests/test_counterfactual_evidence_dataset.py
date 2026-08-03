from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from src.aspect_sentiment.data import (
    EvidenceAwareDataset,
)
from src.aspect_sentiment.models.transformer import (
    CounterfactualEvidenceBatchCollator,
    CounterfactualEvidenceTransformerDataset,
)


class TinyFastEncoding(dict):
    def __init__(
        self,
        values,
        *,
        word_ids,
        sequence_ids,
    ):
        super().__init__(values)

        self._word_ids = word_ids
        self._sequence_ids = sequence_ids

    def word_ids(
        self,
        batch_index=0,
    ):
        return self._word_ids

    def sequence_ids(
        self,
        batch_index=0,
    ):
        return self._sequence_ids


class TinyFastTokenizer:
    is_fast = True
    pad_token_id = 0
    padding_side = "right"

    model_input_names = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ]

    def __call__(
        self,
        words,
        *,
        text_pair,
        is_split_into_words,
        truncation,
        max_length,
        padding,
        return_attention_mask,
        return_token_type_ids,
    ):
        input_ids = (
            [30]
            + [
                index + 1
                for index in range(
                    len(words)
                )
            ]
            + [31]
            + [
                index + 20
                for index in range(
                    len(text_pair)
                )
            ]
            + [31]
        )[:max_length]

        word_ids = (
            [None]
            + list(range(len(words)))
            + [None]
            + list(
                range(len(text_pair))
            )
            + [None]
        )[:max_length]

        sequence_ids = (
            [None]
            + [0] * len(words)
            + [None]
            + [1] * len(text_pair)
            + [None]
        )[:max_length]

        return TinyFastEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": (
                    [1] * len(input_ids)
                ),
                "token_type_ids": [
                    0
                    if sequence_id != 1
                    else 1
                    for sequence_id
                    in sequence_ids
                ],
            },
            word_ids=word_ids,
            sequence_ids=sequence_ids,
        )

    def pad(
        self,
        features,
        *,
        padding,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
    ):
        target_length = max(
            len(feature["input_ids"])
            for feature in features
        )

        output = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
        }

        for feature in features:
            difference = (
                target_length
                - len(feature["input_ids"])
            )

            for key in output:
                output[key].append(
                    feature[key]
                    + [0] * difference
                )

        return {
            key: torch.tensor(
                value,
                dtype=torch.long,
            )
            for key, value in output.items()
        }


def make_instances():
    canonical_records = []
    evidence_records = []

    examples = [
        (
            "one",
            [
                "battery",
                "life",
                "is",
                "excellent",
            ],
            2,
            [3],
        ),
        (
            "two",
            [
                "screen",
                "looks",
                "fairly",
                "average",
            ],
            1,
            [2, 3],
        ),
        (
            "three",
            [
                "keyboard",
                "feels",
                "very",
                "poor",
            ],
            0,
            [3],
        ),
        (
            "four",
            [
                "speaker",
                "quality",
                "is",
                "great",
            ],
            2,
            [],
        ),
    ]

    for (
        instance_id,
        tokens,
        label_id,
        evidence_indices,
    ) in examples:
        polarity = {
            0: "negative",
            1: "neutral",
            2: "positive",
        }[label_id]

        canonical_records.append(
            {
                "instance_id": instance_id,
                "sentence_id": (
                    f"{instance_id}:sentence"
                ),
                "domain": "laptops",
                "split": "train",
                "tokens": tokens,
                "aspect_text": tokens[0],
                "aspect_start": 0,
                "aspect_end": 1,
                "polarity": polarity,
                "label_id": label_id,
            }
        )

        evidence_records.append(
            {
                "schema_version": 1,
                "instance_id": instance_id,
                "sentence_id": (
                    f"{instance_id}:sentence"
                ),
                "domain": "laptops",
                "split": "train",
                "tokens": tokens,
                "aspect_text": tokens[0],
                "aspect_start": 0,
                "aspect_end": 1,
                "polarity": polarity,
                "selected_evidence_indices": (
                    evidence_indices
                ),
                "selected_evidence_tokens": [
                    tokens[index]
                    for index
                    in evidence_indices
                ],
                "evidence_is_empty": (
                    len(evidence_indices) == 0
                ),
                "candidate_count": len(
                    evidence_indices
                ),
                "selection_config": {},
            }
        )

    dataset = EvidenceAwareDataset(
        canonical_records,
        evidence_records,
    )

    return [
        dataset[index]
        for index in range(len(dataset))
    ]


def build_dataset(
    *,
    seed: int = 2026,
):
    return (
        CounterfactualEvidenceTransformerDataset(
            make_instances(),
            tokenizer=TinyFastTokenizer(),
            maximum_length=32,
            seed=seed,
        )
    )


def test_counterfactual_dataset_has_three_views() -> None:
    dataset = build_dataset()

    assert len(dataset) == 4

    item = dataset[0]

    assert (
        item.selected.instance_id
        == item.random_same_sentence.instance_id
        == item.shuffled_cross_instance.instance_id
    )


def test_counterfactual_views_preserve_inputs() -> None:
    dataset = build_dataset()

    for item in dataset:
        for counterfactual in (
            item.random_same_sentence,
            item.shuffled_cross_instance,
        ):
            assert (
                counterfactual.input_ids
                == item.selected.input_ids
            )

            assert (
                counterfactual.attention_mask
                == item.selected.attention_mask
            )

            assert (
                counterfactual
                .aspect_subword_mask
                == item.selected
                .aspect_subword_mask
            )

            assert (
                counterfactual.label_id
                == item.selected.label_id
            )


def test_only_evidence_mask_may_change() -> None:
    dataset = build_dataset()

    changed_random = 0
    changed_cross = 0

    for item in dataset:
        changed_random += (
            item.random_same_sentence
            .evidence_subword_mask
            != item.selected
            .evidence_subword_mask
        )

        changed_cross += (
            item.shuffled_cross_instance
            .evidence_subword_mask
            != item.selected
            .evidence_subword_mask
        )

    assert changed_random > 0
    assert changed_cross > 0


def test_generation_is_deterministic() -> None:
    first = build_dataset(seed=2027)
    second = build_dataset(seed=2027)

    for first_item, second_item in zip(
        first,
        second,
        strict=True,
    ):
        assert (
            first_item
            .random_same_sentence
            .evidence_subword_mask
            == second_item
            .random_same_sentence
            .evidence_subword_mask
        )

        assert (
            first_item
            .shuffled_cross_instance
            .evidence_subword_mask
            == second_item
            .shuffled_cross_instance
            .evidence_subword_mask
        )


def test_cross_instance_has_other_donors() -> None:
    dataset = build_dataset(seed=2028)

    source_ids = {
        instance.instance_id
        for instance in make_instances()
    }

    for original, transformed in zip(
        make_instances(),
        dataset
        .cross_intervention_result
        .instances,
        strict=True,
    ):
        donor_id = (
            transformed.selection_config[
                "donor_instance_id"
            ]
        )

        assert donor_id in source_ids
        assert donor_id != original.instance_id


def test_empty_selected_evidence_stays_explicit() -> None:
    dataset = build_dataset()

    empty_item = dataset[3]

    assert (
        empty_item.selected
        .evidence_is_empty
    )

    assert (
        empty_item.random_same_sentence
        .evidence_is_empty
    )


def test_collator_returns_aligned_batches() -> None:
    dataset = build_dataset()

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=(
            CounterfactualEvidenceBatchCollator(
                tokenizer=TinyFastTokenizer()
            )
        ),
    )

    batch = next(iter(loader))

    assert batch.selected.labels.tolist() == [
        2,
        1,
        0,
        2,
    ]

    assert torch.equal(
        batch.selected.input_ids,
        batch.random_same_sentence.input_ids,
    )

    assert torch.equal(
        batch.selected.input_ids,
        batch.shuffled_cross_instance.input_ids,
    )

    assert torch.equal(
        batch.selected.labels,
        batch.random_same_sentence.labels,
    )

    assert torch.equal(
        batch.selected.labels,
        batch.shuffled_cross_instance.labels,
    )


def test_collated_evidence_masks_differ() -> None:
    dataset = build_dataset()

    collator = (
        CounterfactualEvidenceBatchCollator(
            tokenizer=TinyFastTokenizer()
        )
    )

    batch = collator(
        [
            dataset[index]
            for index in range(len(dataset))
        ]
    )

    assert not torch.equal(
        batch.selected.evidence_subword_mask,
        batch.random_same_sentence
        .evidence_subword_mask,
    )

    assert not torch.equal(
        batch.selected.evidence_subword_mask,
        batch.shuffled_cross_instance
        .evidence_subword_mask,
    )


def test_intervention_metadata_is_retained() -> None:
    dataset = build_dataset(seed=2029)

    assert (
        dataset.random_intervention_result.seed
        == 2029
    )

    assert (
        dataset.cross_intervention_result.seed
        == 2029
    )

    assert (
        dataset.random_intervention_result
        .number_changed
        > 0
    )


def test_empty_dataset_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        CounterfactualEvidenceTransformerDataset(
            [],
            tokenizer=TinyFastTokenizer(),
        )


def test_empty_counterfactual_batch_rejected() -> None:
    collator = (
        CounterfactualEvidenceBatchCollator(
            tokenizer=TinyFastTokenizer()
        )
    )

    with pytest.raises(
        ValueError,
        match="empty counterfactual",
    ):
        collator([])
