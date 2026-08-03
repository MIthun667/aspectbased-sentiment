from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.aspect_sentiment.data import (
    EvidenceAwareDataset,
)
from src.aspect_sentiment.models.transformer import (
    CounterfactualBindingLoss,
    CounterfactualEvidenceBatchCollator,
    CounterfactualEvidenceTransformerDataset,
    EvidenceBindingLoss,
    EvidenceCompatibilityBindingTransformerClassifier,
    build_linear_warmup_scheduler,
    build_transformer_optimizer,
    train_counterfactual_binding_one_epoch,
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

    def word_ids(self, batch_index=0):
        return self._word_ids

    def sequence_ids(self, batch_index=0):
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
                for index in range(len(words))
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
            + list(range(len(text_pair)))
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


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.config = SimpleNamespace(
            hidden_size=8
        )

        self.embedding = nn.Embedding(
            64,
            8,
            padding_idx=0,
        )

        self.layer_norm = nn.LayerNorm(8)

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        token_type_ids=None,
    ):
        return SimpleNamespace(
            last_hidden_state=(
                self.layer_norm(
                    self.embedding(input_ids)
                )
            )
        )


def make_instances():
    canonical_records = []
    evidence_records = []

    examples = [
        (
            "one",
            [
                "battery",
                "is",
                "excellent",
            ],
            2,
            [2],
        ),
        (
            "two",
            [
                "screen",
                "looks",
                "quite",
                "average",
            ],
            1,
            [3],
        ),
        (
            "three",
            [
                "poor",
                "keyboard",
                "response",
            ],
            0,
            [0],
        ),
        (
            "four",
            [
                "speaker",
                "quality",
                "is",
                "really",
                "great",
            ],
            2,
            [3, 4],
        ),
        (
            "five",
            [
                "awful",
                "trackpad",
                "sensitivity",
                "today",
            ],
            0,
            [0],
        ),
        (
            "six",
            [
                "battery",
                "performance",
                "is",
                "ordinary",
            ],
            1,
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
                "aspect_text": "battery",
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
                "aspect_text": "battery",
                "aspect_start": 0,
                "aspect_end": 1,
                "polarity": polarity,
                "selected_evidence_indices": (
                    evidence_indices
                ),
                "selected_evidence_tokens": [
                    tokens[index]
                    for index in evidence_indices
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


def build_components():
    tokenizer = TinyFastTokenizer()

    dataset = (
        CounterfactualEvidenceTransformerDataset(
            make_instances(),
            tokenizer=tokenizer,
            maximum_length=16,
            seed=2026,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=(
            CounterfactualEvidenceBatchCollator(
                tokenizer=tokenizer
            )
        ),
    )

    model = (
        EvidenceCompatibilityBindingTransformerClassifier(
            backbone=TinyEncoder(),
            hidden_dimension=8,
            number_of_classes=3,
            dropout=0.0,
            compatibility_dimension=4,
        )
    )

    base_objective = EvidenceBindingLoss(
        combined_weight=1.0,
        context_weight=0.2,
        evidence_weight=0.5,
        agreement_weight=0.1,
    )

    counterfactual_objective = (
        CounterfactualBindingLoss(
            ranking_weight=0.2,
            probability_margin_weight=0.2,
            ranking_margin=0.2,
            probability_margin=0.05,
        )
    )

    optimizer = build_transformer_optimizer(
        model,
        learning_rate=1e-2,
        weight_decay=1e-3,
    )

    scheduler, _ = (
        build_linear_warmup_scheduler(
            optimizer,
            training_steps=4,
            warmup_ratio=0.25,
        )
    )

    return (
        loader,
        model,
        base_objective,
        counterfactual_objective,
        optimizer,
        scheduler,
    )


def test_counterfactual_training_epoch() -> None:
    (
        loader,
        model,
        base_objective,
        counterfactual_objective,
        optimizer,
        scheduler,
    ) = build_components()

    result = (
        train_counterfactual_binding_one_epoch(
            model=model,
            data_loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            base_objective=base_objective,
            counterfactual_objective=(
                counterfactual_objective
            ),
            device=torch.device("cpu"),
            gradient_clip_norm=1.0,
            use_bfloat16=False,
        )
    )

    assert np.isfinite(result.loss)
    assert np.isfinite(result.base_loss)
    assert result.number_of_instances == 6
    assert result.number_with_evidence == 5

    assert (
        result.number_valid_random
        > 0
    )

    assert (
        result.number_valid_cross
        > 0
    )


def test_counterfactual_components_are_finite() -> None:
    (
        loader,
        model,
        base_objective,
        counterfactual_objective,
        optimizer,
        scheduler,
    ) = build_components()

    result = (
        train_counterfactual_binding_one_epoch(
            model=model,
            data_loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            base_objective=base_objective,
            counterfactual_objective=(
                counterfactual_objective
            ),
            device=torch.device("cpu"),
            gradient_clip_norm=1.0,
            use_bfloat16=False,
        )
    )

    values = (
        result.combined_loss,
        result.context_loss,
        result.evidence_loss,
        result.agreement_loss,
        result.random_ranking_loss,
        result.random_probability_margin_loss,
        result.cross_ranking_loss,
        result.cross_probability_margin_loss,
    )

    assert all(
        np.isfinite(value)
        for value in values
    )


def test_compatibility_network_receives_gradients() -> None:
    (
        loader,
        model,
        base_objective,
        counterfactual_objective,
        optimizer,
        scheduler,
    ) = build_components()

    train_counterfactual_binding_one_epoch(
        model=model,
        data_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        base_objective=base_objective,
        counterfactual_objective=(
            counterfactual_objective
        ),
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
        use_bfloat16=False,
    )

    assert any(
        parameter.grad is not None
        for parameter in (
            model.compatibility_network
            .parameters()
        )
    )


def test_empty_loader_rejected() -> None:
    (
        _,
        model,
        base_objective,
        counterfactual_objective,
        optimizer,
        scheduler,
    ) = build_components()

    with pytest.raises(
        ValueError,
        match="no instances",
    ):
        train_counterfactual_binding_one_epoch(
            model=model,
            data_loader=[],
            optimizer=optimizer,
            scheduler=scheduler,
            base_objective=base_objective,
            counterfactual_objective=(
                counterfactual_objective
            ),
            device=torch.device("cpu"),
            gradient_clip_norm=1.0,
            use_bfloat16=False,
        )


def test_invalid_gradient_clip_rejected() -> None:
    (
        loader,
        model,
        base_objective,
        counterfactual_objective,
        optimizer,
        scheduler,
    ) = build_components()

    with pytest.raises(
        ValueError,
        match="gradient_clip_norm",
    ):
        train_counterfactual_binding_one_epoch(
            model=model,
            data_loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            base_objective=base_objective,
            counterfactual_objective=(
                counterfactual_objective
            ),
            device=torch.device("cpu"),
            gradient_clip_norm=0.0,
            use_bfloat16=False,
        )


def test_training_fixture_has_cross_negatives() -> None:
    (
        loader,
        _,
        _,
        _,
        _,
        _,
    ) = build_components()

    number_changed = 0

    for batch in loader:
        changed = (
            batch.shuffled_cross_instance
            .evidence_subword_mask
            != batch.selected
            .evidence_subword_mask
        ).any(dim=1)

        valid = (
            ~batch.selected.evidence_is_empty
            & ~batch.shuffled_cross_instance
                .evidence_is_empty
            & changed
        )

        number_changed += int(
            valid.sum().item()
        )

    assert number_changed > 0
