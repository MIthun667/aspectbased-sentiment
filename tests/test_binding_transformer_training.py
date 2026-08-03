from __future__ import annotations

from pathlib import Path
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
    EvidenceBindingLoss,
    EvidenceBindingTransformerClassifier,
    EvidenceTransformerBatchCollator,
    EvidenceTransformerDataset,
    build_linear_warmup_scheduler,
    build_transformer_optimizer,
    evaluate_evidence_binding_model,
    load_evidence_binding_checkpoint,
    save_evidence_binding_checkpoint,
    train_evidence_binding_one_epoch,
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
                    self.embedding(
                        input_ids
                    )
                )
            )
        )


def make_instances():
    canonical_records = []
    evidence_records = []

    examples = [
        ("p1", "excellent", 2, [2]),
        ("n1", "awful", 0, [2]),
        ("u1", "average", 1, [2]),
        ("p2", "great", 2, [2]),
        ("n2", "poor", 0, [2]),
        ("u2", "ordinary", 1, []),
    ]

    for (
        instance_id,
        sentiment,
        label_id,
        evidence_indices,
    ) in examples:
        polarity = {
            0: "negative",
            1: "neutral",
            2: "positive",
        }[label_id]

        tokens = [
            "battery",
            "is",
            sentiment,
        ]

        canonical = {
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

        evidence = {
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

        canonical_records.append(
            canonical
        )
        evidence_records.append(
            evidence
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

    dataset = EvidenceTransformerDataset(
        make_instances(),
        tokenizer=tokenizer,
        maximum_length=16,
    )

    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=(
            EvidenceTransformerBatchCollator(
                tokenizer=tokenizer
            )
        ),
    )

    model = (
        EvidenceBindingTransformerClassifier(
            backbone=TinyEncoder(),
            hidden_dimension=8,
            number_of_classes=3,
            dropout=0.0,
            gate_dimension=4,
        )
    )

    objective = EvidenceBindingLoss(
        combined_weight=1.0,
        context_weight=0.2,
        evidence_weight=0.5,
        agreement_weight=0.1,
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
        objective,
        optimizer,
        scheduler,
    )


def test_train_binding_epoch() -> None:
    (
        loader,
        model,
        objective,
        optimizer,
        scheduler,
    ) = build_components()

    result = (
        train_evidence_binding_one_epoch(
            model=model,
            data_loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            objective=objective,
            device=torch.device("cpu"),
            gradient_clip_norm=1.0,
            use_bfloat16=False,
        )
    )

    assert np.isfinite(result.loss)
    assert result.number_of_instances == 6
    assert result.number_with_evidence == 5


def test_binding_evaluation_outputs() -> None:
    (
        loader,
        model,
        objective,
        _,
        _,
    ) = build_components()

    result = (
        evaluate_evidence_binding_model(
            model=model,
            data_loader=loader,
            objective=objective,
            device=torch.device("cpu"),
            use_bfloat16=False,
        )
    )

    assert (
        result.combined_predictions.shape
        == (6,)
    )

    assert (
        result.context_predictions.shape
        == (6,)
    )

    assert (
        result.evidence_predictions.shape
        == (6,)
    )

    assert (
        result.combined_probabilities.shape
        == (6, 3)
    )

    assert result.labels.shape == (6,)
    assert (
        result.evidence_available.shape
        == (6,)
    )

    assert result.number_with_evidence == 5

    assert result.evidence_metrics is not None

    assert 0.0 <= (
        result.mean_gate_value
    ) <= 1.0

    assert (
        result.mean_available_gate_value
        is not None
    )


def test_binding_probabilities_sum_to_one() -> None:
    (
        loader,
        model,
        objective,
        _,
        _,
    ) = build_components()

    result = (
        evaluate_evidence_binding_model(
            model=model,
            data_loader=loader,
            objective=objective,
            device=torch.device("cpu"),
            use_bfloat16=False,
        )
    )

    for probabilities in (
        result.combined_probabilities,
        result.context_probabilities,
        result.evidence_probabilities,
    ):
        assert np.allclose(
            probabilities.sum(axis=1),
            1.0,
        )


def test_binding_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    (
        _,
        model,
        _,
        optimizer,
        scheduler,
    ) = build_components()

    path = tmp_path / "best.pt"

    save_evidence_binding_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=4,
        metric_name="macro_f1",
        metric_value=0.81,
        model_name_or_path="tiny-model",
        model_parameters={
            "hidden_dimension": 8,
            "gate_dimension": 4,
        },
        loss_parameters={
            "combined_weight": 1.0,
            "evidence_weight": 0.5,
        },
    )

    restored = (
        EvidenceBindingTransformerClassifier(
            backbone=TinyEncoder(),
            hidden_dimension=8,
            number_of_classes=3,
            dropout=0.0,
            gate_dimension=4,
        )
    )

    checkpoint = (
        load_evidence_binding_checkpoint(
            path=path,
            model=restored,
            device=torch.device("cpu"),
        )
    )

    assert checkpoint["epoch"] == 4
    assert (
        checkpoint["checkpoint_type"]
        == "evidence_binding_transformer"
    )

    assert (
        checkpoint["loss_parameters"][
            "evidence_weight"
        ]
        == 0.5
    )

    for original, recovered in zip(
        model.parameters(),
        restored.parameters(),
        strict=True,
    ):
        assert torch.allclose(
            original,
            recovered,
        )


def test_wrong_binding_checkpoint_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong.pt"

    torch.save(
        {
            "checkpoint_type": (
                "evidence_transformer"
            ),
            "model_state_dict": {},
        },
        path,
    )

    model = (
        EvidenceBindingTransformerClassifier(
            backbone=TinyEncoder(),
            hidden_dimension=8,
        )
    )

    with pytest.raises(
        ValueError,
        match="evidence-binding",
    ):
        load_evidence_binding_checkpoint(
            path=path,
            model=model,
            device=torch.device("cpu"),
        )


def test_binding_train_empty_loader_rejected() -> None:
    (
        _,
        model,
        objective,
        optimizer,
        scheduler,
    ) = build_components()

    with pytest.raises(
        ValueError,
        match="no instances",
    ):
        train_evidence_binding_one_epoch(
            model=model,
            data_loader=[],
            optimizer=optimizer,
            scheduler=scheduler,
            objective=objective,
            device=torch.device("cpu"),
            gradient_clip_norm=1.0,
            use_bfloat16=False,
        )


def test_binding_eval_empty_loader_rejected() -> None:
    (
        _,
        model,
        objective,
        _,
        _,
    ) = build_components()

    with pytest.raises(
        ValueError,
        match="no instances",
    ):
        evaluate_evidence_binding_model(
            model=model,
            data_loader=[],
            objective=objective,
            device=torch.device("cpu"),
            use_bfloat16=False,
        )
