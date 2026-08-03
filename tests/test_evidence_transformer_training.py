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
    EvidenceAwareTransformerClassifier,
    EvidenceTransformerBatchCollator,
    EvidenceTransformerDataset,
    build_linear_warmup_scheduler,
    build_transformer_optimizer,
    evaluate_evidence_transformer_model,
    load_evidence_transformer_checkpoint,
    move_evidence_transformer_batch_to_device,
    save_evidence_transformer_checkpoint,
    train_evidence_transformer_one_epoch,
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
        sentence_ids = [
            index + 1
            for index in range(len(words))
        ]

        pair_ids = [
            index + 20
            for index in range(
                len(text_pair)
            )
        ]

        input_ids = (
            [30]
            + sentence_ids
            + [31]
            + pair_ids
            + [31]
        )[:max_length]

        full_word_ids = (
            [None]
            + list(range(len(words)))
            + [None]
            + list(
                range(len(text_pair))
            )
            + [None]
        )[:max_length]

        full_sequence_ids = (
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
                    0 if sequence_id != 1
                    else 1
                    for sequence_id in (
                        full_sequence_ids
                    )
                ],
            },
            word_ids=full_word_ids,
            sequence_ids=(
                full_sequence_ids
            ),
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
            padding_length = (
                target_length
                - len(feature["input_ids"])
            )

            for key in output:
                output[key].append(
                    feature[key]
                    + [0] * padding_length
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
        hidden = self.embedding(
            input_ids
        )

        hidden = self.layer_norm(
            hidden
        )

        return SimpleNamespace(
            last_hidden_state=hidden
        )


def make_joined_instances():
    canonical_records = []
    evidence_records = []

    examples = [
        ("positive", "excellent", 2),
        ("negative", "awful", 0),
        ("neutral", "average", 1),
        ("positive-2", "great", 2),
        ("negative-2", "poor", 0),
        ("neutral-2", "ordinary", 1),
    ]

    for instance_id, sentiment, label_id in examples:
        tokens = [
            "battery",
            "is",
            sentiment,
        ]

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
                "selected_evidence_indices": [2],
                "selected_evidence_tokens": [
                    sentiment
                ],
                "evidence_is_empty": False,
                "candidate_count": 1,
                "selection_config": {
                    "maximum_tokens": 3,
                },
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

    dataset = EvidenceTransformerDataset(
        make_joined_instances(),
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
        EvidenceAwareTransformerClassifier(
            backbone=TinyEncoder(),
            hidden_dimension=8,
            number_of_classes=3,
            mode="cls_aspect_evidence",
            dropout=0.0,
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
        optimizer,
        scheduler,
    )


def test_move_batch_to_device() -> None:
    loader, _, _, _ = (
        build_components()
    )

    batch = next(iter(loader))

    moved = (
        move_evidence_transformer_batch_to_device(
            batch,
            device=torch.device("cpu"),
        )
    )

    assert moved.input_ids.device.type == "cpu"

    assert (
        moved.aspect_subword_mask.device.type
        == "cpu"
    )

    assert (
        moved.evidence_subword_mask.device.type
        == "cpu"
    )

    assert moved.instance_ids == (
        batch.instance_ids
    )


def test_train_epoch_returns_finite_loss() -> None:
    (
        loader,
        model,
        optimizer,
        scheduler,
    ) = build_components()

    loss = (
        train_evidence_transformer_one_epoch(
            model=model,
            data_loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            gradient_clip_norm=1.0,
            use_bfloat16=False,
        )
    )

    assert np.isfinite(loss)
    assert loss > 0.0


def test_evaluate_returns_probabilities() -> None:
    loader, model, _, _ = (
        build_components()
    )

    result = (
        evaluate_evidence_transformer_model(
            model=model,
            data_loader=loader,
            loss_function=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            use_bfloat16=False,
        )
    )

    assert result.predictions.shape == (6,)

    assert result.probabilities.shape == (
        6,
        3,
    )

    assert result.labels.shape == (6,)

    assert np.allclose(
        result.probabilities.sum(axis=1),
        1.0,
    )


def test_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    (
        _,
        model,
        optimizer,
        scheduler,
    ) = build_components()

    checkpoint_path = (
        tmp_path / "best.pt"
    )

    save_evidence_transformer_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        metric_name="macro_f1",
        metric_value=0.75,
        model_name_or_path="tiny-model",
        model_parameters={
            "mode": "cls_aspect_evidence",
            "number_of_classes": 3,
        },
    )

    restored = (
        EvidenceAwareTransformerClassifier(
            backbone=TinyEncoder(),
            hidden_dimension=8,
            number_of_classes=3,
            mode="cls_aspect_evidence",
            dropout=0.0,
        )
    )

    checkpoint = (
        load_evidence_transformer_checkpoint(
            path=checkpoint_path,
            model=restored,
            device=torch.device("cpu"),
        )
    )

    assert checkpoint["epoch"] == 3

    assert (
        checkpoint["checkpoint_type"]
        == "evidence_transformer"
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


def test_wrong_checkpoint_type_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong.pt"

    torch.save(
        {
            "checkpoint_type": "other",
            "model_state_dict": {},
        },
        path,
    )

    model = (
        EvidenceAwareTransformerClassifier(
            backbone=TinyEncoder(),
            hidden_dimension=8,
        )
    )

    with pytest.raises(
        ValueError,
        match="not an evidence-transformer",
    ):
        load_evidence_transformer_checkpoint(
            path=path,
            model=model,
            device=torch.device("cpu"),
        )


def test_train_rejects_empty_loader() -> None:
    _, model, optimizer, scheduler = (
        build_components()
    )

    with pytest.raises(
        ValueError,
        match="no instances",
    ):
        train_evidence_transformer_one_epoch(
            model=model,
            data_loader=[],
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            gradient_clip_norm=1.0,
            use_bfloat16=False,
        )
