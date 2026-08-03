from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.aspect_sentiment.models.transformer import (
    TransformerAspectClassifier,
    TransformerAspectDataset,
    TransformerBatchCollator,
    build_linear_warmup_scheduler,
    build_transformer_optimizer,
    evaluate_transformer_model,
    load_transformer_checkpoint,
    save_transformer_checkpoint,
    train_transformer_one_epoch,
)


class TinyTokenizer:
    pad_token_id = 0
    padding_side = "right"
    model_input_names = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ]

    def __call__(
        self,
        text,
        *,
        text_pair,
        truncation,
        max_length,
        padding,
        return_attention_mask,
        return_token_type_ids,
    ):
        tokens = (
            text.lower().split()
            + ["[SEP]"]
            + text_pair.lower().split()
        )

        input_ids = [
            (sum(ord(char) for char in token) % 25) + 1
            for token in tokens[:max_length]
        ]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "token_type_ids": (
                [0] * max(1, len(input_ids) - 1)
                + [1]
            )[:len(input_ids)],
        }

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

            output["input_ids"].append(
                feature["input_ids"]
                + [0] * padding_length
            )
            output["attention_mask"].append(
                feature["attention_mask"]
                + [0] * padding_length
            )
            output["token_type_ids"].append(
                feature["token_type_ids"]
                + [0] * padding_length
            )

        return {
            key: torch.tensor(
                value,
                dtype=torch.long,
            )
            for key, value in output.items()
        }


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.config = SimpleNamespace(
            num_labels=3
        )

        self.embedding = nn.Embedding(
            32,
            8,
            padding_idx=0,
        )

        self.layer_norm = nn.LayerNorm(8)
        self.classifier = nn.Linear(8, 3)

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        token_type_ids=None,
    ):
        embedded = self.embedding(
            input_ids
        )

        mask = (
            attention_mask
            .unsqueeze(-1)
            .to(embedded.dtype)
        )

        pooled = (
            (embedded * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1.0)
        )

        logits = self.classifier(
            self.layer_norm(pooled)
        )

        return SimpleNamespace(
            logits=logits,
            hidden_states=None,
        )


def make_records():
    examples = [
        ("positive-1", "excellent", 2),
        ("positive-2", "great", 2),
        ("negative-1", "poor", 0),
        ("negative-2", "awful", 0),
        ("neutral-1", "average", 1),
        ("neutral-2", "ordinary", 1),
    ]

    return [
        {
            "instance_id": instance_id,
            "sentence_id": (
                f"{instance_id}:sentence"
            ),
            "tokens": [
                "battery",
                "is",
                sentiment,
            ],
            "aspect_text": "battery",
            "label_id": label_id,
        }
        for instance_id, sentiment, label_id
        in examples
    ]


def build_components():
    tokenizer = TinyTokenizer()

    dataset = TransformerAspectDataset(
        make_records(),
        tokenizer=tokenizer,
        maximum_length=16,
    )

    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=TransformerBatchCollator(
            tokenizer=tokenizer
        ),
    )

    model = TransformerAspectClassifier(
        backbone=TinyBackbone()
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

    return loader, model, optimizer, scheduler


def test_optimizer_uses_two_parameter_groups() -> None:
    _, model, _, _ = build_components()

    optimizer = build_transformer_optimizer(
        model,
        learning_rate=1e-3,
        weight_decay=0.01,
    )

    assert len(
        optimizer.param_groups
    ) == 2

    weights = sorted(
        group["weight_decay"]
        for group in optimizer.param_groups
    )

    assert weights == [0.0, 0.01]


def test_scheduler_reports_warmup_steps() -> None:
    _, _, optimizer, _ = build_components()

    _, warmup_steps = (
        build_linear_warmup_scheduler(
            optimizer,
            training_steps=10,
            warmup_ratio=0.2,
        )
    )

    assert warmup_steps == 2


def test_train_epoch_returns_finite_loss() -> None:
    loader, model, optimizer, scheduler = (
        build_components()
    )

    loss = train_transformer_one_epoch(
        model=model,
        data_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
        use_bfloat16=False,
    )

    assert np.isfinite(loss)
    assert loss > 0.0


def test_evaluate_returns_probabilities() -> None:
    loader, model, _, _ = build_components()

    result = evaluate_transformer_model(
        model=model,
        data_loader=loader,
        loss_function=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        use_bfloat16=False,
    )

    assert result.predictions.shape == (6,)
    assert result.probabilities.shape == (6, 3)
    assert result.labels.shape == (6,)

    assert np.allclose(
        result.probabilities.sum(axis=1),
        1.0,
    )

    assert (
        result.metrics["number_of_instances"]
        == 6
    )


def test_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    _, model, optimizer, scheduler = (
        build_components()
    )

    checkpoint_path = (
        tmp_path / "best.pt"
    )

    save_transformer_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        metric_name="macro_f1",
        metric_value=0.7,
        model_name_or_path="tiny-model",
        model_parameters={
            "number_of_classes": 3,
        },
    )

    restored = TransformerAspectClassifier(
        backbone=TinyBackbone()
    )

    checkpoint = load_transformer_checkpoint(
        path=checkpoint_path,
        model=restored,
        device=torch.device("cpu"),
    )

    assert checkpoint["epoch"] == 2
    assert (
        checkpoint["metric_name"]
        == "macro_f1"
    )
    assert checkpoint["metric_value"] == 0.7

    for original, recovered in zip(
        model.parameters(),
        restored.parameters(),
        strict=True,
    ):
        assert torch.allclose(
            original,
            recovered,
        )


@pytest.mark.parametrize(
    ("learning_rate", "weight_decay"),
    [
        (0.0, 0.01),
        (-1e-5, 0.01),
        (1e-5, -0.01),
    ],
)
def test_optimizer_rejects_invalid_values(
    learning_rate,
    weight_decay,
) -> None:
    model = TransformerAspectClassifier(
        backbone=TinyBackbone()
    )

    with pytest.raises(ValueError):
        build_transformer_optimizer(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )


@pytest.mark.parametrize(
    ("training_steps", "warmup_ratio"),
    [
        (0, 0.1),
        (10, -0.1),
        (10, 1.0),
    ],
)
def test_scheduler_rejects_invalid_values(
    training_steps,
    warmup_ratio,
) -> None:
    _, _, optimizer, _ = build_components()

    with pytest.raises(ValueError):
        build_linear_warmup_scheduler(
            optimizer,
            training_steps=training_steps,
            warmup_ratio=warmup_ratio,
        )
