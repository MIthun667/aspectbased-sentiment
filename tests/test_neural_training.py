from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.aspect_sentiment.models.neural import (
    AspectSentimentDataset,
    NeuralBatchCollator,
    TargetAwareBiLSTM,
    Vocabulary,
    evaluate_neural_model,
    load_neural_checkpoint,
    save_neural_checkpoint,
    train_one_epoch,
)


def make_records() -> list[dict[str, object]]:
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
            "tokens": [
                "battery",
                "is",
                sentiment,
            ],
            "aspect_start": 0,
            "aspect_end": 1,
            "label_id": label_id,
        }
        for instance_id, sentiment, label_id
        in examples
    ]


def build_components():
    records = make_records()

    vocabulary = Vocabulary.build(
        records
    )

    dataset = AspectSentimentDataset(
        records,
        vocabulary=vocabulary,
        maximum_length=16,
    )

    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=NeuralBatchCollator(
            pad_id=vocabulary.pad_id
        ),
    )

    model = TargetAwareBiLSTM(
        vocabulary_size=len(vocabulary),
        embedding_dimension=8,
        hidden_dimension=8,
        number_of_layers=1,
        dropout=0.0,
        padding_index=vocabulary.pad_id,
    )

    return vocabulary, loader, model


def test_train_one_epoch_returns_finite_loss() -> None:
    _, loader, model = build_components()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
    )

    loss = train_one_epoch(
        model=model,
        data_loader=loader,
        optimizer=optimizer,
        loss_function=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
    )

    assert np.isfinite(loss)
    assert loss > 0.0


def test_evaluate_returns_metrics_and_probabilities() -> None:
    _, loader, model = build_components()

    result = evaluate_neural_model(
        model=model,
        data_loader=loader,
        loss_function=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )

    assert np.isfinite(result.loss)
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
    vocabulary, loader, model = (
        build_components()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
    )

    train_one_epoch(
        model=model,
        data_loader=loader,
        optimizer=optimizer,
        loss_function=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
    )

    checkpoint_path = (
        tmp_path / "best.pt"
    )

    save_neural_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        metric_name="macro_f1",
        metric_value=0.75,
        vocabulary=vocabulary.token_to_id,
        model_parameters={
            "embedding_dimension": 8,
            "hidden_dimension": 8,
        },
    )

    restored_model = TargetAwareBiLSTM(
        vocabulary_size=len(vocabulary),
        embedding_dimension=8,
        hidden_dimension=8,
        number_of_layers=1,
        dropout=0.0,
        padding_index=vocabulary.pad_id,
    )

    checkpoint = load_neural_checkpoint(
        path=checkpoint_path,
        model=restored_model,
        device=torch.device("cpu"),
    )

    assert checkpoint["epoch"] == 3
    assert (
        checkpoint["metric_name"]
        == "macro_f1"
    )
    assert (
        checkpoint["metric_value"]
        == 0.75
    )

    for original, restored in zip(
        model.parameters(),
        restored_model.parameters(),
        strict=True,
    ):
        assert torch.allclose(
            original,
            restored,
        )


def test_training_updates_parameters() -> None:
    _, loader, model = build_components()

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
    )

    train_one_epoch(
        model=model,
        data_loader=loader,
        optimizer=optimizer,
        loss_function=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
    )

    after = list(model.parameters())

    assert any(
        not torch.allclose(
            previous,
            current,
        )
        for previous, current in zip(
            before,
            after,
            strict=True,
        )
    )
