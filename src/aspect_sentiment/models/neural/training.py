from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from src.aspect_sentiment.evaluation import (
    compute_classification_metrics,
)

from .dataset import NeuralBatch
from .bilstm import TargetAwareBiLSTM


@dataclass(frozen=True, slots=True)
class NeuralEvaluationResult:
    loss: float
    metrics: dict[str, object]
    predictions: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True, slots=True)
class NeuralTrainingResult:
    best_epoch: int
    best_metric_value: float
    epochs_completed: int
    stopped_early: bool
    checkpoint_path: Path


def move_batch_to_device(
    batch: NeuralBatch,
    *,
    device: torch.device,
) -> NeuralBatch:
    return NeuralBatch(
        input_ids=batch.input_ids.to(device),
        attention_mask=(
            batch.attention_mask.to(device)
        ),
        labels=batch.labels.to(device),
        instance_ids=batch.instance_ids,
    )


def train_one_epoch(
    *,
    model: TargetAwareBiLSTM,
    data_loader: Iterable[NeuralBatch],
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    if gradient_clip_norm <= 0.0:
        raise ValueError(
            "gradient_clip_norm must be positive"
        )

    model.train()

    total_loss = 0.0
    total_instances = 0

    for batch in data_loader:
        device_batch = move_batch_to_device(
            batch,
            device=device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            device_batch.input_ids,
            device_batch.attention_mask,
        )

        loss = loss_function(
            output.logits,
            device_batch.labels,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite training loss encountered"
            )

        loss.backward()

        clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
        )

        optimizer.step()

        batch_size = int(
            device_batch.labels.shape[0]
        )

        total_loss += (
            float(loss.detach().cpu())
            * batch_size
        )
        total_instances += batch_size

    if total_instances == 0:
        raise ValueError(
            "Training data loader produced no instances"
        )

    return total_loss / total_instances


@torch.no_grad()
def evaluate_neural_model(
    *,
    model: TargetAwareBiLSTM,
    data_loader: Iterable[NeuralBatch],
    loss_function: nn.Module,
    device: torch.device,
) -> NeuralEvaluationResult:
    model.eval()

    total_loss = 0.0
    total_instances = 0

    probability_batches: list[np.ndarray] = []
    prediction_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []

    for batch in data_loader:
        device_batch = move_batch_to_device(
            batch,
            device=device,
        )

        output = model(
            device_batch.input_ids,
            device_batch.attention_mask,
        )

        loss = loss_function(
            output.logits,
            device_batch.labels,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite evaluation loss encountered"
            )

        probabilities = torch.softmax(
            output.logits,
            dim=-1,
        )

        predictions = probabilities.argmax(
            dim=-1
        )

        batch_size = int(
            device_batch.labels.shape[0]
        )

        total_loss += (
            float(loss.detach().cpu())
            * batch_size
        )
        total_instances += batch_size

        probability_batches.append(
            probabilities.cpu().numpy()
        )
        prediction_batches.append(
            predictions.cpu().numpy()
        )
        label_batches.append(
            device_batch.labels.cpu().numpy()
        )

    if total_instances == 0:
        raise ValueError(
            "Evaluation data loader produced no instances"
        )

    all_probabilities = np.concatenate(
        probability_batches,
        axis=0,
    )

    all_predictions = np.concatenate(
        prediction_batches,
        axis=0,
    )

    all_labels = np.concatenate(
        label_batches,
        axis=0,
    )

    metrics = compute_classification_metrics(
        all_labels,
        all_predictions,
        probabilities=all_probabilities,
    )

    return NeuralEvaluationResult(
        loss=total_loss / total_instances,
        metrics=metrics,
        predictions=all_predictions,
        probabilities=all_probabilities,
        labels=all_labels,
    )


def save_neural_checkpoint(
    *,
    path: str | Path,
    model: TargetAwareBiLSTM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric_name: str,
    metric_value: float,
    vocabulary: dict[str, int],
    model_parameters: dict[str, object],
) -> Path:
    checkpoint_path = Path(path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "vocabulary": vocabulary,
            "model_parameters": (
                model_parameters
            ),
        },
        checkpoint_path,
    )

    return checkpoint_path


def load_neural_checkpoint(
    *,
    path: str | Path,
    model: TargetAwareBiLSTM,
    device: torch.device,
) -> dict[str, object]:
    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return checkpoint
