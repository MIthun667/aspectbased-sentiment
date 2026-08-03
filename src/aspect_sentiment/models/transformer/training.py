from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup

from src.aspect_sentiment.evaluation import (
    compute_classification_metrics,
)

from .dataset import TransformerBatch
from .model import TransformerAspectClassifier


@dataclass(frozen=True, slots=True)
class TransformerEvaluationResult:
    loss: float
    metrics: dict[str, object]
    predictions: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True, slots=True)
class TransformerTrainingState:
    best_epoch: int
    best_metric_value: float
    epochs_completed: int
    stopped_early: bool
    checkpoint_path: Path


def move_transformer_batch_to_device(
    batch: TransformerBatch,
    *,
    device: torch.device,
) -> TransformerBatch:
    return TransformerBatch(
        input_ids=batch.input_ids.to(
            device,
            non_blocking=True,
        ),
        attention_mask=batch.attention_mask.to(
            device,
            non_blocking=True,
        ),
        token_type_ids=(
            batch.token_type_ids.to(
                device,
                non_blocking=True,
            )
            if batch.token_type_ids is not None
            else None
        ),
        labels=batch.labels.to(
            device,
            non_blocking=True,
        ),
        instance_ids=batch.instance_ids,
        sentence_ids=batch.sentence_ids,
    )


def build_transformer_optimizer(
    model: TransformerAspectClassifier,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    if learning_rate <= 0.0:
        raise ValueError(
            "learning_rate must be positive"
        )

    if weight_decay < 0.0:
        raise ValueError(
            "weight_decay must be non-negative"
        )

    no_decay_names = (
        "bias",
        "LayerNorm.weight",
        "layer_norm.weight",
    )

    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        if any(
            excluded_name in name
            for excluded_name in no_decay_names
        ):
            no_decay_parameters.append(
                parameter
            )
        else:
            decay_parameters.append(
                parameter
            )

    if not decay_parameters:
        raise ValueError(
            "No weight-decay parameters found"
        )

    if not no_decay_parameters:
        raise ValueError(
            "No no-decay parameters found"
        )

    return torch.optim.AdamW(
        [
            {
                "params": decay_parameters,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
        foreach=False,
        fused=False,
    )


def build_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    training_steps: int,
    warmup_ratio: float,
):
    if training_steps <= 0:
        raise ValueError(
            "training_steps must be positive"
        )

    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(
            "warmup_ratio must be in [0, 1)"
        )

    warmup_steps = int(
        training_steps * warmup_ratio
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=training_steps,
    )

    return scheduler, warmup_steps


def train_transformer_one_epoch(
    *,
    model: TransformerAspectClassifier,
    data_loader: Iterable[TransformerBatch],
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_function: nn.Module,
    device: torch.device,
    gradient_clip_norm: float,
    use_bfloat16: bool,
) -> float:
    if gradient_clip_norm <= 0.0:
        raise ValueError(
            "gradient_clip_norm must be positive"
        )

    model.train()

    total_loss = 0.0
    total_instances = 0

    autocast_enabled = (
        use_bfloat16
        and device.type == "cuda"
    )

    for batch in data_loader:
        device_batch = (
            move_transformer_batch_to_device(
                batch,
                device=device,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            output = model(
                input_ids=device_batch.input_ids,
                attention_mask=(
                    device_batch.attention_mask
                ),
                token_type_ids=(
                    device_batch.token_type_ids
                ),
            )

            loss = loss_function(
                output.logits,
                device_batch.labels,
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite transformer training loss"
            )

        loss.backward()

        clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
            error_if_nonfinite=True,
            foreach=False,
        )

        optimizer.step()
        scheduler.step()

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
            "Training loader produced no instances"
        )

    return total_loss / total_instances


@torch.no_grad()
def evaluate_transformer_model(
    *,
    model: TransformerAspectClassifier,
    data_loader: Iterable[TransformerBatch],
    loss_function: nn.Module,
    device: torch.device,
    use_bfloat16: bool,
) -> TransformerEvaluationResult:
    model.eval()

    total_loss = 0.0
    total_instances = 0

    probability_batches: list[np.ndarray] = []
    prediction_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []

    autocast_enabled = (
        use_bfloat16
        and device.type == "cuda"
    )

    for batch in data_loader:
        device_batch = (
            move_transformer_batch_to_device(
                batch,
                device=device,
            )
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            output = model(
                input_ids=device_batch.input_ids,
                attention_mask=(
                    device_batch.attention_mask
                ),
                token_type_ids=(
                    device_batch.token_type_ids
                ),
            )

            loss = loss_function(
                output.logits,
                device_batch.labels,
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite transformer evaluation loss"
            )

        probabilities = torch.softmax(
            output.logits.float(),
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
            "Evaluation loader produced no instances"
        )

    probabilities = np.concatenate(
        probability_batches,
        axis=0,
    )
    predictions = np.concatenate(
        prediction_batches,
        axis=0,
    )
    labels = np.concatenate(
        label_batches,
        axis=0,
    )

    metrics = compute_classification_metrics(
        labels,
        predictions,
        probabilities=probabilities,
    )

    return TransformerEvaluationResult(
        loss=total_loss / total_instances,
        metrics=metrics,
        predictions=predictions,
        probabilities=probabilities,
        labels=labels,
    )


def save_transformer_checkpoint(
    *,
    path: str | Path,
    model: TransformerAspectClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metric_name: str,
    metric_value: float,
    model_name_or_path: str,
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
            "model_name_or_path": (
                model_name_or_path
            ),
            "model_parameters": (
                model_parameters
            ),
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
            ),
        },
        checkpoint_path,
    )

    return checkpoint_path


def load_transformer_checkpoint(
    *,
    path: str | Path,
    model: TransformerAspectClassifier,
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
