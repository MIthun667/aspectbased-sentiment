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


from .evidence_dataset import (
    EvidenceTransformerBatch,
)
from .evidence_model import (
    EvidenceAwareTransformerClassifier,
)


def move_evidence_transformer_batch_to_device(
    batch: EvidenceTransformerBatch,
    *,
    device: torch.device,
) -> EvidenceTransformerBatch:
    return EvidenceTransformerBatch(
        input_ids=batch.input_ids.to(
            device,
            non_blocking=True,
        ),
        attention_mask=(
            batch.attention_mask.to(
                device,
                non_blocking=True,
            )
        ),
        token_type_ids=(
            batch.token_type_ids.to(
                device,
                non_blocking=True,
            )
            if batch.token_type_ids is not None
            else None
        ),
        aspect_subword_mask=(
            batch.aspect_subword_mask.to(
                device,
                non_blocking=True,
            )
        ),
        evidence_subword_mask=(
            batch.evidence_subword_mask.to(
                device,
                non_blocking=True,
            )
        ),
        sentence_subword_mask=(
            batch.sentence_subword_mask.to(
                device,
                non_blocking=True,
            )
        ),
        pair_aspect_subword_mask=(
            batch
            .pair_aspect_subword_mask
            .to(
                device,
                non_blocking=True,
            )
        ),
        labels=batch.labels.to(
            device,
            non_blocking=True,
        ),
        evidence_is_empty=(
            batch.evidence_is_empty.to(
                device,
                non_blocking=True,
            )
        ),
        instance_ids=batch.instance_ids,
        sentence_ids=batch.sentence_ids,
    )


def train_evidence_transformer_one_epoch(
    *,
    model: EvidenceAwareTransformerClassifier,
    data_loader: Iterable[
        EvidenceTransformerBatch
    ],
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
            move_evidence_transformer_batch_to_device(
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
                input_ids=(
                    device_batch.input_ids
                ),
                attention_mask=(
                    device_batch.attention_mask
                ),
                token_type_ids=(
                    device_batch.token_type_ids
                ),
                aspect_subword_mask=(
                    device_batch
                    .aspect_subword_mask
                ),
                evidence_subword_mask=(
                    device_batch
                    .evidence_subword_mask
                ),
            )

            loss = loss_function(
                output.logits,
                device_batch.labels,
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite evidence-transformer "
                "training loss"
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
def evaluate_evidence_transformer_model(
    *,
    model: EvidenceAwareTransformerClassifier,
    data_loader: Iterable[
        EvidenceTransformerBatch
    ],
    loss_function: nn.Module,
    device: torch.device,
    use_bfloat16: bool,
) -> TransformerEvaluationResult:
    model.eval()

    total_loss = 0.0
    total_instances = 0

    probability_batches: list[
        np.ndarray
    ] = []

    prediction_batches: list[
        np.ndarray
    ] = []

    label_batches: list[
        np.ndarray
    ] = []

    autocast_enabled = (
        use_bfloat16
        and device.type == "cuda"
    )

    for batch in data_loader:
        device_batch = (
            move_evidence_transformer_batch_to_device(
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
                input_ids=(
                    device_batch.input_ids
                ),
                attention_mask=(
                    device_batch.attention_mask
                ),
                token_type_ids=(
                    device_batch.token_type_ids
                ),
                aspect_subword_mask=(
                    device_batch
                    .aspect_subword_mask
                ),
                evidence_subword_mask=(
                    device_batch
                    .evidence_subword_mask
                ),
            )

            loss = loss_function(
                output.logits,
                device_batch.labels,
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite evidence-transformer "
                "evaluation loss"
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


def save_evidence_transformer_checkpoint(
    *,
    path: str | Path,
    model: EvidenceAwareTransformerClassifier,
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
            "checkpoint_type": (
                "evidence_transformer"
            ),
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


def load_evidence_transformer_checkpoint(
    *,
    path: str | Path,
    model: EvidenceAwareTransformerClassifier,
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

    checkpoint_type = checkpoint.get(
        "checkpoint_type"
    )

    if checkpoint_type != (
        "evidence_transformer"
    ):
        raise ValueError(
            "Checkpoint is not an "
            "evidence-transformer checkpoint"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return checkpoint


from .binding_losses import (
    EvidenceBindingLoss,
)
from .binding_model import (
    EvidenceBindingTransformerClassifier,
)


@dataclass(frozen=True, slots=True)
class EvidenceBindingTrainingResult:
    loss: float
    combined_loss: float
    context_loss: float
    evidence_loss: float
    agreement_loss: float
    number_of_instances: int
    number_with_evidence: int


@dataclass(frozen=True, slots=True)
class EvidenceBindingEvaluationResult:
    loss: float
    combined_loss: float
    context_loss: float
    evidence_loss: float
    agreement_loss: float

    combined_metrics: dict[str, object]
    context_metrics: dict[str, object]
    evidence_metrics: dict[str, object] | None

    combined_predictions: np.ndarray
    context_predictions: np.ndarray
    evidence_predictions: np.ndarray

    combined_probabilities: np.ndarray
    context_probabilities: np.ndarray
    evidence_probabilities: np.ndarray

    labels: np.ndarray
    evidence_available: np.ndarray

    mean_gate_value: float
    gate_standard_deviation: float
    mean_available_gate_value: float | None
    available_gate_standard_deviation: float | None
    instance_gate_mean_standard_deviation: float | None
    minimum_available_gate_value: float | None
    maximum_available_gate_value: float | None

    number_of_instances: int
    number_with_evidence: int


def _forward_evidence_binding_model(
    *,
    model: EvidenceBindingTransformerClassifier,
    batch: EvidenceTransformerBatch,
):
    return model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        token_type_ids=batch.token_type_ids,
        aspect_subword_mask=(
            batch.aspect_subword_mask
        ),
        evidence_subword_mask=(
            batch.evidence_subword_mask
        ),
    )


def train_evidence_binding_one_epoch(
    *,
    model: EvidenceBindingTransformerClassifier,
    data_loader: Iterable[
        EvidenceTransformerBatch
    ],
    optimizer: torch.optim.Optimizer,
    scheduler,
    objective: EvidenceBindingLoss,
    device: torch.device,
    gradient_clip_norm: float,
    use_bfloat16: bool,
) -> EvidenceBindingTrainingResult:
    if gradient_clip_norm <= 0.0:
        raise ValueError(
            "gradient_clip_norm must be positive"
        )

    model.train()

    total_loss = 0.0
    total_combined_loss = 0.0
    total_context_loss = 0.0
    total_evidence_loss = 0.0
    total_agreement_loss = 0.0

    total_instances = 0
    total_with_evidence = 0

    autocast_enabled = (
        use_bfloat16
        and device.type == "cuda"
    )

    for batch in data_loader:
        device_batch = (
            move_evidence_transformer_batch_to_device(
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
            output = (
                _forward_evidence_binding_model(
                    model=model,
                    batch=device_batch,
                )
            )

            loss_output = objective(
                output,
                device_batch.labels,
                evidence_is_empty=(
                    device_batch
                    .evidence_is_empty
                ),
            )

        if not torch.isfinite(
            loss_output.loss
        ):
            raise FloatingPointError(
                "Non-finite evidence-binding "
                "training loss"
            )

        loss_output.loss.backward()

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
            float(
                loss_output
                .loss
                .detach()
                .cpu()
            )
            * batch_size
        )

        total_combined_loss += (
            float(
                loss_output
                .combined_loss
                .detach()
                .cpu()
            )
            * batch_size
        )

        total_context_loss += (
            float(
                loss_output
                .context_loss
                .detach()
                .cpu()
            )
            * batch_size
        )

        total_evidence_loss += (
            float(
                loss_output
                .evidence_loss
                .detach()
                .cpu()
            )
            * batch_size
        )

        total_agreement_loss += (
            float(
                loss_output
                .agreement_loss
                .detach()
                .cpu()
            )
            * batch_size
        )

        total_instances += batch_size
        total_with_evidence += (
            loss_output
            .number_with_evidence
        )

    if total_instances == 0:
        raise ValueError(
            "Training loader produced no instances"
        )

    return EvidenceBindingTrainingResult(
        loss=total_loss / total_instances,
        combined_loss=(
            total_combined_loss
            / total_instances
        ),
        context_loss=(
            total_context_loss
            / total_instances
        ),
        evidence_loss=(
            total_evidence_loss
            / total_instances
        ),
        agreement_loss=(
            total_agreement_loss
            / total_instances
        ),
        number_of_instances=(
            total_instances
        ),
        number_with_evidence=(
            total_with_evidence
        ),
    )


@torch.no_grad()
def evaluate_evidence_binding_model(
    *,
    model: EvidenceBindingTransformerClassifier,
    data_loader: Iterable[
        EvidenceTransformerBatch
    ],
    objective: EvidenceBindingLoss,
    device: torch.device,
    use_bfloat16: bool,
) -> EvidenceBindingEvaluationResult:
    model.eval()

    total_loss = 0.0
    total_combined_loss = 0.0
    total_context_loss = 0.0
    total_evidence_loss = 0.0
    total_agreement_loss = 0.0

    total_instances = 0
    total_with_evidence = 0

    combined_probability_batches: list[
        np.ndarray
    ] = []
    context_probability_batches: list[
        np.ndarray
    ] = []
    evidence_probability_batches: list[
        np.ndarray
    ] = []

    label_batches: list[np.ndarray] = []
    availability_batches: list[
        np.ndarray
    ] = []
    gate_batches: list[np.ndarray] = []

    autocast_enabled = (
        use_bfloat16
        and device.type == "cuda"
    )

    for batch in data_loader:
        device_batch = (
            move_evidence_transformer_batch_to_device(
                batch,
                device=device,
            )
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            output = (
                _forward_evidence_binding_model(
                    model=model,
                    batch=device_batch,
                )
            )

            loss_output = objective(
                output,
                device_batch.labels,
                evidence_is_empty=(
                    device_batch
                    .evidence_is_empty
                ),
            )

        if not torch.isfinite(
            loss_output.loss
        ):
            raise FloatingPointError(
                "Non-finite evidence-binding "
                "evaluation loss"
            )

        combined_probabilities = torch.softmax(
            output.logits.float(),
            dim=-1,
        )

        context_probabilities = torch.softmax(
            output.context_logits.float(),
            dim=-1,
        )

        evidence_probabilities = torch.softmax(
            output.evidence_logits.float(),
            dim=-1,
        )

        batch_size = int(
            device_batch.labels.shape[0]
        )

        total_loss += (
            float(
                loss_output.loss.detach().cpu()
            )
            * batch_size
        )

        total_combined_loss += (
            float(
                loss_output
                .combined_loss
                .detach()
                .cpu()
            )
            * batch_size
        )

        total_context_loss += (
            float(
                loss_output
                .context_loss
                .detach()
                .cpu()
            )
            * batch_size
        )

        total_evidence_loss += (
            float(
                loss_output
                .evidence_loss
                .detach()
                .cpu()
            )
            * batch_size
        )

        total_agreement_loss += (
            float(
                loss_output
                .agreement_loss
                .detach()
                .cpu()
            )
            * batch_size
        )

        total_instances += batch_size
        total_with_evidence += (
            loss_output
            .number_with_evidence
        )

        combined_probability_batches.append(
            combined_probabilities
            .cpu()
            .numpy()
        )

        context_probability_batches.append(
            context_probabilities
            .cpu()
            .numpy()
        )

        evidence_probability_batches.append(
            evidence_probabilities
            .cpu()
            .numpy()
        )

        label_batches.append(
            device_batch
            .labels
            .cpu()
            .numpy()
        )

        availability_batches.append(
            (
                ~device_batch
                .evidence_is_empty
            )
            .cpu()
            .numpy()
        )

        gate_batches.append(
            output
            .gate_values
            .float()
            .cpu()
            .numpy()
        )

    if total_instances == 0:
        raise ValueError(
            "Evaluation loader produced no instances"
        )

    combined_probabilities = np.concatenate(
        combined_probability_batches,
        axis=0,
    )

    context_probabilities = np.concatenate(
        context_probability_batches,
        axis=0,
    )

    evidence_probabilities = np.concatenate(
        evidence_probability_batches,
        axis=0,
    )

    labels = np.concatenate(
        label_batches,
        axis=0,
    )

    evidence_available = np.concatenate(
        availability_batches,
        axis=0,
    ).astype(bool)

    gate_values = np.concatenate(
        gate_batches,
        axis=0,
    )

    combined_predictions = (
        combined_probabilities.argmax(
            axis=1
        )
    )

    context_predictions = (
        context_probabilities.argmax(
            axis=1
        )
    )

    evidence_predictions = (
        evidence_probabilities.argmax(
            axis=1
        )
    )

    combined_metrics = (
        compute_classification_metrics(
            labels,
            combined_predictions,
            probabilities=(
                combined_probabilities
            ),
        )
    )

    context_metrics = (
        compute_classification_metrics(
            labels,
            context_predictions,
            probabilities=(
                context_probabilities
            ),
        )
    )

    evidence_metrics = None

    if evidence_available.any():
        evidence_metrics = (
            compute_classification_metrics(
                labels[evidence_available],
                evidence_predictions[
                    evidence_available
                ],
                probabilities=(
                    evidence_probabilities[
                        evidence_available
                    ]
                ),
            )
        )

    mean_available_gate_value = None
    available_gate_standard_deviation = None
    instance_gate_mean_standard_deviation = None
    minimum_available_gate_value = None
    maximum_available_gate_value = None

    if evidence_available.any():
        available_gate_values = gate_values[
            evidence_available
        ]

        available_instance_gate_means = (
            available_gate_values.mean(
                axis=1
            )
        )

        mean_available_gate_value = float(
            available_gate_values.mean()
        )

        available_gate_standard_deviation = float(
            available_gate_values.std()
        )

        instance_gate_mean_standard_deviation = float(
            available_instance_gate_means.std()
        )

        minimum_available_gate_value = float(
            available_gate_values.min()
        )

        maximum_available_gate_value = float(
            available_gate_values.max()
        )

    return EvidenceBindingEvaluationResult(
        loss=total_loss / total_instances,
        combined_loss=(
            total_combined_loss
            / total_instances
        ),
        context_loss=(
            total_context_loss
            / total_instances
        ),
        evidence_loss=(
            total_evidence_loss
            / total_instances
        ),
        agreement_loss=(
            total_agreement_loss
            / total_instances
        ),
        combined_metrics=combined_metrics,
        context_metrics=context_metrics,
        evidence_metrics=evidence_metrics,
        combined_predictions=(
            combined_predictions
        ),
        context_predictions=(
            context_predictions
        ),
        evidence_predictions=(
            evidence_predictions
        ),
        combined_probabilities=(
            combined_probabilities
        ),
        context_probabilities=(
            context_probabilities
        ),
        evidence_probabilities=(
            evidence_probabilities
        ),
        labels=labels,
        evidence_available=(
            evidence_available
        ),
        mean_gate_value=float(
            gate_values.mean()
        ),
        gate_standard_deviation=float(
            gate_values.std()
        ),
        mean_available_gate_value=(
            mean_available_gate_value
        ),
        available_gate_standard_deviation=(
            available_gate_standard_deviation
        ),
        instance_gate_mean_standard_deviation=(
            instance_gate_mean_standard_deviation
        ),
        minimum_available_gate_value=(
            minimum_available_gate_value
        ),
        maximum_available_gate_value=(
            maximum_available_gate_value
        ),
        number_of_instances=(
            total_instances
        ),
        number_with_evidence=(
            total_with_evidence
        ),
    )


def save_evidence_binding_checkpoint(
    *,
    path: str | Path,
    model: EvidenceBindingTransformerClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metric_name: str,
    metric_value: float,
    model_name_or_path: str,
    model_parameters: dict[str, object],
    loss_parameters: dict[str, object],
) -> Path:
    checkpoint_path = Path(path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "checkpoint_type": (
                "evidence_binding_transformer"
            ),
            "epoch": epoch,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "model_name_or_path": (
                model_name_or_path
            ),
            "model_parameters": (
                model_parameters
            ),
            "loss_parameters": (
                loss_parameters
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


def load_evidence_binding_checkpoint(
    *,
    path: str | Path,
    model: EvidenceBindingTransformerClassifier,
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

    if checkpoint.get(
        "checkpoint_type"
    ) != "evidence_binding_transformer":
        raise ValueError(
            "Checkpoint is not an "
            "evidence-binding transformer "
            "checkpoint"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return checkpoint
