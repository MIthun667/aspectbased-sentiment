from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

os.environ.setdefault(
    "HF_HUB_DISABLE_PROGRESS_BARS",
    "1",
)

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.utils import (
    logging as transformers_logging,
)

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPOSITORY_ROOT),
    )

from scripts.run_evidence_transformer import (
    class_weights_from_records,
    create_evidence_transformer_data_loader,
    metric_improved,
    prediction_records,
)
from src.aspect_sentiment.artifacts import (
    ExperimentArtifactWriter,
    collect_run_metadata,
    utc_now_iso,
)
from src.aspect_sentiment.config import (
    ExperimentConfig,
    load_experiment_config,
)
from src.aspect_sentiment.data import (
    EvidenceAwareDataset,
    load_canonical_split,
)
from src.aspect_sentiment.evaluation import (
    exclude_text_overlaps,
)
from src.aspect_sentiment.models.transformer import (
    CounterfactualBindingLoss,
    CounterfactualEvidenceBatchCollator,
    CounterfactualEvidenceTransformerDataset,
    EvidenceBindingLoss,
    EvidenceCompatibilityBindingTransformerClassifier,
    build_linear_warmup_scheduler,
    build_transformer_optimizer,
    evaluate_evidence_binding_model,
    load_counterfactual_binding_checkpoint,
    save_counterfactual_binding_checkpoint,
    train_counterfactual_binding_one_epoch,
)
from src.aspect_sentiment.utils import (
    print_evaluation_summary,
    print_experiment_summary,
    print_training_complete,
    seed_everything,
)


def sha256_file(
    path: str | Path,
) -> str:
    source_path = Path(path)

    digest = hashlib.sha256()

    with source_path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_instance_id_file(
    path: str | Path,
) -> tuple[str, ...]:
    source_path = Path(path)

    if not source_path.is_file():
        raise FileNotFoundError(
            "Instance-ID file not found: "
            f"{source_path}"
        )

    payload = json.loads(
        source_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise TypeError(
            "Instance-ID file must contain "
            "a JSON object"
        )

    schema_version = payload.get(
        "schema_version"
    )

    if schema_version != 1:
        raise ValueError(
            "Unsupported instance-ID "
            f"schema_version: {schema_version}"
        )

    declared_count = payload.get("count")

    if (
        isinstance(declared_count, bool)
        or not isinstance(
            declared_count,
            int,
        )
    ):
        raise TypeError(
            "Instance-ID count must be "
            "an integer"
        )

    values = payload.get("instance_ids")

    if not isinstance(values, list):
        raise TypeError(
            "instance_ids must be a list"
        )

    instance_ids: list[str] = []

    for value in values:
        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                "Every instance ID must be "
                "a non-empty string"
            )

        instance_ids.append(value)

    if not instance_ids:
        raise ValueError(
            "Instance-ID file must not "
            "be empty"
        )

    if declared_count != len(instance_ids):
        raise ValueError(
            "Declared instance-ID count "
            "does not match instance_ids"
        )

    if len(set(instance_ids)) != len(
        instance_ids
    ):
        raise ValueError(
            "Instance-ID file contains "
            "duplicate IDs"
        )

    return tuple(instance_ids)


def record_instance_id(record) -> str:
    if isinstance(record, dict):
        value = record.get("instance_id")
    else:
        value = getattr(
            record,
            "instance_id",
            None,
        )

    if not isinstance(value, str):
        raise TypeError(
            "Record instance_id must be "
            "a string"
        )

    return value


def record_sentence_id(record) -> str:
    if isinstance(record, dict):
        value = record.get("sentence_id")
    else:
        value = getattr(
            record,
            "sentence_id",
            None,
        )

    if not isinstance(value, str):
        raise TypeError(
            "Record sentence_id must be "
            "a string"
        )

    return value


def filter_records_by_instance_ids(
    records,
    instance_ids,
    *,
    subset_name: str,
):
    requested = tuple(instance_ids)

    if not requested:
        raise ValueError(
            f"{subset_name} IDs must not "
            "be empty"
        )

    if len(set(requested)) != len(
        requested
    ):
        raise ValueError(
            f"{subset_name} IDs contain "
            "duplicates"
        )

    indexed = {}

    for record in records:
        instance_id = record_instance_id(
            record
        )

        if instance_id in indexed:
            raise ValueError(
                "Records contain duplicate "
                f"instance_id: {instance_id}"
            )

        indexed[instance_id] = record

    unknown = sorted(
        set(requested) - set(indexed)
    )

    if unknown:
        raise ValueError(
            f"{subset_name} contains unknown "
            f"instance IDs: {unknown[:10]}"
        )

    requested_set = set(requested)

    filtered = [
        record
        for record in records
        if record_instance_id(record)
        in requested_set
    ]

    if len(filtered) != len(requested):
        raise RuntimeError(
            f"{subset_name} filtering did "
            "not preserve the requested count"
        )

    return filtered


def validate_subset_separation(
    training_records,
    validation_records,
) -> None:
    training_ids = {
        record_instance_id(record)
        for record in training_records
    }

    validation_ids = {
        record_instance_id(record)
        for record in validation_records
    }

    overlapping_ids = sorted(
        training_ids & validation_ids
    )

    if overlapping_ids:
        raise ValueError(
            "Training and validation "
            "instance-ID subsets overlap: "
            f"{overlapping_ids[:10]}"
        )

    training_sentences = {
        record_sentence_id(record)
        for record in training_records
    }

    validation_sentences = {
        record_sentence_id(record)
        for record in validation_records
    }

    overlapping_sentences = sorted(
        training_sentences
        & validation_sentences
    )

    if overlapping_sentences:
        raise ValueError(
            "Training and validation "
            "sentence groups overlap: "
            f"{overlapping_sentences[:10]}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the counterfactual "
            "target-evidence binding DeBERTa classifier."
        )
    )

    parser.add_argument(
        "config",
        type=Path,
        help=(
            "Path to a counterfactual binding "
            "experiment configuration."
        ),
    )

    return parser.parse_args()


def binding_metric_payload(
    result,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "combined": dict(
            result.combined_metrics
        ),
        "context": dict(
            result.context_metrics
        ),
        "evidence": (
            dict(result.evidence_metrics)
            if result.evidence_metrics
            is not None
            else None
        ),
        "losses": {
            "total": float(
                result.loss
            ),
            "combined": float(
                result.combined_loss
            ),
            "context": float(
                result.context_loss
            ),
            "evidence": float(
                result.evidence_loss
            ),
            "agreement": float(
                result.agreement_loss
            ),
        },
        "gate": {
            "mean": float(
                result.mean_gate_value
            ),
            "standard_deviation": float(
                result.gate_standard_deviation
            ),
            "mean_available": (
                float(
                    result
                    .mean_available_gate_value
                )
                if result
                .mean_available_gate_value
                is not None
                else None
            ),
            "available_standard_deviation": (
                float(
                    result
                    .available_gate_standard_deviation
                )
                if result
                .available_gate_standard_deviation
                is not None
                else None
            ),
            "instance_mean_standard_deviation": (
                float(
                    result
                    .instance_gate_mean_standard_deviation
                )
                if result
                .instance_gate_mean_standard_deviation
                is not None
                else None
            ),
            "minimum_available": (
                float(
                    result
                    .minimum_available_gate_value
                )
                if result
                .minimum_available_gate_value
                is not None
                else None
            ),
            "maximum_available": (
                float(
                    result
                    .maximum_available_gate_value
                )
                if result
                .maximum_available_gate_value
                is not None
                else None
            ),
        },
        "number_of_instances": int(
            result.number_of_instances
        ),
        "number_with_evidence": int(
            result.number_with_evidence
        ),
    }

    return payload


def print_binding_evaluation(
    *,
    domain: str,
    split: str,
    result,
) -> None:
    print_evaluation_summary(
        domain=domain,
        split=split,
        metrics=result.combined_metrics,
    )

    context = result.context_metrics

    print()
    print("HEAD AND BINDING DIAGNOSTICS")
    print("-" * 80)
    print(
        "Context Macro-F1:       "
        f"{context['macro_f1']:.4f}"
    )
    print(
        "Context accuracy:       "
        f"{context['accuracy']:.4f}"
    )

    if result.evidence_metrics is None:
        print(
            "Evidence Macro-F1:      "
            "Not available"
        )
    else:
        print(
            "Evidence Macro-F1:      "
            f"{result.evidence_metrics['macro_f1']:.4f}"
        )
        print(
            "Evidence accuracy:      "
            f"{result.evidence_metrics['accuracy']:.4f}"
        )

    print(
        "Examples with evidence: "
        f"{result.number_with_evidence}/"
        f"{result.number_of_instances}"
    )
    print(
        "Mean gate:              "
        f"{result.mean_gate_value:.4f}"
    )

    if (
        result.mean_available_gate_value
        is None
    ):
        print(
            "Mean available gate:    "
            "Not available"
        )
    else:
        print(
            "Mean available gate:    "
            f"{result.mean_available_gate_value:.4f}"
        )

    print(
        "Total loss:             "
        f"{result.loss:.4f}"
    )
    print(
        "Combined loss:          "
        f"{result.combined_loss:.4f}"
    )
    print(
        "Context loss:           "
        f"{result.context_loss:.4f}"
    )
    print(
        "Evidence loss:          "
        f"{result.evidence_loss:.4f}"
    )
    print(
        "Agreement loss:         "
        f"{result.agreement_loss:.4f}"
    )


def write_binding_artifacts(
    *,
    config: ExperimentConfig,
    writer: ExperimentArtifactWriter,
    domain: str,
    split: str,
    records: list[dict[str, Any]],
    result,
) -> None:
    writer.write_metrics(
        domain=domain,
        split=split,
        metrics=binding_metric_payload(
            result
        ),
    )

    if config.evaluation.save_predictions:
        writer.write_predictions(
            domain=domain,
            split=split,
            predictions=prediction_records(
                records,
                result.combined_predictions,
                result.combined_probabilities,
                split_name=split,
            ),
        )

    print_binding_evaluation(
        domain=domain,
        split=split,
        result=result,
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    command: str | None = None,
) -> Path:
    started_at = utc_now_iso()

    seed_everything(
        config.training.seed
    )

    if config.model.name != (
        "counterfactual_binding_deberta"
    ):
        raise ValueError(
            "run_counterfactual_binding_transformer.py "
            "supports only model.name="
            "'counterfactual_binding_deberta'"
        )

    parameters = dict(
        config.model.parameters
    )

    model_name_or_path = str(
        parameters.pop(
            "model_name_or_path",
            "microsoft/deberta-v3-base",
        )
    )

    evidence_root = Path(
        str(
            parameters.pop(
                "evidence_root",
                "data/derived/evidence",
            )
        )
    )

    maximum_length = int(
        parameters.pop(
            "maximum_length",
            128,
        )
    )

    reject_truncation = bool(
        parameters.pop(
            "reject_truncation",
            True,
        )
    )

    dropout = float(
        parameters.pop(
            "dropout",
            0.1,
        )
    )

    compatibility_dimension = int(
        parameters.pop(
            "compatibility_dimension",
            256,
        )
    )

    combined_weight = float(
        parameters.pop(
            "combined_weight",
            1.0,
        )
    )

    context_weight = float(
        parameters.pop(
            "context_weight",
            0.2,
        )
    )

    evidence_weight = float(
        parameters.pop(
            "evidence_weight",
            0.5,
        )
    )

    agreement_weight = float(
        parameters.pop(
            "agreement_weight",
            0.1,
        )
    )

    agreement_temperature = float(
        parameters.pop(
            "agreement_temperature",
            1.0,
        )
    )

    ranking_weight = float(
        parameters.pop(
            "ranking_weight",
            0.2,
        )
    )

    probability_margin_weight = float(
        parameters.pop(
            "probability_margin_weight",
            0.2,
        )
    )

    ranking_margin = float(
        parameters.pop(
            "ranking_margin",
            0.2,
        )
    )

    probability_margin = float(
        parameters.pop(
            "probability_margin",
            0.05,
        )
    )

    counterfactual_seed = int(
        parameters.pop(
            "counterfactual_seed",
            config.training.seed,
        )
    )

    gradient_clip_norm = float(
        parameters.pop(
            "gradient_clip_norm",
            1.0,
        )
    )

    warmup_ratio = float(
        parameters.pop(
            "warmup_ratio",
            0.1,
        )
    )

    class_weighting = bool(
        parameters.pop(
            "class_weighting",
            False,
        )
    )

    use_bfloat16 = bool(
        parameters.pop(
            "use_bfloat16",
            False,
        )
    )

    local_files_only = bool(
        parameters.pop(
            "local_files_only",
            True,
        )
    )

    num_workers = int(
        parameters.pop(
            "num_workers",
            0,
        )
    )

    if parameters:
        raise ValueError(
            "Unsupported binding parameters: "
            f"{sorted(parameters)}"
        )

    if maximum_length <= 0:
        raise ValueError(
            "maximum_length must be positive"
        )

    if compatibility_dimension <= 0:
        raise ValueError(
            "compatibility_dimension must be "
            "positive"
        )

    if counterfactual_seed < 0:
        raise ValueError(
            "counterfactual_seed must be "
            "non-negative"
        )

    if gradient_clip_norm <= 0.0:
        raise ValueError(
            "gradient_clip_norm must be positive"
        )

    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(
            "warmup_ratio must be in [0, 1)"
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers must be non-negative"
        )

    if use_bfloat16:
        raise ValueError(
            "BF16 is disabled for the validated "
            "binding configuration"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print_experiment_summary(
        experiment_name=(
            config.output.experiment_name
        ),
        model_name=config.model.name,
        train_domain=config.data.train_domain,
        evaluation_domains=(
            config.data.evaluation_domains
        ),
        seed=config.training.seed,
        selection_metric=(
            config.training.selection_metric
        ),
        output_directory=(
            config.output_directory()
        ),
    )

    print(
        f"Device:               {device}"
    )

    if device.type == "cuda":
        print(
            "GPU:                  "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        "Pretrained model:     "
        f"{model_name_or_path}"
    )
    print(
        "Maximum length:       "
        f"{maximum_length}"
    )
    print(
        "Evidence root:        "
        f"{evidence_root}"
    )
    print(
        "Compatibility dim.:   "
        f"{compatibility_dimension}"
    )
    print(
        "Counterfactual seed:  "
        f"{counterfactual_seed}"
    )
    print(
        "Reject truncation:    "
        f"{reject_truncation}"
    )
    print(
        "Parameter dtype:      "
        "torch.float32"
    )
    print(
        "Warmup ratio:         "
        f"{warmup_ratio:.4f}"
    )
    print(
        "Base loss weights:    "
        f"combined={combined_weight:.3f}, "
        f"context={context_weight:.3f}, "
        f"evidence={evidence_weight:.3f}, "
        f"agreement={agreement_weight:.3f}"
    )
    print(
        "Counterfactual loss:  "
        f"ranking={ranking_weight:.3f}, "
        f"probability={probability_margin_weight:.3f}, "
        f"rank_margin={ranking_margin:.3f}, "
        f"prob_margin={probability_margin:.3f}"
    )
    print()

    writer = ExperimentArtifactWriter(
        config
    )

    output_directory = writer.prepare()
    writer.write_resolved_config()

    checkpoint_path = (
        output_directory
        / "checkpoints"
        / "best.pt"
    )

    validation_source_split = (
        config.data.validation_source_split
        or config.data.validation_split
    )

    train_records = load_canonical_split(
        config.data.processed_root,
        domain=config.data.train_domain,
        split=config.data.train_split,
    )

    validation_records = (
        load_canonical_split(
            config.data.processed_root,
            domain=(
                config.data.train_domain
            ),
            split=validation_source_split,
        )
    )

    train_joined = list(
        EvidenceAwareDataset.from_split(
            processed_root=(
                config.data.processed_root
            ),
            evidence_root=evidence_root,
            domain=config.data.train_domain,
            split=config.data.train_split,
        )
    )

    validation_joined = list(
        EvidenceAwareDataset.from_split(
            processed_root=(
                config.data.processed_root
            ),
            evidence_root=evidence_root,
            domain=config.data.train_domain,
            split=validation_source_split,
        )
    )

    train_instance_ids = None
    validation_instance_ids = None

    if (
        config.data.train_instance_ids_path
        is not None
    ):
        train_instance_ids = (
            load_instance_id_file(
                config.data
                .train_instance_ids_path
            )
        )

        train_records = (
            filter_records_by_instance_ids(
                train_records,
                train_instance_ids,
                subset_name=(
                    "Training subset"
                ),
            )
        )

        train_joined = (
            filter_records_by_instance_ids(
                train_joined,
                train_instance_ids,
                subset_name=(
                    "Joined training subset"
                ),
            )
        )

    if (
        config.data
        .validation_instance_ids_path
        is not None
    ):
        validation_instance_ids = (
            load_instance_id_file(
                config.data
                .validation_instance_ids_path
            )
        )

        validation_records = (
            filter_records_by_instance_ids(
                validation_records,
                validation_instance_ids,
                subset_name=(
                    "Validation subset"
                ),
            )
        )

        validation_joined = (
            filter_records_by_instance_ids(
                validation_joined,
                validation_instance_ids,
                subset_name=(
                    "Joined validation subset"
                ),
            )
        )

    if (
        train_instance_ids is not None
        or validation_instance_ids
        is not None
    ):
        validate_subset_separation(
            train_records,
            validation_records,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        local_files_only=(
            local_files_only
        ),
    )

    train_dataset = (
        CounterfactualEvidenceTransformerDataset(
            list(train_joined),
            tokenizer=tokenizer,
            maximum_length=maximum_length,
            reject_truncation=(
                reject_truncation
            ),
            seed=counterfactual_seed,
        )
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(
        config.training.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=(
            config.training.batch_size
        ),
        shuffle=True,
        generator=train_generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=(
            CounterfactualEvidenceBatchCollator(
                tokenizer=tokenizer
            )
        ),
    )

    validation_loader = (
        create_evidence_transformer_data_loader(
            list(validation_joined),
            tokenizer=tokenizer,
            maximum_length=maximum_length,
            batch_size=(
                config.evaluation.batch_size
            ),
            shuffle=False,
            seed=config.training.seed,
            num_workers=num_workers,
            reject_truncation=(
                reject_truncation
            ),
        )
    )

    print(
        "Training instances:   "
        f"{len(train_records)}"
    )
    print(
        "Validation instances: "
        f"{len(validation_records)}"
    )
    print(
        "Training batches:     "
        f"{len(train_loader)}"
    )
    print()

    model_parameters: dict[
        str,
        object,
    ] = {
        "number_of_classes": 3,
        "dropout": dropout,
        "compatibility_dimension": (
            compatibility_dimension
        ),
        "counterfactual_seed": (
            counterfactual_seed
        ),
        "maximum_length": maximum_length,
        "evidence_root": str(
            evidence_root
        ),
        "reject_truncation": (
            reject_truncation
        ),
        "validation_source_split": (
            validation_source_split
        ),
        "train_instance_ids_path": (
            config.data
            .train_instance_ids_path
        ),
        "train_instance_ids_sha256": (
            sha256_file(
                config.data
                .train_instance_ids_path
            )
            if config.data
            .train_instance_ids_path
            is not None
            else None
        ),
        "validation_instance_ids_path": (
            config.data
            .validation_instance_ids_path
        ),
        "validation_instance_ids_sha256": (
            sha256_file(
                config.data
                .validation_instance_ids_path
            )
            if config.data
            .validation_instance_ids_path
            is not None
            else None
        ),
        "filtered_training_count": len(
            train_records
        ),
        "filtered_validation_count": len(
            validation_records
        ),
        "dtype": "float32",
    }

    base_loss_parameters: dict[
        str,
        object,
    ] = {
        "combined_weight": (
            combined_weight
        ),
        "context_weight": (
            context_weight
        ),
        "evidence_weight": (
            evidence_weight
        ),
        "agreement_weight": (
            agreement_weight
        ),
        "agreement_temperature": (
            agreement_temperature
        ),
    }

    counterfactual_loss_parameters: dict[
        str,
        object,
    ] = {
        "ranking_weight": ranking_weight,
        "probability_margin_weight": (
            probability_margin_weight
        ),
        "ranking_margin": ranking_margin,
        "probability_margin": (
            probability_margin
        ),
    }

    model = (
        EvidenceCompatibilityBindingTransformerClassifier
        .from_pretrained(
            model_name_or_path,
            number_of_classes=3,
            dropout=dropout,
            compatibility_dimension=(
                compatibility_dimension
            ),
            local_files_only=(
                local_files_only
            ),
            dtype=torch.float32,
        )
        .to(device)
    )

    parameter_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
    }

    if parameter_dtypes != {
        torch.float32
    }:
        raise TypeError(
            "Counterfactual binding model "
            "parameters are not "
            f"entirely FP32: {parameter_dtypes}"
        )

    base_objective = EvidenceBindingLoss(
        combined_weight=(
            combined_weight
        ),
        context_weight=context_weight,
        evidence_weight=evidence_weight,
        agreement_weight=(
            agreement_weight
        ),
        agreement_temperature=(
            agreement_temperature
        ),
    )

    counterfactual_objective = (
        CounterfactualBindingLoss(
            ranking_weight=ranking_weight,
            probability_margin_weight=(
                probability_margin_weight
            ),
            ranking_margin=ranking_margin,
            probability_margin=(
                probability_margin
            ),
        )
    )

    optimizer = build_transformer_optimizer(
        model,
        learning_rate=(
            config.training.learning_rate
        ),
        weight_decay=(
            config.training.weight_decay
        ),
    )

    total_training_steps = (
        len(train_loader)
        * config.training.epochs
    )

    scheduler, warmup_steps = (
        build_linear_warmup_scheduler(
            optimizer,
            training_steps=(
                total_training_steps
            ),
            warmup_ratio=warmup_ratio,
        )
    )

    print(
        "Total optimizer steps: "
        f"{total_training_steps}"
    )
    print(
        "Warmup steps:         "
        f"{warmup_steps}"
    )
    print()

    if class_weighting:
        class_weights_from_records(
            train_records,
            device=device,
        )

        raise ValueError(
            "Class weighting is not yet supported "
            "by EvidenceBindingLoss"
        )

    best_metric_value = (
        -math.inf
        if config.training
        .maximize_selection_metric
        else math.inf
    )

    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    epochs_completed = 0

    try:
        for epoch in range(
            1,
            config.training.epochs + 1,
        ):
            training_result = (
                train_counterfactual_binding_one_epoch(
                    model=model,
                    data_loader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    base_objective=(
                        base_objective
                    ),
                    counterfactual_objective=(
                        counterfactual_objective
                    ),
                    device=device,
                    gradient_clip_norm=(
                        gradient_clip_norm
                    ),
                    use_bfloat16=False,
                )
            )

            validation_result = (
                evaluate_evidence_binding_model(
                    model=model,
                    data_loader=(
                        validation_loader
                    ),
                    objective=base_objective,
                    device=device,
                    use_bfloat16=False,
                )
            )

            metric_value = float(
                validation_result
                .combined_metrics[
                    config.training
                    .selection_metric
                ]
            )

            improved = metric_improved(
                current=metric_value,
                best=best_metric_value,
                maximize=(
                    config.training
                    .maximize_selection_metric
                ),
            )

            if improved:
                best_metric_value = (
                    metric_value
                )
                best_epoch = epoch
                epochs_without_improvement = 0

                save_counterfactual_binding_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metric_name=(
                        config.training
                        .selection_metric
                    ),
                    metric_value=(
                        metric_value
                    ),
                    model_name_or_path=(
                        model_name_or_path
                    ),
                    model_parameters=(
                        model_parameters
                    ),
                    base_loss_parameters=(
                        base_loss_parameters
                    ),
                    counterfactual_loss_parameters=(
                        counterfactual_loss_parameters
                    ),
                )
            else:
                epochs_without_improvement += 1

            epochs_completed = epoch

            evidence_macro_f1 = (
                float(
                    validation_result
                    .evidence_metrics[
                        "macro_f1"
                    ]
                )
                if validation_result
                .evidence_metrics
                is not None
                else float("nan")
            )

            available_gate = (
                validation_result
                .mean_available_gate_value
            )

            available_gate_text = (
                f"{available_gate:.4f}"
                if available_gate
                is not None
                else "N/A"
            )

            print(
                f"Epoch {epoch}/"
                f"{config.training.epochs} "
                f"| Train Total "
                f"{training_result.loss:.4f} "
                f"| Train Base "
                f"{training_result.base_loss:.4f} "
                f"| Train Rank-R "
                f"{training_result.random_ranking_loss:.4f} "
                f"| Train Rank-X "
                f"{training_result.cross_ranking_loss:.4f} "
                f"| Valid R/X "
                f"{training_result.number_valid_random}/"
                f"{training_result.number_valid_cross} "
                f"| Val Combined F1 "
                f"{metric_value:.4f} "
                f"| Val Context F1 "
                f"{validation_result.context_metrics['macro_f1']:.4f} "
                f"| Val Evidence F1 "
                f"{evidence_macro_f1:.4f} "
                f"| Val NLL "
                f"{validation_result.combined_metrics['negative_log_likelihood']:.4f} "
                f"| Gate "
                f"{available_gate_text} "
                f"| LR "
                f"{optimizer.param_groups[0]['lr']:.2e}"
            )

            if (
                config.training
                .early_stopping_patience
                > 0
                and epochs_without_improvement
                >= config.training
                .early_stopping_patience
            ):
                stopped_early = True
                break

        if best_epoch == 0:
            raise RuntimeError(
                "Training completed without a "
                "valid counterfactual binding "
                "checkpoint"
            )

        load_counterfactual_binding_checkpoint(
            path=checkpoint_path,
            model=model,
            device=device,
        )

        validation_result = (
            evaluate_evidence_binding_model(
                model=model,
                data_loader=validation_loader,
                objective=base_objective,
                device=device,
                use_bfloat16=False,
            )
        )

        print()

        write_binding_artifacts(
            config=config,
            writer=writer,
            domain=(
                config.data.train_domain
            ),
            split=(
                config.data.validation_split
            ),
            records=validation_records,
            result=validation_result,
        )

        for domain in (
            config.data.evaluation_domains
        ):
            test_records = load_canonical_split(
                config.data.processed_root,
                domain=domain,
                split=config.data.test_split,
            )

            test_joined = (
                EvidenceAwareDataset.from_split(
                    processed_root=(
                        config.data.processed_root
                    ),
                    evidence_root=(
                        evidence_root
                    ),
                    domain=domain,
                    split=(
                        config.data.test_split
                    ),
                )
            )

            test_loader = (
                create_evidence_transformer_data_loader(
                    list(test_joined),
                    tokenizer=tokenizer,
                    maximum_length=(
                        maximum_length
                    ),
                    batch_size=(
                        config.evaluation
                        .batch_size
                    ),
                    shuffle=False,
                    seed=config.training.seed,
                    num_workers=num_workers,
                    reject_truncation=(
                        reject_truncation
                    ),
                )
            )

            test_result = (
                evaluate_evidence_binding_model(
                    model=model,
                    data_loader=test_loader,
                    objective=base_objective,
                    device=device,
                    use_bfloat16=False,
                )
            )

            write_binding_artifacts(
                config=config,
                writer=writer,
                domain=domain,
                split=config.data.test_split,
                records=test_records,
                result=test_result,
            )

            if (
                config.evaluation
                .duplicate_excluded_sensitivity
            ):
                training_reference = (
                    load_canonical_split(
                        config.data.processed_root,
                        domain=domain,
                        split=(
                            config.data.train_split
                        ),
                    )
                )

                retained, excluded = (
                    exclude_text_overlaps(
                        test_records,
                        reference_records=(
                            training_reference
                        ),
                    )
                )

                print()
                print(
                    "Duplicate sensitivity: "
                    f"excluded={len(excluded)}, "
                    f"retained={len(retained)}"
                )

                if retained:
                    test_by_id = {
                        instance.instance_id: (
                            instance
                        )
                        for instance
                        in test_joined
                    }

                    retained_joined = [
                        test_by_id[
                            str(
                                record[
                                    "instance_id"
                                ]
                            )
                        ]
                        for record in retained
                    ]

                    sensitivity_loader = (
                        create_evidence_transformer_data_loader(
                            retained_joined,
                            tokenizer=tokenizer,
                            maximum_length=(
                                maximum_length
                            ),
                            batch_size=(
                                config.evaluation
                                .batch_size
                            ),
                            shuffle=False,
                            seed=(
                                config.training
                                .seed
                            ),
                            num_workers=(
                                num_workers
                            ),
                            reject_truncation=(
                                reject_truncation
                            ),
                        )
                    )

                    sensitivity_result = (
                        evaluate_evidence_binding_model(
                            model=model,
                            data_loader=(
                                sensitivity_loader
                            ),
                            objective=base_objective,
                            device=device,
                            use_bfloat16=False,
                        )
                    )

                    write_binding_artifacts(
                        config=config,
                        writer=writer,
                        domain=domain,
                        split=(
                            "test_duplicate_excluded"
                        ),
                        records=retained,
                        result=(
                            sensitivity_result
                        ),
                    )

        writer.write_run_metadata(
            collect_run_metadata(
                experiment_name=(
                    config.output
                    .experiment_name
                ),
                model_name=(
                    config.model.name
                ),
                seed=config.training.seed,
                status="COMPLETE",
                started_at_utc=started_at,
                completed_at_utc=(
                    utc_now_iso()
                ),
                command=command,
                repository_root=".",
            )
        )

        print_training_complete(
            status="COMPLETE",
            best_epoch=best_epoch,
            best_metric_name=(
                config.training
                .selection_metric
            ),
            best_metric_value=(
                best_metric_value
            ),
            output_directory=(
                output_directory
            ),
        )

        print(
            "Epochs completed:     "
            f"{epochs_completed}"
        )
        print(
            "Stopped early:        "
            f"{stopped_early}"
        )

        return output_directory

    except Exception:
        writer.write_run_metadata(
            collect_run_metadata(
                experiment_name=(
                    config.output
                    .experiment_name
                ),
                model_name=(
                    config.model.name
                ),
                seed=config.training.seed,
                status="FAILED",
                started_at_utc=started_at,
                completed_at_utc=(
                    utc_now_iso()
                ),
                command=command,
                repository_root=".",
            )
        )

        raise


def main() -> None:
    arguments = parse_arguments()

    config = load_experiment_config(
        arguments.config
    )

    run_experiment(
        config,
        command=" ".join(sys.argv),
    )


if __name__ == "__main__":
    main()
