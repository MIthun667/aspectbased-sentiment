from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.aspect_sentiment.artifacts import (
    ExperimentArtifactWriter,
    PredictionRecord,
    collect_run_metadata,
    utc_now_iso,
)
from src.aspect_sentiment.config import (
    ExperimentConfig,
    load_experiment_config,
)
from src.aspect_sentiment.data import (
    EvidenceAwareDataset,
    EvidenceAwareInstance,
    load_canonical_split,
)
from src.aspect_sentiment.evaluation import (
    ID_TO_POLARITY,
    exclude_text_overlaps,
)
from src.aspect_sentiment.models.transformer import (
    SUPPORTED_EVIDENCE_POOLING_MODES,
    EvidenceAwareTransformerClassifier,
    EvidenceTransformerBatchCollator,
    EvidenceTransformerDataset,
    build_linear_warmup_scheduler,
    build_transformer_optimizer,
    evaluate_evidence_transformer_model,
    load_evidence_transformer_checkpoint,
    save_evidence_transformer_checkpoint,
    train_evidence_transformer_one_epoch,
)
from src.aspect_sentiment.utils import (
    EpochReport,
    print_epoch_report,
    print_evaluation_summary,
    print_experiment_summary,
    print_training_complete,
    seed_everything,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune and evaluate a reproducible "
            "evidence-aware transformer classifier."
        )
    )

    parser.add_argument(
        "config",
        type=Path,
        help=(
            "Path to the evidence-transformer "
            "experiment YAML file."
        ),
    )

    return parser.parse_args()


def create_evidence_transformer_data_loader(
    records: list[EvidenceAwareInstance],
    *,
    tokenizer,
    maximum_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    reject_truncation: bool,
) -> DataLoader:
    dataset = EvidenceTransformerDataset(
        records,
        tokenizer=tokenizer,
        maximum_length=maximum_length,
        reject_truncation=reject_truncation,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        collate_fn=EvidenceTransformerBatchCollator(
            tokenizer=tokenizer
        ),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def prediction_records(
    records: list[dict[str, Any]],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    split_name: str,
) -> list[PredictionRecord]:
    if len(records) != len(predictions):
        raise ValueError(
            "Record and prediction counts do not match"
        )

    if len(records) != len(probabilities):
        raise ValueError(
            "Record and probability counts do not match"
        )

    output: list[PredictionRecord] = []

    for record, prediction, probability in zip(
        records,
        predictions,
        probabilities,
        strict=True,
    ):
        predicted_label_id = int(prediction)

        output.append(
            PredictionRecord(
                instance_id=str(record["instance_id"]),
                sentence_id=str(record["sentence_id"]),
                domain=str(record["domain"]),
                split=split_name,
                target=str(record["aspect_text"]),
                gold_label=str(record["polarity"]),
                gold_label_id=int(record["label_id"]),
                predicted_label=(
                    ID_TO_POLARITY[predicted_label_id]
                ),
                predicted_label_id=predicted_label_id,
                probabilities=tuple(
                    float(value)
                    for value in probability
                ),
            )
        )

    return output


def class_weights_from_records(
    records: list[dict[str, Any]],
    *,
    device: torch.device,
) -> torch.Tensor:
    counts = np.bincount(
        [
            int(record["label_id"])
            for record in records
        ],
        minlength=3,
    ).astype(np.float64)

    if np.any(counts <= 0):
        raise ValueError(
            "All classes must appear when class weighting "
            "is enabled"
        )

    weights = len(records) / (3.0 * counts)

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def metric_improved(
    *,
    current: float,
    best: float,
    maximize: bool,
) -> bool:
    return (
        current > best
        if maximize
        else current < best
    )


def write_evaluation_artifacts(
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
        metrics=result.metrics,
    )

    if config.evaluation.save_predictions:
        writer.write_predictions(
            domain=domain,
            split=split,
            predictions=prediction_records(
                records,
                result.predictions,
                result.probabilities,
                split_name=split,
            ),
        )

    print_evaluation_summary(
        domain=domain,
        split=split,
        metrics=result.metrics,
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
        "evidence_deberta_classifier"
    ):
        raise ValueError(
            "run_evidence_transformer.py supports only "
            "model.name="
            "'evidence_deberta_classifier'"
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

    maximum_length = int(
        parameters.pop(
            "maximum_length",
            128,
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

    mode = str(
        parameters.pop(
            "mode",
            "cls_aspect_evidence",
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
            "Unsupported transformer parameters: "
            f"{sorted(parameters)}"
        )

    if maximum_length <= 0:
        raise ValueError(
            "maximum_length must be positive"
        )

    if mode not in (
        SUPPORTED_EVIDENCE_POOLING_MODES
    ):
        raise ValueError(
            f"Unsupported evidence mode: {mode!r}"
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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if use_bfloat16:
        raise ValueError(
            "BF16 is disabled for this baseline because "
            "the validated configuration uses FP32"
        )

    print_experiment_summary(
        experiment_name=config.output.experiment_name,
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

    print(f"Device:               {device}")

    if device.type == "cuda":
        print(
            "GPU:                  "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Pretrained model:     {model_name_or_path}"
    )
    print(
        f"Maximum length:       {maximum_length}"
    )
    print(
        f"Evidence mode:        {mode}"
    )
    print(
        f"Evidence root:        {evidence_root}"
    )
    print(
        "Reject truncation:    "
        f"{reject_truncation}"
    )
    print("Parameter dtype:      torch.float32")
    print(
        f"Warmup ratio:         {warmup_ratio:.4f}"
    )
    print()

    writer = ExperimentArtifactWriter(config)
    output_directory = writer.prepare()
    writer.write_resolved_config()

    checkpoint_path = (
        output_directory
        / "checkpoints"
        / "best.pt"
    )

    train_records = load_canonical_split(
        config.data.processed_root,
        domain=config.data.train_domain,
        split=config.data.train_split,
    )

    validation_records = load_canonical_split(
        config.data.processed_root,
        domain=config.data.train_domain,
        split=config.data.validation_split,
    )

    train_joined = EvidenceAwareDataset.from_split(
        processed_root=config.data.processed_root,
        evidence_root=evidence_root,
        domain=config.data.train_domain,
        split=config.data.train_split,
    )

    validation_joined = (
        EvidenceAwareDataset.from_split(
            processed_root=(
                config.data.processed_root
            ),
            evidence_root=evidence_root,
            domain=config.data.train_domain,
            split=config.data.validation_split,
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        local_files_only=local_files_only,
    )

    train_loader = (
        create_evidence_transformer_data_loader(
            list(train_joined),
            tokenizer=tokenizer,
            maximum_length=maximum_length,
            batch_size=config.training.batch_size,
            shuffle=True,
            seed=config.training.seed,
            num_workers=num_workers,
            reject_truncation=reject_truncation,
        )
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
            reject_truncation=reject_truncation,
        )
    )

    print(
        f"Training instances:   {len(train_records)}"
    )
    print(
        f"Validation instances: {len(validation_records)}"
    )
    print(
        f"Training batches:     {len(train_loader)}"
    )
    print()

    model_parameters: dict[str, object] = {
        "number_of_classes": 3,
        "dropout": dropout,
        "maximum_length": maximum_length,
        "mode": mode,
        "evidence_root": str(evidence_root),
        "reject_truncation": (
            reject_truncation
        ),
        "dtype": "float32",
    }

    model = (
        EvidenceAwareTransformerClassifier
        .from_pretrained(
            model_name_or_path,
            number_of_classes=3,
            mode=mode,
            dropout=dropout,
            local_files_only=local_files_only,
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
            "Transformer parameters were not loaded "
            f"entirely in FP32: {parameter_dtypes}"
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
            training_steps=total_training_steps,
            warmup_ratio=warmup_ratio,
        )
    )

    print(
        f"Total optimizer steps: {total_training_steps}"
    )
    print(
        f"Warmup steps:         {warmup_steps}"
    )
    print()

    class_weights = (
        class_weights_from_records(
            train_records,
            device=device,
        )
        if class_weighting
        else None
    )

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights
    )

    best_metric_value = (
        -math.inf
        if config.training.maximize_selection_metric
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
            training_loss = (
                train_evidence_transformer_one_epoch(
                    model=model,
                    data_loader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_function=loss_function,
                    device=device,
                    gradient_clip_norm=(
                        gradient_clip_norm
                    ),
                    use_bfloat16=False,
                )
            )

            validation_result = (
                evaluate_evidence_transformer_model(
                    model=model,
                    data_loader=validation_loader,
                    loss_function=loss_function,
                    device=device,
                    use_bfloat16=False,
                )
            )

            metric_value = float(
                validation_result.metrics[
                    config.training.selection_metric
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
                best_metric_value = metric_value
                best_epoch = epoch
                epochs_without_improvement = 0

                save_evidence_transformer_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metric_name=(
                        config.training
                        .selection_metric
                    ),
                    metric_value=metric_value,
                    model_name_or_path=(
                        model_name_or_path
                    ),
                    model_parameters=(
                        model_parameters
                    ),
                )
            else:
                epochs_without_improvement += 1

            epochs_completed = epoch

            epoch_validation_metrics = {
                "macro_f1": float(
                    validation_result.metrics[
                        "macro_f1"
                    ]
                ),
                "accuracy": float(
                    validation_result.metrics[
                        "accuracy"
                    ]
                ),
                "balanced_accuracy": float(
                    validation_result.metrics[
                        "balanced_accuracy"
                    ]
                ),
                "negative_log_likelihood": float(
                    validation_result.metrics[
                        "negative_log_likelihood"
                    ]
                ),
                "loss": float(
                    validation_result.loss
                ),
            }

            print_epoch_report(
                EpochReport(
                    epoch=epoch,
                    total_epochs=(
                        config.training.epochs
                    ),
                    train_loss=training_loss,
                    validation_metrics=(
                        epoch_validation_metrics
                    ),
                    learning_rate=float(
                        optimizer.param_groups[0]["lr"]
                    ),
                )
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
                "Training completed without saving "
                "a valid transformer checkpoint"
            )

        load_evidence_transformer_checkpoint(
            path=checkpoint_path,
            model=model,
            device=device,
        )

        validation_result = (
            evaluate_evidence_transformer_model(
                model=model,
                data_loader=validation_loader,
                loss_function=loss_function,
                device=device,
                use_bfloat16=False,
            )
        )

        print()

        write_evaluation_artifacts(
            config=config,
            writer=writer,
            domain=config.data.train_domain,
            split=config.data.validation_split,
            records=validation_records,
            result=validation_result,
        )

        for domain in config.data.evaluation_domains:
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
                    evidence_root=evidence_root,
                    domain=domain,
                    split=config.data.test_split,
                )
            )

            test_loader = (
                create_evidence_transformer_data_loader(
                    list(test_joined),
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

            test_result = (
                evaluate_evidence_transformer_model(
                    model=model,
                    data_loader=test_loader,
                    loss_function=loss_function,
                    device=device,
                    use_bfloat16=False,
                )
            )

            write_evaluation_artifacts(
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
                domain_training_records = (
                    load_canonical_split(
                        config.data.processed_root,
                        domain=domain,
                        split=config.data.train_split,
                    )
                )

                retained, excluded = (
                    exclude_text_overlaps(
                        test_records,
                        reference_records=(
                            domain_training_records
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
                        instance.instance_id: instance
                        for instance in test_joined
                    }

                    retained_joined = [
                        test_by_id[
                            str(record["instance_id"])
                        ]
                        for record in retained
                    ]

                    sensitivity_loader = (
                        create_evidence_transformer_data_loader(
                            retained_joined,
                            tokenizer=tokenizer,
                            maximum_length=maximum_length,
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

                    sensitivity_result = (
                        evaluate_evidence_transformer_model(
                            model=model,
                            data_loader=(
                                sensitivity_loader
                            ),
                            loss_function=loss_function,
                            device=device,
                            use_bfloat16=False,
                        )
                    )

                    write_evaluation_artifacts(
                        config=config,
                        writer=writer,
                        domain=domain,
                        split=(
                            "test_duplicate_excluded"
                        ),
                        records=retained,
                        result=sensitivity_result,
                    )

        writer.write_run_metadata(
            collect_run_metadata(
                experiment_name=(
                    config.output.experiment_name
                ),
                model_name=config.model.name,
                seed=config.training.seed,
                status="COMPLETE",
                started_at_utc=started_at,
                completed_at_utc=utc_now_iso(),
                command=command,
                repository_root=".",
            )
        )

        print_training_complete(
            status="COMPLETE",
            best_epoch=best_epoch,
            best_metric_name=(
                config.training.selection_metric
            ),
            best_metric_value=best_metric_value,
            output_directory=output_directory,
        )

        print(
            f"Epochs completed:     {epochs_completed}"
        )
        print(
            f"Stopped early:        {stopped_early}"
        )

        return output_directory

    except Exception:
        writer.write_run_metadata(
            collect_run_metadata(
                experiment_name=(
                    config.output.experiment_name
                ),
                model_name=config.model.name,
                seed=config.training.seed,
                status="FAILED",
                started_at_utc=started_at,
                completed_at_utc=utc_now_iso(),
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
