from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
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
from torch import nn
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
    create_evidence_transformer_data_loader,
)
from src.aspect_sentiment.data import (
    EvidenceAwareDataset,
)
from src.aspect_sentiment.evaluation import (
    ID_TO_POLARITY,
    compute_classification_metrics,
    compute_intervention_metrics,
)
from src.aspect_sentiment.evidence import (
    EvidenceIntervention,
    generate_evidence_intervention,
)
from src.aspect_sentiment.models.transformer import (
    EvidenceAwareTransformerClassifier,
    EvidenceBindingLoss,
    EvidenceBindingTransformerClassifier,
    EvidenceCompatibilityBindingTransformerClassifier,
    evaluate_evidence_binding_model,
    evaluate_evidence_transformer_model,
    load_counterfactual_binding_checkpoint,
    load_evidence_binding_checkpoint,
    load_evidence_transformer_checkpoint,
)
from src.aspect_sentiment.utils import (
    seed_everything,
)


INTERVENTIONS = (
    EvidenceIntervention.ORIGINAL,
    EvidenceIntervention.EMPTY,
    EvidenceIntervention.ASPECT_ONLY,
    EvidenceIntervention.FULL_SENTENCE,
    EvidenceIntervention.RANDOM_SAME_SENTENCE,
    EvidenceIntervention.SHUFFLED_CROSS_INSTANCE,
)


SUPPORTED_CHECKPOINT_TYPES = frozenset(
    {
        "evidence_transformer",
        "evidence_binding_transformer",
        (
            "counterfactual_evidence_"
            "binding_transformer"
        ),
    }
)


@dataclass(frozen=True, slots=True)
class InterventionHeadResult:
    metrics: dict[str, Any]
    predictions: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray


def normalized_evaluation_heads(
    result,
    *,
    checkpoint_type: str,
) -> dict[str, InterventionHeadResult]:
    if checkpoint_type == "evidence_transformer":
        return {
            "combined": InterventionHeadResult(
                metrics=dict(result.metrics),
                predictions=result.predictions,
                probabilities=result.probabilities,
                labels=result.labels,
            )
        }

    if checkpoint_type in {
        "evidence_binding_transformer",
        (
            "counterfactual_evidence_"
            "binding_transformer"
        ),
    }:
        heads = {
            "combined": InterventionHeadResult(
                metrics=dict(
                    result.combined_metrics
                ),
                predictions=(
                    result.combined_predictions
                ),
                probabilities=(
                    result.combined_probabilities
                ),
                labels=result.labels,
            ),
            "context": InterventionHeadResult(
                metrics=dict(
                    result.context_metrics
                ),
                predictions=(
                    result.context_predictions
                ),
                probabilities=(
                    result.context_probabilities
                ),
                labels=result.labels,
            ),
        }

        heads["evidence"] = (
            InterventionHeadResult(
                metrics=(
                    dict(
                        result.evidence_metrics
                    )
                    if result.evidence_metrics
                    is not None
                    else {}
                ),
                predictions=(
                    result.evidence_predictions
                ),
                probabilities=(
                    result.evidence_probabilities
                ),
                labels=result.labels,
            )
        )

        return heads

    if checkpoint_type == (
        "counterfactual_evidence_"
        "binding_transformer"
    ):
        return (
            validate_compatibility_binding_model_parameters(
                parameters
            )
        )

    raise ValueError(
        "Unsupported checkpoint type: "
        f"{checkpoint_type!r}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained evidence-aware "
            "transformer under controlled evidence "
            "interventions."
        )
    )

    parser.add_argument(
        "experiment_directory",
        type=Path,
        help=(
            "Path to a completed experiment seed "
            "directory containing resolved_config.json "
            "and checkpoints/best.pt."
        ),
    )

    parser.add_argument(
        "--domain",
        default="laptops",
        choices=(
            "laptops",
            "restaurants",
            "tweets",
        ),
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=(
            "train",
            "validation",
            "test",
            "train_full",
        ),
    )

    parser.add_argument(
        "--intervention-seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return value


def write_json(
    path: Path,
    value: dict[str, Any],
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")

    return path


def write_jsonl(
    path: Path,
    records: list[
        dict[str, Any]
    ],
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")

    return path


def resolve_output_directory(
    experiment_directory: Path,
    *,
    domain: str,
    split: str,
    intervention_seed: int,
    requested: Path | None,
) -> Path:
    if requested is not None:
        return requested

    return (
        experiment_directory
        / "evidence_interventions"
        / f"{domain}__{split}"
        / f"intervention_seed_{intervention_seed}"
    )


def prepare_output_directory(
    path: Path,
    *,
    overwrite: bool,
) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                "Intervention output directory "
                f"already exists: {path}"
            )

        import shutil

        shutil.rmtree(path)

    (path / "metrics").mkdir(
        parents=True,
        exist_ok=True,
    )

    (path / "predictions").mkdir(
        parents=True,
        exist_ok=True,
    )


def classification_metrics_for_head(
    result: InterventionHeadResult,
) -> dict[str, Any]:
    return compute_classification_metrics(
        result.labels,
        result.predictions,
        probabilities=result.probabilities,
    )


def binding_gate_payload(
    result,
) -> dict[str, Any] | None:
    if not hasattr(
        result,
        "mean_gate_value",
    ):
        return None

    return {
        "mean": float(
            result.mean_gate_value
        ),
        "standard_deviation": float(
            result.gate_standard_deviation
        ),
        "mean_available": (
            float(
                result.mean_available_gate_value
            )
            if result.mean_available_gate_value
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
                result.minimum_available_gate_value
            )
            if result.minimum_available_gate_value
            is not None
            else None
        ),
        "maximum_available": (
            float(
                result.maximum_available_gate_value
            )
            if result.maximum_available_gate_value
            is not None
            else None
        ),
    }


def validate_model_parameters(
    parameters: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "number_of_classes",
        "dropout",
        "maximum_length",
        "mode",
        "evidence_root",
        "reject_truncation",
        "dtype",
    }

    missing = sorted(
        required.difference(parameters)
    )

    if missing:
        raise ValueError(
            "Checkpoint model parameters are "
            f"missing fields: {missing}"
        )

    if parameters["mode"] != (
        "cls_aspect_evidence"
    ):
        raise ValueError(
            "Evidence interventions require a "
            "cls_aspect_evidence checkpoint, "
            f"received mode={parameters['mode']!r}"
        )

    return parameters


def validate_binding_model_parameters(
    parameters: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "number_of_classes",
        "dropout",
        "gate_dimension",
        "maximum_length",
        "evidence_root",
        "reject_truncation",
        "dtype",
    }

    missing = sorted(
        required.difference(parameters)
    )

    if missing:
        raise ValueError(
            "Binding checkpoint model parameters "
            f"are missing fields: {missing}"
        )

    if int(
        parameters["gate_dimension"]
    ) <= 0:
        raise ValueError(
            "Binding checkpoint gate_dimension "
            "must be positive"
        )

    return parameters


def validate_compatibility_binding_model_parameters(
    parameters: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "number_of_classes",
        "dropout",
        "compatibility_dimension",
        "maximum_length",
        "evidence_root",
        "reject_truncation",
        "dtype",
    }

    missing = sorted(
        required - set(parameters)
    )

    if missing:
        raise ValueError(
            "Compatibility binding model "
            f"parameters are missing: {missing}"
        )

    if int(
        parameters["number_of_classes"]
    ) <= 1:
        raise ValueError(
            "number_of_classes must exceed one"
        )

    if int(
        parameters["compatibility_dimension"]
    ) <= 0:
        raise ValueError(
            "compatibility_dimension must be "
            "positive"
        )

    if int(
        parameters["maximum_length"]
    ) <= 0:
        raise ValueError(
            "maximum_length must be positive"
        )

    return parameters


def validate_checkpoint_parameters(
    checkpoint_type: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    if checkpoint_type == (
        "evidence_transformer"
    ):
        return validate_model_parameters(
            parameters
        )

    if checkpoint_type == (
        "evidence_binding_transformer"
    ):
        return validate_binding_model_parameters(
            parameters
        )

    if checkpoint_type == (
        "counterfactual_evidence_"
        "binding_transformer"
    ):
        return (
            validate_compatibility_binding_model_parameters(
                parameters
            )
        )

    raise ValueError(
        "Unsupported checkpoint type: "
        f"{checkpoint_type!r}"
    )


def paired_prediction_records(
    *,
    original_instances,
    intervened_instances,
    original_result,
    intervened_result,
    intervention_name: str,
) -> list[dict[str, Any]]:
    number_of_instances = len(
        original_instances
    )

    if len(intervened_instances) != (
        number_of_instances
    ):
        raise ValueError(
            "Original and intervened instance "
            "counts do not match"
        )

    if original_result.labels.shape[0] != (
        number_of_instances
    ):
        raise ValueError(
            "Original result count does not "
            "match instances"
        )

    if intervened_result.labels.shape[0] != (
        number_of_instances
    ):
        raise ValueError(
            "Intervened result count does not "
            "match instances"
        )

    records: list[
        dict[str, Any]
    ] = []

    for index, (
        original_instance,
        intervened_instance,
    ) in enumerate(
        zip(
            original_instances,
            intervened_instances,
            strict=True,
        )
    ):
        if (
            original_instance.instance_id
            != intervened_instance.instance_id
        ):
            raise ValueError(
                "Intervention changed instance "
                "ordering or identity"
            )

        original_prediction = int(
            original_result.predictions[index]
        )

        intervened_prediction = int(
            intervened_result.predictions[index]
        )

        gold_label_id = int(
            original_result.labels[index]
        )

        original_probabilities = tuple(
            float(value)
            for value in (
                original_result
                .probabilities[index]
            )
        )

        intervened_probabilities = tuple(
            float(value)
            for value in (
                intervened_result
                .probabilities[index]
            )
        )

        original_gold_probability = (
            original_probabilities[
                gold_label_id
            ]
        )

        intervened_gold_probability = (
            intervened_probabilities[
                gold_label_id
            ]
        )

        records.append(
            {
                "schema_version": "1.0",
                "intervention": (
                    intervention_name
                ),
                "instance_id": (
                    original_instance
                    .instance_id
                ),
                "sentence_id": (
                    original_instance
                    .sentence_id
                ),
                "domain": (
                    original_instance.domain
                ),
                "split": (
                    original_instance.split
                ),
                "tokens": list(
                    original_instance.tokens
                ),
                "aspect_text": (
                    original_instance
                    .aspect_text
                ),
                "aspect_start": (
                    original_instance
                    .aspect_start
                ),
                "aspect_end": (
                    original_instance
                    .aspect_end
                ),
                "gold_label_id": (
                    gold_label_id
                ),
                "gold_label": (
                    ID_TO_POLARITY[
                        gold_label_id
                    ]
                ),
                "original_evidence_indices": (
                    list(
                        original_instance
                        .selected_evidence_indices
                    )
                ),
                "intervened_evidence_indices": (
                    list(
                        intervened_instance
                        .selected_evidence_indices
                    )
                ),
                "original_evidence_tokens": (
                    list(
                        original_instance
                        .selected_evidence_tokens
                    )
                ),
                "intervened_evidence_tokens": (
                    list(
                        intervened_instance
                        .selected_evidence_tokens
                    )
                ),
                "original_prediction_id": (
                    original_prediction
                ),
                "original_prediction": (
                    ID_TO_POLARITY[
                        original_prediction
                    ]
                ),
                "intervened_prediction_id": (
                    intervened_prediction
                ),
                "intervened_prediction": (
                    ID_TO_POLARITY[
                        intervened_prediction
                    ]
                ),
                "prediction_flipped": (
                    original_prediction
                    != intervened_prediction
                ),
                "original_probabilities": (
                    list(
                        original_probabilities
                    )
                ),
                "intervened_probabilities": (
                    list(
                        intervened_probabilities
                    )
                ),
                "gold_probability_change": (
                    intervened_gold_probability
                    - original_gold_probability
                ),
                "confidence_change": (
                    max(
                        intervened_probabilities
                    )
                    - max(
                        original_probabilities
                    )
                ),
                "selection_config": dict(
                    intervened_instance
                    .selection_config
                ),
            }
        )

    return records


def run_intervention_evaluation(
    *,
    experiment_directory: Path,
    domain: str,
    split: str,
    intervention_seed: int,
    output_directory: Path | None,
    overwrite: bool,
) -> Path:
    experiment_directory = (
        experiment_directory.resolve()
    )

    resolved_config_path = (
        experiment_directory
        / "resolved_config.json"
    )

    checkpoint_path = (
        experiment_directory
        / "checkpoints"
        / "best.pt"
    )

    resolved_config = load_json(
        resolved_config_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_type = str(
        checkpoint.get(
            "checkpoint_type",
            "",
        )
    )

    if checkpoint_type not in (
        SUPPORTED_CHECKPOINT_TYPES
    ):
        raise ValueError(
            "Unsupported evidence checkpoint "
            f"type: {checkpoint_type!r}"
        )

    model_parameters = (
        validate_checkpoint_parameters(
            checkpoint_type,
            dict(
                checkpoint[
                    "model_parameters"
                ]
            ),
        )
    )

    training_config = (
        resolved_config.get(
            "training",
            {}
        )
    )

    evaluation_config = (
        resolved_config.get(
            "evaluation",
            {}
        )
    )

    data_config = resolved_config.get(
        "data",
        {}
    )

    seed = int(
        training_config["seed"]
    )

    seed_everything(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model_name_or_path = str(
        checkpoint[
            "model_name_or_path"
        ]
    )

    evidence_root = Path(
        str(
            model_parameters[
                "evidence_root"
            ]
        )
    )

    processed_root = Path(
        str(
            data_config[
                "processed_root"
            ]
        )
    )

    maximum_length = int(
        model_parameters[
            "maximum_length"
        ]
    )

    reject_truncation = bool(
        model_parameters[
            "reject_truncation"
        ]
    )

    batch_size = int(
        evaluation_config.get(
            "batch_size",
            32,
        )
    )

    local_files_only = bool(
        resolved_config[
            "model"
        ][
            "parameters"
        ].get(
            "local_files_only",
            True,
        )
    )

    num_workers = int(
        resolved_config[
            "model"
        ][
            "parameters"
        ].get(
            "num_workers",
            0,
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        local_files_only=(
            local_files_only
        ),
    )

    dataset = EvidenceAwareDataset.from_split(
        processed_root=processed_root,
        evidence_root=evidence_root,
        domain=domain,
        split=split,
    )

    original_instances = [
        dataset[index]
        for index in range(len(dataset))
    ]

    binding_objective = None
    loss_function = None

    if checkpoint_type == (
        "evidence_transformer"
    ):
        model = (
            EvidenceAwareTransformerClassifier
            .from_pretrained(
                model_name_or_path,
                number_of_classes=int(
                    model_parameters[
                        "number_of_classes"
                    ]
                ),
                mode=str(
                    model_parameters["mode"]
                ),
                dropout=float(
                    model_parameters["dropout"]
                ),
                local_files_only=(
                    local_files_only
                ),
                dtype=torch.float32,
            )
            .to(device)
        )

        load_evidence_transformer_checkpoint(
            path=checkpoint_path,
            model=model,
            device=device,
        )

        loss_function = nn.CrossEntropyLoss()

    elif checkpoint_type == (
        "evidence_binding_transformer"
    ):
        model = (
            EvidenceBindingTransformerClassifier
            .from_pretrained(
                model_name_or_path,
                number_of_classes=int(
                    model_parameters[
                        "number_of_classes"
                    ]
                ),
                dropout=float(
                    model_parameters["dropout"]
                ),
                gate_dimension=int(
                    model_parameters[
                        "gate_dimension"
                    ]
                ),
                local_files_only=(
                    local_files_only
                ),
                dtype=torch.float32,
            )
            .to(device)
        )

        load_evidence_binding_checkpoint(
            path=checkpoint_path,
            model=model,
            device=device,
        )

        loss_parameters = dict(
            checkpoint.get(
                "loss_parameters",
                {},
            )
        )

        binding_objective = EvidenceBindingLoss(
            combined_weight=float(
                loss_parameters.get(
                    "combined_weight",
                    1.0,
                )
            ),
            context_weight=float(
                loss_parameters.get(
                    "context_weight",
                    0.0,
                )
            ),
            evidence_weight=float(
                loss_parameters.get(
                    "evidence_weight",
                    0.0,
                )
            ),
            agreement_weight=float(
                loss_parameters.get(
                    "agreement_weight",
                    0.0,
                )
            ),
            agreement_temperature=float(
                loss_parameters.get(
                    "agreement_temperature",
                    1.0,
                )
            ),
        )

    else:
        model = (
            EvidenceCompatibilityBindingTransformerClassifier
            .from_pretrained(
                model_name_or_path,
                number_of_classes=int(
                    model_parameters[
                        "number_of_classes"
                    ]
                ),
                dropout=float(
                    model_parameters["dropout"]
                ),
                compatibility_dimension=int(
                    model_parameters[
                        "compatibility_dimension"
                    ]
                ),
                local_files_only=(
                    local_files_only
                ),
                dtype=torch.float32,
            )
            .to(device)
        )

        load_counterfactual_binding_checkpoint(
            path=checkpoint_path,
            model=model,
            device=device,
        )

        loss_parameters = dict(
            checkpoint.get(
                "base_loss_parameters",
                {},
            )
        )

        binding_objective = EvidenceBindingLoss(
            combined_weight=float(
                loss_parameters.get(
                    "combined_weight",
                    1.0,
                )
            ),
            context_weight=float(
                loss_parameters.get(
                    "context_weight",
                    0.0,
                )
            ),
            evidence_weight=float(
                loss_parameters.get(
                    "evidence_weight",
                    0.0,
                )
            ),
            agreement_weight=float(
                loss_parameters.get(
                    "agreement_weight",
                    0.0,
                )
            ),
            agreement_temperature=float(
                loss_parameters.get(
                    "agreement_temperature",
                    1.0,
                )
            ),
        )

    output_root = (
        resolve_output_directory(
            experiment_directory,
            domain=domain,
            split=split,
            intervention_seed=(
                intervention_seed
            ),
            requested=output_directory,
        )
    )

    prepare_output_directory(
        output_root,
        overwrite=overwrite,
    )

    intervention_results = {}
    evaluation_results = {}
    evaluation_head_results = {}

    print("=" * 88)
    print("EVIDENCE INTERVENTION EVALUATION")
    print("=" * 88)
    print(
        f"Experiment:        "
        f"{experiment_directory}"
    )
    print(
        f"Checkpoint type:   "
        f"{checkpoint_type}"
    )
    print(
        f"Checkpoint epoch:  "
        f"{checkpoint['epoch']}"
    )
    print(
        f"Checkpoint metric: "
        f"{checkpoint['metric_name']}="
        f"{checkpoint['metric_value']:.6f}"
    )
    print(
        f"Domain / split:    "
        f"{domain} / {split}"
    )
    print(
        f"Instances:         "
        f"{len(original_instances)}"
    )
    print(
        f"Device:            {device}"
    )
    print(
        f"Output:            {output_root}"
    )
    print("-" * 88)

    for intervention in INTERVENTIONS:
        generated = (
            generate_evidence_intervention(
                original_instances,
                intervention=intervention,
                seed=intervention_seed,
            )
        )

        loader = (
            create_evidence_transformer_data_loader(
                list(generated.instances),
                tokenizer=tokenizer,
                maximum_length=(
                    maximum_length
                ),
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
                num_workers=num_workers,
                reject_truncation=(
                    reject_truncation
                ),
            )
        )

        if checkpoint_type == (
            "evidence_transformer"
        ):
            assert loss_function is not None

            evaluation = (
                evaluate_evidence_transformer_model(
                    model=model,
                    data_loader=loader,
                    loss_function=loss_function,
                    device=device,
                    use_bfloat16=False,
                )
            )
        else:
            assert binding_objective is not None

            evaluation = (
                evaluate_evidence_binding_model(
                    model=model,
                    data_loader=loader,
                    objective=binding_objective,
                    device=device,
                    use_bfloat16=False,
                )
            )

        heads = normalized_evaluation_heads(
            evaluation,
            checkpoint_type=checkpoint_type,
        )

        intervention_results[
            intervention.value
        ] = generated

        evaluation_results[
            intervention.value
        ] = evaluation

        evaluation_head_results[
            intervention.value
        ] = heads

        combined = heads["combined"]

        combined_metrics = (
            classification_metrics_for_head(
                combined
            )
        )

        line = (
            f"{intervention.value:26s} "
            f"Macro-F1="
            f"{combined_metrics['macro_f1']:.4f} "
            f"Acc="
            f"{combined_metrics['accuracy']:.4f} "
            f"NLL="
            f"{combined_metrics['negative_log_likelihood']:.4f} "
            f"Changed="
            f"{generated.number_changed}"
        )

        if checkpoint_type == (
            "evidence_binding_transformer"
        ):
            context_metrics = (
                classification_metrics_for_head(
                    heads["context"]
                )
            )

            evidence_metrics = (
                classification_metrics_for_head(
                    heads["evidence"]
                )
            )

            line += (
                f" Context-F1="
                f"{context_metrics['macro_f1']:.4f}"
                f" Evidence-F1="
                f"{evidence_metrics['macro_f1']:.4f}"
            )

        print(line)

    original_result = evaluation_results[
        EvidenceIntervention
        .ORIGINAL
        .value
    ]

    summary: dict[
        str,
        Any,
    ] = {
        "schema_version": "1.0",
        "experiment_directory": str(
            experiment_directory
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
        "checkpoint_type": (
            checkpoint_type
        ),
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "checkpoint_metric_name": str(
            checkpoint["metric_name"]
        ),
        "checkpoint_metric_value": float(
            checkpoint["metric_value"]
        ),
        "domain": domain,
        "split": split,
        "training_seed": seed,
        "intervention_seed": (
            intervention_seed
        ),
        "number_of_instances": len(
            original_instances
        ),
        "interventions": {},
    }

    for intervention in INTERVENTIONS:
        name = intervention.value

        generated = (
            intervention_results[name]
        )

        evaluation = (
            evaluation_results[name]
        )

        original_heads = (
            evaluation_head_results[
                EvidenceIntervention
                .ORIGINAL
                .value
            ]
        )

        current_heads = (
            evaluation_head_results[name]
        )

        head_payloads: dict[
            str,
            dict[str, Any],
        ] = {}

        for head_name, original_head in (
            original_heads.items()
        ):
            current_head = (
                current_heads[head_name]
            )

            head_paired_metrics = (
                compute_intervention_metrics(
                    labels=(
                        original_head.labels
                    ),
                    original_predictions=(
                        original_head.predictions
                    ),
                    original_probabilities=(
                        original_head.probabilities
                    ),
                    intervened_predictions=(
                        current_head.predictions
                    ),
                    intervened_probabilities=(
                        current_head.probabilities
                    ),
                )
            )

            head_payloads[head_name] = {
                "classification_metrics": (
                    classification_metrics_for_head(
                        current_head
                    )
                ),
                "paired_metrics": (
                    head_paired_metrics
                ),
            }

        paired_metrics = (
            head_payloads[
                "combined"
            ][
                "paired_metrics"
            ]
        )

        combined_classification_metrics = (
            head_payloads[
                "combined"
            ][
                "classification_metrics"
            ]
        )

        payload = {
            "schema_version": "1.0",
            "intervention": name,
            "training_seed": seed,
            "intervention_seed": (
                generated.seed
            ),
            "number_changed": (
                generated.number_changed
            ),
            "number_unchanged": (
                generated.number_unchanged
            ),
            "number_empty": (
                generated.number_empty
            ),
            "classification_metrics": (
                combined_classification_metrics
            ),
            "paired_metrics": (
                paired_metrics
            ),
            "heads": head_payloads,
            "gate": binding_gate_payload(
                evaluation
            ),
        }

        write_json(
            output_root
            / "metrics"
            / f"{name}.json",
            payload,
        )

        records = paired_prediction_records(
            original_instances=(
                original_instances
            ),
            intervened_instances=(
                generated.instances
            ),
            original_result=(
                original_heads[
                    "combined"
                ]
            ),
            intervened_result=(
                current_heads[
                    "combined"
                ]
            ),
            intervention_name=name,
        )

        write_jsonl(
            output_root
            / "predictions"
            / f"{name}.jsonl",
            records,
        )

        if checkpoint_type == (
            "evidence_binding_transformer"
        ):
            for head_name in (
                "combined",
                "context",
                "evidence",
            ):
                head_records = (
                    paired_prediction_records(
                        original_instances=(
                            original_instances
                        ),
                        intervened_instances=(
                            generated.instances
                        ),
                        original_result=(
                            original_heads[
                                head_name
                            ]
                        ),
                        intervened_result=(
                            current_heads[
                                head_name
                            ]
                        ),
                        intervention_name=(
                            f"{name}:{head_name}"
                        ),
                    )
                )

                write_jsonl(
                    output_root
                    / "predictions"
                    / (
                        f"{name}__"
                        f"{head_name}.jsonl"
                    ),
                    head_records,
                )

        summary["interventions"][
            name
        ] = {
            "number_changed": (
                generated.number_changed
            ),
            "number_unchanged": (
                generated.number_unchanged
            ),
            "number_empty": (
                generated.number_empty
            ),
            "classification_metrics": (
                combined_classification_metrics
            ),
            "paired_metrics": {
                key: value
                for key, value in (
                    paired_metrics.items()
                )
                if key not in {
                    "original_metrics",
                    "intervened_metrics",
                }
            },
            "heads": {
                head_name: {
                    "classification_metrics": (
                        head_value[
                            "classification_metrics"
                        ]
                    ),
                    "paired_metrics": {
                        key: value
                        for key, value in (
                            head_value[
                                "paired_metrics"
                            ].items()
                        )
                        if key not in {
                            "original_metrics",
                            "intervened_metrics",
                        }
                    },
                }
                for head_name, head_value
                in head_payloads.items()
            },
            "gate": binding_gate_payload(
                evaluation
            ),
        }

    write_json(
        output_root / "summary.json",
        summary,
    )

    print("-" * 88)
    print(
        "Evidence intervention evaluation "
        "complete."
    )

    return output_root


def main() -> None:
    arguments = parse_arguments()

    run_intervention_evaluation(
        experiment_directory=(
            arguments.experiment_directory
        ),
        domain=arguments.domain,
        split=arguments.split,
        intervention_seed=(
            arguments.intervention_seed
        ),
        output_directory=(
            arguments.output_directory
        ),
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
