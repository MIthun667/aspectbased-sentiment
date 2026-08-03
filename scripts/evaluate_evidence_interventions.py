from __future__ import annotations

import argparse
import json
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
    compute_intervention_metrics,
)
from src.aspect_sentiment.evidence import (
    EvidenceIntervention,
    generate_evidence_intervention,
)
from src.aspect_sentiment.models.transformer import (
    EvidenceAwareTransformerClassifier,
    evaluate_evidence_transformer_model,
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

    if checkpoint.get(
        "checkpoint_type"
    ) != "evidence_transformer":
        raise ValueError(
            "Checkpoint is not an evidence "
            "transformer checkpoint"
        )

    model_parameters = (
        validate_model_parameters(
            dict(
                checkpoint[
                    "model_parameters"
                ]
            )
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

    print("=" * 88)
    print("EVIDENCE INTERVENTION EVALUATION")
    print("=" * 88)
    print(
        f"Experiment:        "
        f"{experiment_directory}"
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

        evaluation = (
            evaluate_evidence_transformer_model(
                model=model,
                data_loader=loader,
                loss_function=loss_function,
                device=device,
                use_bfloat16=False,
            )
        )

        intervention_results[
            intervention.value
        ] = generated

        evaluation_results[
            intervention.value
        ] = evaluation

        print(
            f"{intervention.value:26s} "
            f"Macro-F1="
            f"{evaluation.metrics['macro_f1']:.4f} "
            f"Acc="
            f"{evaluation.metrics['accuracy']:.4f} "
            f"NLL="
            f"{evaluation.metrics['negative_log_likelihood']:.4f} "
            f"Changed="
            f"{generated.number_changed}"
        )

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

        if intervention is (
            EvidenceIntervention.ORIGINAL
        ):
            paired_metrics = (
                compute_intervention_metrics(
                    labels=(
                        original_result.labels
                    ),
                    original_predictions=(
                        original_result
                        .predictions
                    ),
                    original_probabilities=(
                        original_result
                        .probabilities
                    ),
                    intervened_predictions=(
                        original_result
                        .predictions
                    ),
                    intervened_probabilities=(
                        original_result
                        .probabilities
                    ),
                )
            )
        else:
            paired_metrics = (
                compute_intervention_metrics(
                    labels=(
                        original_result.labels
                    ),
                    original_predictions=(
                        original_result
                        .predictions
                    ),
                    original_probabilities=(
                        original_result
                        .probabilities
                    ),
                    intervened_predictions=(
                        evaluation.predictions
                    ),
                    intervened_probabilities=(
                        evaluation.probabilities
                    ),
                )
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
                evaluation.metrics
            ),
            "paired_metrics": (
                paired_metrics
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
                original_result
            ),
            intervened_result=(
                evaluation
            ),
            intervention_name=name,
        )

        write_jsonl(
            output_root
            / "predictions"
            / f"{name}.jsonl",
            records,
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
                evaluation.metrics
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
