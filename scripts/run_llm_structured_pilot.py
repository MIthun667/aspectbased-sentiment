from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPOSITORY_ROOT),
    )


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.aspect_sentiment.evaluation.metrics import (
    compute_classification_metrics,
)
from src.aspect_sentiment.models.llm import (
    ID_TO_SENTIMENT,
    LLMABSADataset,
    SENTIMENT_TO_ID,
    build_chat_messages,
    diagnose_evidence,
    parse_recoverable_structured_output,
)


SCHEMA_VERSION = "1.0"


def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    ).isoformat()


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    value = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise TypeError(
            "Config root must be a mapping"
        )

    return value


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def git_value(
    *arguments: str,
) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None

    return result.stdout.strip() or None


def git_dirty() -> bool | None:
    value = git_value(
        "status",
        "--porcelain",
    )

    if value is None:
        return None

    return bool(value)


def write_json(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    records: Sequence[
        Mapping[str, Any]
    ],
) -> None:
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


def deterministic_stratified_subset(
    dataset: LLMABSADataset,
    *,
    instances_per_label: int,
    seed: int,
) -> list[Any]:
    if instances_per_label <= 0:
        raise ValueError(
            "instances_per_label must be positive"
        )

    grouped: dict[str, list[Any]] = {
        "negative": [],
        "neutral": [],
        "positive": [],
    }

    for instance in dataset:
        grouped[
            instance.gold_sentiment
        ].append(instance)

    selected = []

    for label_id, label in enumerate(
        (
            "negative",
            "neutral",
            "positive",
        )
    ):
        candidates = sorted(
            grouped[label],
            key=lambda item: item.instance_id,
        )

        if len(candidates) < (
            instances_per_label
        ):
            raise ValueError(
                f"Not enough {label} instances: "
                f"{len(candidates)}"
            )

        label_rng = random.Random(
            seed + label_id
        )

        indices = sorted(
            label_rng.sample(
                range(len(candidates)),
                instances_per_label,
            )
        )

        selected.extend(
            candidates[index]
            for index in indices
        )

    selected.sort(
        key=lambda item: (
            item.gold_label_id,
            item.instance_id,
        )
    )

    return selected


def select_evaluation_instances(
    dataset: LLMABSADataset,
    *,
    instances_per_label: int | None,
    seed: int,
) -> tuple[list[Any], str]:
    if instances_per_label is None:
        return (
            list(dataset.instances),
            "complete_split",
        )

    return (
        deterministic_stratified_subset(
            dataset,
            instances_per_label=(
                instances_per_label
            ),
            seed=seed,
        ),
        "deterministic_stratified_subset",
    )


def dtype_from_name(
    value: str,
) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    if value not in mapping:
        raise ValueError(
            f"Unsupported dtype: {value!r}"
        )

    return mapping[value]


def confidence_summary(
    records: Sequence[
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    valid = [
        record
        for record in records
        if record[
            "recoverable_valid"
        ]
    ]

    correct = [
        float(record["confidence"])
        for record in valid
        if record["correct"]
    ]

    incorrect = [
        float(record["confidence"])
        for record in valid
        if not record["correct"]
    ]

    overconfident_errors = sum(
        confidence >= 0.8
        for confidence in incorrect
    )

    return {
        "valid_confidence_count": len(
            valid
        ),
        "mean_confidence_correct": (
            float(np.mean(correct))
            if correct
            else None
        ),
        "mean_confidence_incorrect": (
            float(np.mean(incorrect))
            if incorrect
            else None
        ),
        "overconfident_error_count": (
            overconfident_errors
        ),
        "overconfident_error_rate": (
            overconfident_errors
            / len(incorrect)
            if incorrect
            else 0.0
        ),
    }


def run_pilot(
    config_path: Path,
    *,
    seed: int,
    overwrite: bool,
) -> Path:
    config_path = config_path.resolve()
    config = load_yaml(config_path)

    model_parameters = config[
        "model"
    ]["parameters"]

    data_config = config["data"]
    output_config = config["output"]

    model_name = str(
        model_parameters[
            "model_name_or_path"
        ]
    )

    prompt_mode = str(
        model_parameters[
            "prompt_mode"
        ]
    )

    strict_output_keys = bool(
        model_parameters.get(
            "strict_output_keys",
            True,
        )
    )

    maximum_input_length = int(
        model_parameters.get(
            "maximum_input_length",
            2048,
        )
    )

    maximum_new_tokens = int(
        model_parameters.get(
            "maximum_new_tokens",
            128,
        )
    )

    local_files_only = bool(
        model_parameters.get(
            "local_files_only",
            True,
        )
    )

    device_map = model_parameters.get(
        "device_map",
        "auto",
    )

    torch_dtype = dtype_from_name(
        str(
            model_parameters.get(
                "dtype",
                "bfloat16",
            )
        )
    )

    processed_root = Path(
        data_config["processed_root"]
    )

    domain = str(
        data_config["domain"]
    )

    split = str(
        data_config["split"]
    )

    instances_per_label_value = (
        data_config.get(
            "pilot_instances_per_label"
        )
    )

    instances_per_label = (
        None
        if instances_per_label_value is None
        else int(
            instances_per_label_value
        )
    )

    output_directory = (
        Path(output_config["root"])
        / str(
            output_config[
                "experiment_name"
            ]
        )
        / f"seed_{seed}"
    )

    if output_directory.exists():
        if not overwrite:
            raise FileExistsError(
                "Output directory exists: "
                f"{output_directory}"
            )

        import shutil

        shutil.rmtree(
            output_directory
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = LLMABSADataset.from_split(
        processed_root,
        domain=domain,
        split=split,
    )

    (
        selected_instances,
        selection_mode,
    ) = select_evaluation_instances(
        dataset,
        instances_per_label=(
            instances_per_label
        ),
        seed=seed,
    )

    subset_manifest = {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "seed": seed,
        "domain": domain,
        "split": split,
        "selection_mode": (
            selection_mode
        ),
        "instances_per_label": (
            instances_per_label
        ),
        "record_count": len(
            selected_instances
        ),
        "label_counts": dict(
            Counter(
                instance.gold_sentiment
                for instance in (
                    selected_instances
                )
            )
        ),
        "instance_ids": [
            instance.instance_id
            for instance in (
                selected_instances
            )
        ],
    }

    write_json(
        output_directory
        / "subset_manifest.json",
        subset_manifest,
    )

    started_at = utc_now_iso()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=(
            local_files_only
        ),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=(
            local_files_only
        ),
        dtype=torch_dtype,
        device_map=device_map,
    )

    model.eval()

    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    records = []

    for index, instance in enumerate(
        selected_instances,
        start=1,
    ):
        messages = build_chat_messages(
            instance,
            prompt_mode=prompt_mode,
        )

        rendered_prompt = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        encoded = tokenizer(
            rendered_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_token_count = int(
            encoded["input_ids"].shape[1]
        )

        if input_token_count > (
            maximum_input_length
        ):
            raise RuntimeError(
                "Prompt exceeds maximum input "
                f"length for {instance.instance_id}: "
                f"{input_token_count} > "
                f"{maximum_input_length}"
            )

        encoded = {
            key: value.to(model.device)
            for key, value in encoded.items()
        }

        started = time.perf_counter()

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=(
                    maximum_new_tokens
                ),
                do_sample=False,
                pad_token_id=(
                    tokenizer.eos_token_id
                ),
                eos_token_id=(
                    tokenizer.eos_token_id
                ),
                use_cache=True,
            )

        runtime_seconds = (
            time.perf_counter()
            - started
        )

        continuation = generated[
            0,
            input_token_count:,
        ]

        generated_token_count = int(
            continuation.numel()
        )

        raw_output = tokenizer.decode(
            continuation,
            skip_special_tokens=True,
        ).strip()

        parse_result = (
            parse_recoverable_structured_output(
                raw_output,
                prompt_mode=prompt_mode,
                number_of_tokens=len(
                    instance.tokens
                ),
                strict_keys=(
                    strict_output_keys
                ),
            )
        )

        prediction = (
            parse_result.recovered_output
        )

        record: dict[str, Any] = {
            "schema_version": (
                SCHEMA_VERSION
            ),
            "condition_index": index,
            "instance_id": (
                instance.instance_id
            ),
            "sentence_id": (
                instance.sentence_id
            ),
            "domain": instance.domain,
            "split": instance.split,
            "target": (
                instance.aspect_text
            ),
            "target_start": (
                instance.aspect_start
            ),
            "target_end": (
                instance.aspect_end
            ),
            "gold_sentiment": (
                instance.gold_sentiment
            ),
            "gold_label_id": (
                instance.gold_label_id
            ),
            "is_multi_aspect": (
                instance.is_multi_aspect
            ),
            "number_of_aspects": (
                instance.number_of_aspects
            ),
            "raw_output": raw_output,
            "strict_valid": (
                parse_result
                .strict_result
                .valid
            ),
            "strict_error_code": (
                parse_result
                .strict_result
                .error_code
            ),
            "strict_error_message": (
                parse_result
                .strict_result
                .error_message
            ),
            "recoverable_valid": (
                parse_result
                .recoverable_valid
            ),
            "recovery_actions": list(
                parse_result
                .recovery_actions
            ),
            "input_token_count": (
                input_token_count
            ),
            "generated_token_count": (
                generated_token_count
            ),
            "runtime_seconds": (
                runtime_seconds
            ),
        }

        if prediction is None:
            record.update(
                {
                    "predicted_sentiment": (
                        None
                    ),
                    "predicted_label_id": (
                        None
                    ),
                    "confidence": None,
                    "correct": False,
                    "evidence_indices": [],
                    "evidence_tokens": [],
                    "evidence_target_only": (
                        None
                    ),
                    "evidence_contains_target": (
                        None
                    ),
                    "evidence_contains_non_target": (
                        None
                    ),
                    "evidence_count": None,
                }
            )
        else:
            diagnostics = diagnose_evidence(
                instance,
                prediction,
            )

            record.update(
                {
                    "predicted_sentiment": (
                        prediction.sentiment
                    ),
                    "predicted_label_id": (
                        prediction.sentiment_id
                    ),
                    "confidence": float(
                        prediction.confidence
                    ),
                    "correct": (
                        prediction.sentiment_id
                        == instance.gold_label_id
                    ),
                    "evidence_indices": list(
                        prediction
                        .evidence_indices
                    ),
                    "evidence_tokens": list(
                        diagnostics
                        .evidence_tokens
                    ),
                    "evidence_target_only": (
                        diagnostics
                        .evidence_target_only
                    ),
                    "evidence_contains_target": (
                        diagnostics
                        .evidence_contains_target
                    ),
                    "evidence_contains_non_target": (
                        diagnostics
                        .evidence_contains_non_target
                    ),
                    "evidence_count": (
                        diagnostics
                        .evidence_count
                    ),
                }
            )

        records.append(record)

        print(
            f"[{index:02d}/"
            f"{len(selected_instances):02d}] "
            f"{instance.gold_sentiment:8s} "
            f"strict={record['strict_valid']} "
            f"recoverable="
            f"{record['recoverable_valid']} "
            f"pred="
            f"{record['predicted_sentiment']} "
            f"target_only="
            f"{record['evidence_target_only']}"
        )

    predictions_path = (
        output_directory
        / "predictions.jsonl"
    )

    write_jsonl(
        predictions_path,
        records,
    )

    valid_records = [
        record
        for record in records
        if record["recoverable_valid"]
    ]

    strict_valid_count = sum(
        bool(record["strict_valid"])
        for record in records
    )

    recoverable_valid_count = len(
        valid_records
    )

    target_only_count = sum(
        bool(
            record[
                "evidence_target_only"
            ]
        )
        for record in valid_records
    )

    contextual_evidence_count = sum(
        bool(
            record[
                "evidence_contains_non_target"
            ]
        )
        for record in valid_records
    )

    classification_metrics = None

    if valid_records:
        classification_metrics = (
            compute_classification_metrics(
                [
                    int(
                        record[
                            "gold_label_id"
                        ]
                    )
                    for record in (
                        valid_records
                    )
                ],
                [
                    int(
                        record[
                            "predicted_label_id"
                        ]
                    )
                    for record in (
                        valid_records
                    )
                ],
            )
        )

    metrics = {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "model_name_or_path": (
            model_name
        ),
        "prompt_mode": prompt_mode,
        "seed": seed,
        "record_count": len(records),
        "strict_valid_count": (
            strict_valid_count
        ),
        "strict_valid_rate": (
            strict_valid_count
            / len(records)
        ),
        "recoverable_valid_count": (
            recoverable_valid_count
        ),
        "recoverable_valid_rate": (
            recoverable_valid_count
            / len(records)
        ),
        "parse_failure_count": (
            len(records)
            - recoverable_valid_count
        ),
        "target_only_evidence_count": (
            target_only_count
        ),
        "target_only_evidence_rate": (
            target_only_count
            / recoverable_valid_count
            if recoverable_valid_count
            else None
        ),
        "contextual_evidence_count": (
            contextual_evidence_count
        ),
        "contextual_evidence_rate": (
            contextual_evidence_count
            / recoverable_valid_count
            if recoverable_valid_count
            else None
        ),
        "strict_error_counts": dict(
            Counter(
                str(
                    record[
                        "strict_error_code"
                    ]
                )
                for record in records
                if not record[
                    "strict_valid"
                ]
            )
        ),
        "recovery_action_counts": dict(
            Counter(
                action
                for record in records
                for action in record[
                    "recovery_actions"
                ]
            )
        ),
        "classification": (
            classification_metrics
        ),
        "confidence": (
            confidence_summary(records)
        ),
        "runtime": {
            "total_generation_seconds": (
                sum(
                    float(
                        record[
                            "runtime_seconds"
                        ]
                    )
                    for record in records
                )
            ),
            "mean_generation_seconds": (
                float(
                    np.mean(
                        [
                            record[
                                "runtime_seconds"
                            ]
                            for record in records
                        ]
                    )
                )
            ),
            "mean_input_tokens": float(
                np.mean(
                    [
                        record[
                            "input_token_count"
                        ]
                        for record in records
                    ]
                )
            ),
            "mean_generated_tokens": (
                float(
                    np.mean(
                        [
                            record[
                                "generated_token_count"
                            ]
                            for record in records
                        ]
                    )
                )
            ),
            "peak_gpu_memory_gb": (
                (
                    torch.cuda
                    .max_memory_allocated()
                    / (1024 ** 3)
                )
                if torch.cuda.is_available()
                else None
            ),
        },
    }

    write_json(
        output_directory
        / "metrics.json",
        metrics,
    )

    completed_at = utc_now_iso()

    metadata = {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "status": "completed",
        "started_at_utc": started_at,
        "completed_at_utc": (
            completed_at
        ),
        "config_path": str(
            config_path
        ),
        "config_sha256": (
            sha256_file(
                config_path
            )
        ),
        "model_name_or_path": (
            model_name
        ),
        "prompt_mode": prompt_mode,
        "seed": seed,
        "python_version": (
            sys.version
        ),
        "platform": (
            platform.platform()
        ),
        "torch_version": (
            torch.__version__
        ),
        "transformers_version": (
            __import__(
                "transformers"
            ).__version__
        ),
        "cuda_available": (
            torch.cuda.is_available()
        ),
        "cuda_device_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "git_commit": git_value(
            "rev-parse",
            "HEAD",
        ),
        "git_branch": git_value(
            "branch",
            "--show-current",
        ),
        "git_dirty": git_dirty(),
        "command": " ".join(
            sys.argv
        ),
    }

    write_json(
        output_directory
        / "run_metadata.json",
        metadata,
    )

    print("=" * 88)
    print("STRUCTURED LLM PILOT COMPLETE")
    print("=" * 88)
    print(
        "Output:",
        output_directory,
    )
    print(
        "Strict validity:",
        f"{metrics['strict_valid_rate']:.4f}",
    )
    print(
        "Recoverable validity:",
        f"{metrics['recoverable_valid_rate']:.4f}",
    )

    if classification_metrics:
        print(
            "Macro-F1:",
            f"{classification_metrics['macro_f1']:.4f}",
        )
        print(
            "Accuracy:",
            f"{classification_metrics['accuracy']:.4f}",
        )

    print(
        "Target-only evidence:",
        metrics[
            "target_only_evidence_rate"
        ],
    )

    return output_directory


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic structured "
            "LLM ABSA pilot."
        )
    )

    parser.add_argument(
        "config",
        type=Path,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    run_pilot(
        arguments.config,
        seed=arguments.seed,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
