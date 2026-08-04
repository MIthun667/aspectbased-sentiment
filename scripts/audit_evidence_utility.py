from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_THRESHOLD = 0.05


def read_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    resolved_path = Path(path)

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"JSONL file does not exist: "
            f"{resolved_path}"
        )

    records: list[dict[str, Any]] = []

    with resolved_path.open(
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON at "
                    f"{resolved_path}:{line_number}"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(
                    "Each JSONL record must be "
                    "an object"
                )

            records.append(record)

    if not records:
        raise ValueError(
            f"No records found in {resolved_path}"
        )

    return records


def index_records(
    records: Iterable[dict[str, Any]],
    *,
    name: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[
        str,
        dict[str, Any],
    ] = {}

    for record in records:
        instance_id = record.get(
            "instance_id"
        )

        if not isinstance(
            instance_id,
            str,
        ) or not instance_id.strip():
            raise ValueError(
                f"{name} record has no valid "
                "instance_id"
            )

        if instance_id in indexed:
            raise ValueError(
                f"Duplicate instance_id in "
                f"{name}: {instance_id}"
            )

        indexed[instance_id] = record

    return indexed


def classify_utility(
    utility_delta: float,
    *,
    threshold: float,
) -> str:
    if threshold < 0.0:
        raise ValueError(
            "threshold must be non-negative"
        )

    if utility_delta >= threshold:
        return "helpful"

    if utility_delta <= -threshold:
        return "harmful"

    return "neutral"


def derive_harm_category(
    *,
    selected_correct: bool,
    empty_correct: bool,
    prediction_flipped: bool,
    utility_class: str,
) -> str:
    if (
        selected_correct
        and not empty_correct
    ):
        return "selected_prevents_error"

    if (
        not selected_correct
        and empty_correct
    ):
        return "selected_causes_error"

    if (
        selected_correct
        and empty_correct
        and utility_class == "harmful"
    ):
        return "confidence_harm_only"

    if (
        not selected_correct
        and not empty_correct
        and prediction_flipped
    ):
        return "wrong_to_different_wrong"

    if (
        not selected_correct
        and not empty_correct
        and utility_class == "harmful"
    ):
        return "wrong_and_more_confident"

    if utility_class == "helpful":
        return "probability_help"

    if utility_class == "harmful":
        return "probability_harm"

    return "neutral_effect"


def validate_paired_records(
    original: dict[str, Any],
    empty: dict[str, Any],
) -> None:
    fields = (
        "instance_id",
        "sentence_id",
        "domain",
        "split",
        "gold_label",
        "gold_label_id",
    )

    for field in fields:
        if original.get(field) != empty.get(
            field
        ):
            raise ValueError(
                "Original and empty records "
                f"differ on {field}: "
                f"{original.get('instance_id')}"
            )

    if empty.get("intervention") != "empty":
        raise ValueError(
            "Expected empty intervention record"
        )


def build_utility_record(
    *,
    original: dict[str, Any],
    empty: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    validate_paired_records(
        original,
        empty,
    )

    gold_label_id = int(
        original["gold_label_id"]
    )

    original_probabilities = [
        float(value)
        for value in original[
            "original_probabilities"
        ]
    ]

    empty_probabilities = [
        float(value)
        for value in empty[
            "intervened_probabilities"
        ]
    ]

    if not 0 <= gold_label_id < len(
        original_probabilities
    ):
        raise ValueError(
            "gold_label_id is outside the "
            "probability vector"
        )

    if len(original_probabilities) != len(
        empty_probabilities
    ):
        raise ValueError(
            "Original and empty probability "
            "dimensions do not match"
        )

    selected_gold_probability = (
        original_probabilities[
            gold_label_id
        ]
    )

    empty_gold_probability = (
        empty_probabilities[
            gold_label_id
        ]
    )

    utility_delta = (
        selected_gold_probability
        - empty_gold_probability
    )

    utility_class = classify_utility(
        utility_delta,
        threshold=threshold,
    )

    selected_prediction_id = int(
        original["original_prediction_id"]
    )

    empty_prediction_id = int(
        empty["intervened_prediction_id"]
    )

    selected_correct = (
        selected_prediction_id
        == gold_label_id
    )

    empty_correct = (
        empty_prediction_id
        == gold_label_id
    )

    prediction_flipped = (
        selected_prediction_id
        != empty_prediction_id
    )

    harm_category = derive_harm_category(
        selected_correct=selected_correct,
        empty_correct=empty_correct,
        prediction_flipped=prediction_flipped,
        utility_class=utility_class,
    )

    tokens = [
        str(token)
        for token in original["tokens"]
    ]

    selected_indices = [
        int(index)
        for index in original[
            "original_evidence_indices"
        ]
    ]

    selected_tokens = [
        str(token)
        for token in original[
            "original_evidence_tokens"
        ]
    ]

    return {
        "schema_version": "1.0",
        "instance_id": (
            original["instance_id"]
        ),
        "sentence_id": (
            original["sentence_id"]
        ),
        "domain": original["domain"],
        "split": original["split"],
        "sentence": " ".join(tokens),
        "tokens": tokens,
        "aspect_text": (
            original["aspect_text"]
        ),
        "aspect_start": int(
            original["aspect_start"]
        ),
        "aspect_end": int(
            original["aspect_end"]
        ),
        "gold_label": (
            original["gold_label"]
        ),
        "gold_label_id": gold_label_id,
        "selected_evidence_indices": (
            selected_indices
        ),
        "selected_evidence_tokens": (
            selected_tokens
        ),
        "selected_evidence_text": (
            " ".join(selected_tokens)
        ),
        "selected_prediction": (
            original[
                "original_prediction"
            ]
        ),
        "selected_prediction_id": (
            selected_prediction_id
        ),
        "empty_prediction": (
            empty[
                "intervened_prediction"
            ]
        ),
        "empty_prediction_id": (
            empty_prediction_id
        ),
        "selected_correct": (
            selected_correct
        ),
        "empty_correct": empty_correct,
        "prediction_flipped": (
            prediction_flipped
        ),
        "selected_gold_probability": (
            selected_gold_probability
        ),
        "empty_gold_probability": (
            empty_gold_probability
        ),
        "utility_delta": utility_delta,
        "absolute_utility_delta": abs(
            utility_delta
        ),
        "utility_threshold": threshold,
        "utility_class": utility_class,
        "harm_category": harm_category,
        "original_compatibility_score": (
            empty.get(
                "original_compatibility_score"
            )
        ),
        "empty_compatibility_score": (
            empty.get(
                "intervened_compatibility_score"
            )
        ),
        "compatibility_score_change": (
            empty.get(
                "compatibility_score_change"
            )
        ),
        "original_gate_value": (
            empty.get(
                "original_gate_value"
            )
        ),
        "empty_gate_value": (
            empty.get(
                "intervened_gate_value"
            )
        ),
        "gate_value_change": (
            empty.get(
                "gate_value_change"
            )
        ),
    }


def build_utility_audit(
    *,
    original_records: list[
        dict[str, Any]
    ],
    empty_records: list[
        dict[str, Any]
    ],
    threshold: float,
) -> list[dict[str, Any]]:
    if threshold < 0.0:
        raise ValueError(
            "threshold must be non-negative"
        )

    original_index = index_records(
        original_records,
        name="original",
    )

    empty_index = index_records(
        empty_records,
        name="empty",
    )

    if set(original_index) != set(
        empty_index
    ):
        missing_from_empty = sorted(
            set(original_index)
            - set(empty_index)
        )

        missing_from_original = sorted(
            set(empty_index)
            - set(original_index)
        )

        raise ValueError(
            "Original and empty instance sets "
            "do not match. "
            f"Missing from empty: "
            f"{missing_from_empty[:5]}; "
            f"missing from original: "
            f"{missing_from_original[:5]}"
        )

    audit = [
        build_utility_record(
            original=original_index[
                instance_id
            ],
            empty=empty_index[
                instance_id
            ],
            threshold=threshold,
        )
        for instance_id in sorted(
            original_index
        )
    ]

    return audit


def safe_mean(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return sum(values) / len(values)


def summarize_audit(
    records: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    if not records:
        raise ValueError(
            "Cannot summarize an empty audit"
        )

    class_counts = Counter(
        str(record["utility_class"])
        for record in records
    )

    harm_counts = Counter(
        str(record["harm_category"])
        for record in records
    )

    label_counts: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    for record in records:
        label_counts[
            str(record["gold_label"])
        ][
            str(record["utility_class"])
        ] += 1

    utility_values = [
        float(record["utility_delta"])
        for record in records
    ]

    available_records = [
        record
        for record in records
        if record[
            "selected_evidence_indices"
        ]
    ]

    available_utility_values = [
        float(record["utility_delta"])
        for record in available_records
    ]

    helpful = [
        value
        for value in utility_values
        if value >= threshold
    ]

    harmful = [
        value
        for value in utility_values
        if value <= -threshold
    ]

    selected_causes_error = [
        record
        for record in records
        if record["harm_category"]
        == "selected_causes_error"
    ]

    selected_prevents_error = [
        record
        for record in records
        if record["harm_category"]
        == "selected_prevents_error"
    ]

    sorted_harmful = sorted(
        records,
        key=lambda record: float(
            record["utility_delta"]
        ),
    )

    sorted_helpful = sorted(
        records,
        key=lambda record: float(
            record["utility_delta"]
        ),
        reverse=True,
    )

    number_of_records = len(records)

    return {
        "schema_version": "1.0",
        "utility_definition": (
            "P(gold | selected evidence) - "
            "P(gold | empty evidence)"
        ),
        "utility_threshold": threshold,
        "number_of_instances": (
            number_of_records
        ),
        "number_with_selected_evidence": (
            len(available_records)
        ),
        "utility_class_counts": dict(
            sorted(class_counts.items())
        ),
        "utility_class_rates": {
            name: count
            / number_of_records
            for name, count in sorted(
                class_counts.items()
            )
        },
        "harm_category_counts": dict(
            sorted(harm_counts.items())
        ),
        "label_utility_counts": {
            label: dict(
                sorted(counts.items())
            )
            for label, counts in sorted(
                label_counts.items()
            )
        },
        "mean_utility_delta": safe_mean(
            utility_values
        ),
        "mean_utility_delta_with_evidence": (
            safe_mean(
                available_utility_values
            )
        ),
        "mean_helpful_utility_delta": (
            safe_mean(helpful)
        ),
        "mean_harmful_utility_delta": (
            safe_mean(harmful)
        ),
        "minimum_utility_delta": min(
            utility_values
        ),
        "maximum_utility_delta": max(
            utility_values
        ),
        "number_selected_causes_error": (
            len(selected_causes_error)
        ),
        "number_selected_prevents_error": (
            len(selected_prevents_error)
        ),
        "top_harmful_instances": [
            {
                "instance_id": record[
                    "instance_id"
                ],
                "aspect_text": record[
                    "aspect_text"
                ],
                "gold_label": record[
                    "gold_label"
                ],
                "selected_evidence_text": (
                    record[
                        "selected_evidence_text"
                    ]
                ),
                "utility_delta": record[
                    "utility_delta"
                ],
                "harm_category": record[
                    "harm_category"
                ],
            }
            for record in sorted_harmful[
                :20
            ]
        ],
        "top_helpful_instances": [
            {
                "instance_id": record[
                    "instance_id"
                ],
                "aspect_text": record[
                    "aspect_text"
                ],
                "gold_label": record[
                    "gold_label"
                ],
                "selected_evidence_text": (
                    record[
                        "selected_evidence_text"
                    ]
                ),
                "utility_delta": record[
                    "utility_delta"
                ],
                "harm_category": record[
                    "harm_category"
                ],
            }
            for record in sorted_helpful[
                :20
            ]
        ],
    }


def write_jsonl(
    path: str | Path,
    records: Iterable[dict[str, Any]],
) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with resolved_path.open(
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
                + "\n"
            )

    return resolved_path


def csv_value(
    value: Any,
) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    return value


def write_csv(
    path: str | Path,
    records: list[dict[str, Any]],
) -> Path:
    if not records:
        raise ValueError(
            "Cannot write an empty CSV"
        )

    resolved_path = Path(path)
    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        records[0].keys()
    )

    with resolved_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    key: csv_value(
                        record.get(key)
                    )
                    for key in fieldnames
                }
            )

    return resolved_path


def write_json(
    path: str | Path,
    payload: dict[str, Any],
) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return resolved_path


def default_output_directory(
    intervention_directory: str | Path,
) -> Path:
    return (
        Path(intervention_directory)
        / "utility_audit"
    )


def run_audit(
    *,
    intervention_directory: str | Path,
    output_directory: (
        str | Path | None
    ),
    threshold: float,
    overwrite: bool,
) -> dict[str, Path]:
    intervention_path = Path(
        intervention_directory
    ).resolve()

    original_path = (
        intervention_path
        / "predictions"
        / "original.jsonl"
    )

    empty_path = (
        intervention_path
        / "predictions"
        / "empty.jsonl"
    )

    resolved_output = (
        default_output_directory(
            intervention_path
        )
        if output_directory is None
        else Path(output_directory).resolve()
    )

    if (
        resolved_output.exists()
        and not overwrite
    ):
        raise FileExistsError(
            "Utility audit output directory "
            f"already exists: {resolved_output}"
        )

    if resolved_output.exists():
        import shutil

        shutil.rmtree(
            resolved_output
        )

    original_records = read_jsonl(
        original_path
    )

    empty_records = read_jsonl(
        empty_path
    )

    audit = build_utility_audit(
        original_records=original_records,
        empty_records=empty_records,
        threshold=threshold,
    )

    summary = summarize_audit(
        audit,
        threshold=threshold,
    )

    jsonl_path = write_jsonl(
        resolved_output
        / "evidence_utility_audit.jsonl",
        audit,
    )

    csv_path = write_csv(
        resolved_output
        / "evidence_utility_audit.csv",
        audit,
    )

    summary_path = write_json(
        resolved_output
        / "evidence_utility_summary.json",
        summary,
    )

    return {
        "jsonl": jsonl_path,
        "csv": csv_path,
        "summary": summary_path,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the predictive utility of "
            "selected weak evidence relative "
            "to empty-evidence inference."
        )
    )

    parser.add_argument(
        "intervention_directory",
        type=Path,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()

    paths = run_audit(
        intervention_directory=(
            arguments.intervention_directory
        ),
        output_directory=(
            arguments.output_directory
        ),
        threshold=arguments.threshold,
        overwrite=arguments.overwrite,
    )

    print("=" * 88)
    print("EVIDENCE UTILITY AUDIT")
    print("=" * 88)

    for name, path in paths.items():
        print(
            f"{name.capitalize():<10} {path}"
        )


if __name__ == "__main__":
    main()
