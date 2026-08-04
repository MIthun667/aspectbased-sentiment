from __future__ import annotations

import argparse
import hashlib
import json
import random
import shlex
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPOSITORY_ROOT),
    )


from src.aspect_sentiment.data import (
    load_canonical_split,
)


SCHEMA_VERSION = 1
SUPPORTED_DOMAINS = {
    "laptops",
    "restaurants",
    "tweets",
}


def utc_timestamp() -> str:
    return datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    ).isoformat()


def canonical_json_dumps(
    value: object,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
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


def creation_command() -> str:
    return " ".join(
        shlex.quote(argument)
        for argument in sys.argv
    )


def require_string(
    record: Mapping[str, Any],
    field: str,
) -> str:
    value = record.get(field)

    if not isinstance(value, str):
        raise TypeError(
            f"{field} must be a string"
        )

    if not value.strip():
        raise ValueError(
            f"{field} must not be empty"
        )

    return value


def require_label_id(
    record: Mapping[str, Any],
) -> int:
    value = record.get("label_id")

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise TypeError(
            "label_id must be an integer"
        )

    if value not in {0, 1, 2}:
        raise ValueError(
            f"Unsupported label_id: {value}"
        )

    return value


def group_records_by_sentence(
    records: Sequence[
        Mapping[str, Any]
    ],
) -> dict[
    str,
    list[dict[str, Any]],
]:
    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    seen_instance_ids: set[str] = set()

    for source_record in records:
        record = dict(source_record)

        instance_id = require_string(
            record,
            "instance_id",
        )

        sentence_id = require_string(
            record,
            "sentence_id",
        )

        require_label_id(record)

        if instance_id in seen_instance_ids:
            raise ValueError(
                "Duplicate instance_id: "
                f"{instance_id}"
            )

        seen_instance_ids.add(
            instance_id
        )

        grouped[sentence_id].append(
            record
        )

    if not grouped:
        raise ValueError(
            "No sentence groups found"
        )

    for sentence_records in grouped.values():
        sentence_records.sort(
            key=lambda record: (
                int(
                    record.get(
                        "aspect_start",
                        0,
                    )
                ),
                int(
                    record.get(
                        "aspect_end",
                        0,
                    )
                ),
                str(
                    record["instance_id"]
                ),
            )
        )

    return dict(grouped)


def group_label_counts(
    records: Sequence[
        Mapping[str, Any]
    ],
) -> Counter[int]:
    return Counter(
        require_label_id(record)
        for record in records
    )


def fold_assignment_cost(
    *,
    fold_instance_count: int,
    fold_label_counts: Counter[int],
    group_instance_count: int,
    group_label_counts_value: Counter[int],
    target_instance_count: float,
    target_label_counts: dict[int, float],
) -> float:
    projected_instances = (
        fold_instance_count
        + group_instance_count
    )

    instance_error = abs(
        projected_instances
        - target_instance_count
    ) / max(
        target_instance_count,
        1.0,
    )

    label_error = 0.0

    for label_id in (0, 1, 2):
        projected_label_count = (
            fold_label_counts[label_id]
            + group_label_counts_value[
                label_id
            ]
        )

        target = max(
            target_label_counts[
                label_id
            ],
            1.0,
        )

        label_error += abs(
            projected_label_count
            - target
        ) / target

    return (
        instance_error
        + label_error
    )


def assign_sentence_groups(
    records: Sequence[
        Mapping[str, Any]
    ],
    *,
    number_of_folds: int,
    seed: int,
) -> list[list[str]]:
    if number_of_folds < 2:
        raise ValueError(
            "number_of_folds must be at "
            "least two"
        )

    grouped = group_records_by_sentence(
        records
    )

    if number_of_folds > len(grouped):
        raise ValueError(
            "number_of_folds cannot exceed "
            "the number of sentence groups"
        )

    total_label_counts = group_label_counts(
        records
    )

    target_instance_count = (
        len(records) / number_of_folds
    )

    target_label_counts = {
        label_id: (
            total_label_counts[label_id]
            / number_of_folds
        )
        for label_id in (0, 1, 2)
    }

    random_generator = random.Random(
        seed
    )

    sentence_ids = sorted(
        grouped
    )

    random_generator.shuffle(
        sentence_ids
    )

    sentence_ids.sort(
        key=lambda sentence_id: (
            -len(grouped[sentence_id]),
            -max(
                group_label_counts(
                    grouped[sentence_id]
                ).values()
            ),
        )
    )

    folds: list[list[str]] = [
        []
        for _ in range(number_of_folds)
    ]

    fold_instance_counts = [
        0
        for _ in range(number_of_folds)
    ]

    fold_label_counts = [
        Counter()
        for _ in range(number_of_folds)
    ]

    for sentence_id in sentence_ids:
        sentence_records = grouped[
            sentence_id
        ]

        sentence_instance_count = len(
            sentence_records
        )

        sentence_label_counts = (
            group_label_counts(
                sentence_records
            )
        )

        costs = []

        for fold_index in range(
            number_of_folds
        ):
            cost = fold_assignment_cost(
                fold_instance_count=(
                    fold_instance_counts[
                        fold_index
                    ]
                ),
                fold_label_counts=(
                    fold_label_counts[
                        fold_index
                    ]
                ),
                group_instance_count=(
                    sentence_instance_count
                ),
                group_label_counts_value=(
                    sentence_label_counts
                ),
                target_instance_count=(
                    target_instance_count
                ),
                target_label_counts=(
                    target_label_counts
                ),
            )

            costs.append(
                (
                    fold_instance_counts[
                        fold_index
                    ],
                    cost,
                    len(
                        folds[fold_index]
                    ),
                    fold_index,
                )
            )

        _, _, _, selected_fold = min(
            costs
        )

        folds[selected_fold].append(
            sentence_id
        )

        fold_instance_counts[
            selected_fold
        ] += sentence_instance_count

        fold_label_counts[
            selected_fold
        ].update(
            sentence_label_counts
        )

    return [
        sorted(sentence_ids)
        for sentence_ids in folds
    ]


def instance_ids_for_sentences(
    grouped: Mapping[
        str,
        Sequence[Mapping[str, Any]],
    ],
    sentence_ids: Iterable[str],
) -> list[str]:
    output: list[str] = []

    for sentence_id in sorted(
        sentence_ids
    ):
        if sentence_id not in grouped:
            raise ValueError(
                "Unknown sentence_id: "
                f"{sentence_id}"
            )

        output.extend(
            str(record["instance_id"])
            for record in grouped[
                sentence_id
            ]
        )

    return sorted(output)


def split_statistics(
    records: Sequence[
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    sentence_ids = {
        require_string(
            record,
            "sentence_id",
        )
        for record in records
    }

    label_counts = group_label_counts(
        records
    )

    return {
        "instance_count": len(records),
        "sentence_count": len(
            sentence_ids
        ),
        "label_counts": {
            str(label_id): (
                label_counts[label_id]
            )
            for label_id in (0, 1, 2)
        },
    }


def build_crossfit_manifest(
    records: Sequence[
        Mapping[str, Any]
    ],
    *,
    domain: str,
    split: str,
    number_of_folds: int,
    seed: int,
) -> dict[str, Any]:
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(
            f"Unsupported domain: {domain}"
        )

    grouped = group_records_by_sentence(
        records
    )

    heldout_sentence_folds = (
        assign_sentence_groups(
            records,
            number_of_folds=(
                number_of_folds
            ),
            seed=seed,
        )
    )

    all_sentence_ids = set(
        grouped
    )

    fold_payloads = []

    heldout_instance_occurrences: Counter[
        str
    ] = Counter()

    for fold_index, heldout_sentences in enumerate(
        heldout_sentence_folds
    ):
        heldout_sentence_set = set(
            heldout_sentences
        )

        training_sentence_set = (
            all_sentence_ids
            - heldout_sentence_set
        )

        if (
            heldout_sentence_set
            & training_sentence_set
        ):
            raise RuntimeError(
                "Training and held-out "
                "sentences overlap"
            )

        heldout_instance_ids = (
            instance_ids_for_sentences(
                grouped,
                heldout_sentence_set,
            )
        )

        training_instance_ids = (
            instance_ids_for_sentences(
                grouped,
                training_sentence_set,
            )
        )

        heldout_instance_occurrences.update(
            heldout_instance_ids
        )

        heldout_records = [
            record
            for record in records
            if str(
                record["instance_id"]
            )
            in set(
                heldout_instance_ids
            )
        ]

        training_records = [
            record
            for record in records
            if str(
                record["instance_id"]
            )
            in set(
                training_instance_ids
            )
        ]

        fold_payloads.append(
            {
                "fold_index": (
                    fold_index
                ),
                "training_sentence_ids": (
                    sorted(
                        training_sentence_set
                    )
                ),
                "heldout_sentence_ids": (
                    sorted(
                        heldout_sentence_set
                    )
                ),
                "training_instance_ids": (
                    training_instance_ids
                ),
                "heldout_instance_ids": (
                    heldout_instance_ids
                ),
                "training_statistics": (
                    split_statistics(
                        training_records
                    )
                ),
                "heldout_statistics": (
                    split_statistics(
                        heldout_records
                    )
                ),
            }
        )

    all_instance_ids = {
        require_string(
            record,
            "instance_id",
        )
        for record in records
    }

    if set(
        heldout_instance_occurrences
    ) != all_instance_ids:
        raise RuntimeError(
            "Held-out folds do not cover "
            "every instance"
        )

    if any(
        count != 1
        for count in (
            heldout_instance_occurrences
            .values()
        )
    ):
        raise RuntimeError(
            "Each instance must be held out "
            "exactly once"
        )

    return {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "domain": domain,
        "source_split": split,
        "seed": seed,
        "number_of_folds": (
            number_of_folds
        ),
        "global_statistics": (
            split_statistics(records)
        ),
        "folds": fold_payloads,
    }


def write_json(
    value: Mapping[str, Any],
    path: Path,
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


def write_json_list(
    values: Sequence[str],
    path: Path,
) -> None:
    write_json(
        {
            "schema_version": (
                SCHEMA_VERSION
            ),
            "count": len(values),
            "instance_ids": list(
                values
            ),
        },
        path,
    )


def run_builder(
    *,
    processed_root: Path,
    output_directory: Path,
    domain: str,
    split: str,
    number_of_folds: int,
    seed: int,
    overwrite: bool,
) -> Path:
    output_directory = (
        output_directory.resolve()
    )

    if (
        output_directory.exists()
        and not overwrite
    ):
        raise FileExistsError(
            "Cross-fit output directory "
            f"already exists: "
            f"{output_directory}"
        )

    if output_directory.exists():
        import shutil

        shutil.rmtree(
            output_directory
        )

    records = load_canonical_split(
        processed_root,
        domain=domain,
        split=split,
    )

    manifest = build_crossfit_manifest(
        records,
        domain=domain,
        split=split,
        number_of_folds=(
            number_of_folds
        ),
        seed=seed,
    )

    fold_file_payloads = []

    for fold in manifest["folds"]:
        fold_index = int(
            fold["fold_index"]
        )

        fold_directory = (
            output_directory
            / f"fold_{fold_index}"
        )

        training_path = (
            fold_directory
            / "train_instance_ids.json"
        )

        heldout_path = (
            fold_directory
            / "heldout_instance_ids.json"
        )

        write_json_list(
            fold[
                "training_instance_ids"
            ],
            training_path,
        )

        write_json_list(
            fold[
                "heldout_instance_ids"
            ],
            heldout_path,
        )

        fold_file_payloads.append(
            {
                "fold_index": (
                    fold_index
                ),
                "training_instance_file": (
                    str(
                        training_path.resolve()
                    )
                ),
                "training_instance_sha256": (
                    sha256_file(
                        training_path
                    )
                ),
                "heldout_instance_file": (
                    str(
                        heldout_path.resolve()
                    )
                ),
                "heldout_instance_sha256": (
                    sha256_file(
                        heldout_path
                    )
                ),
            }
        )

    manifest_path = (
        output_directory
        / "fold_manifest.json"
    )

    manifest_to_write = {
        **manifest,
        "created_at": utc_timestamp(),
        "creation_command": (
            creation_command()
        ),
        "processed_root": str(
            processed_root.resolve()
        ),
        "fold_files": (
            fold_file_payloads
        ),
    }

    write_json(
        manifest_to_write,
        manifest_path,
    )

    return manifest_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic grouped "
            "cross-fitting folds."
        )
    )

    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(
            "data/processed"
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--domain",
        choices=sorted(
            SUPPORTED_DOMAINS
        ),
        required=True,
    )

    parser.add_argument(
        "--split",
        default="train",
    )

    parser.add_argument(
        "--number-of-folds",
        type=int,
        default=2,
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

    return parser


def main() -> None:
    arguments = (
        build_argument_parser()
        .parse_args()
    )

    manifest_path = run_builder(
        processed_root=(
            arguments.processed_root
        ),
        output_directory=(
            arguments.output_directory
        ),
        domain=arguments.domain,
        split=arguments.split,
        number_of_folds=(
            arguments.number_of_folds
        ),
        seed=arguments.seed,
        overwrite=arguments.overwrite,
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    print("=" * 88)
    print("CROSS-FIT FOLD ARTIFACT")
    print("=" * 88)
    print(
        "Manifest:",
        manifest_path,
    )
    print(
        "Instances:",
        manifest[
            "global_statistics"
        ]["instance_count"],
    )
    print(
        "Sentences:",
        manifest[
            "global_statistics"
        ]["sentence_count"],
    )

    for fold in manifest["folds"]:
        print("-" * 88)
        print(
            "Fold:",
            fold["fold_index"],
        )
        print(
            "Training instances:",
            fold[
                "training_statistics"
            ]["instance_count"],
        )
        print(
            "Held-out instances:",
            fold[
                "heldout_statistics"
            ]["instance_count"],
        )
        print(
            "Held-out labels:",
            fold[
                "heldout_statistics"
            ]["label_counts"],
        )


if __name__ == "__main__":
    main()
