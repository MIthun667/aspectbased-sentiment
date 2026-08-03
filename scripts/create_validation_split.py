from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc

    return records


def write_jsonl(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def normalize_tokens(tokens: list[str]) -> str:
    """
    Normalize a tokenized sentence for duplicate-aware grouping.
    """
    text = " ".join(str(token) for token in tokens)
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text


def sentence_signature(
    records: list[dict[str, Any]],
) -> tuple[int, int, int]:
    counts = Counter(
        record["polarity"]
        for record in records
    )

    return (
        counts.get("negative", 0),
        counts.get("neutral", 0),
        counts.get("positive", 0),
    )


def duplicate_group_key(
    records: list[dict[str, Any]],
) -> str:
    """
    Return the normalized sentence text shared by one sentence group.
    """
    normalized_texts = {
        normalize_tokens(record["tokens"])
        for record in records
    }

    if len(normalized_texts) != 1:
        raise ValueError(
            "A sentence_id group contains inconsistent sentence text."
        )

    return next(iter(normalized_texts))


def split_sentence_groups(
    records: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
    excluded_validation_texts: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split data by normalized-sentence clusters.

    All aspects from the same source sentence remain together. Exact repeated
    normalized sentences also remain together, even when they have different
    source sentence identifiers.
    """
    if excluded_validation_texts is None:
        excluded_validation_texts = set()

    sentence_groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        sentence_groups[
            record["sentence_id"]
        ].append(record)

    duplicate_clusters: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for sentence_id, sentence_records in (
        sentence_groups.items()
    ):
        key = duplicate_group_key(sentence_records)
        duplicate_clusters[key].append(sentence_id)

    cluster_records: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for cluster_key, sentence_ids in (
        duplicate_clusters.items()
    ):
        cluster_records[cluster_key] = [
            record
            for sentence_id in sentence_ids
            for record in sentence_groups[sentence_id]
        ]

    signature_groups: dict[
        tuple[int, int, int],
        list[str],
    ] = defaultdict(list)

    for cluster_key, grouped_records in (
        cluster_records.items()
    ):
        signature = sentence_signature(
            grouped_records
        )
        signature_groups[signature].append(
            cluster_key
        )

    rng = random.Random(seed)
    validation_cluster_keys: set[str] = set()

    for cluster_keys in signature_groups.values():
        eligible_cluster_keys = sorted(
            cluster_key
            for cluster_key in cluster_keys
            if cluster_key not in excluded_validation_texts
        )
        rng.shuffle(eligible_cluster_keys)

        raw_count = (
            len(eligible_cluster_keys)
            * validation_ratio
        )
        validation_count = int(round(raw_count))

        if len(eligible_cluster_keys) > 1:
            validation_count = max(
                1,
                validation_count,
            )
            validation_count = min(
                validation_count,
                len(eligible_cluster_keys) - 1,
            )
        else:
            validation_count = 0

        validation_cluster_keys.update(
            eligible_cluster_keys[:validation_count]
        )

    validation_sentence_ids: set[str] = {
        sentence_id
        for cluster_key in validation_cluster_keys
        for sentence_id in duplicate_clusters[
            cluster_key
        ]
    }

    train_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []

    for record in records:
        if (
            record["sentence_id"]
            in validation_sentence_ids
        ):
            validation_records.append(record)
        else:
            train_records.append(record)

    train_sentence_ids = {
        record["sentence_id"]
        for record in train_records
    }
    validation_sentence_ids_check = {
        record["sentence_id"]
        for record in validation_records
    }

    sentence_overlap = (
        train_sentence_ids
        & validation_sentence_ids_check
    )

    if sentence_overlap:
        raise RuntimeError(
            "Sentence leakage detected: "
            f"{sorted(sentence_overlap)[:5]}"
        )

    train_texts = {
        normalize_tokens(record["tokens"])
        for record in train_records
    }
    validation_texts = {
        normalize_tokens(record["tokens"])
        for record in validation_records
    }

    text_overlap = train_texts & validation_texts

    if text_overlap:
        raise RuntimeError(
            "Normalized-text leakage detected: "
            f"{sorted(text_overlap)[:3]}"
        )

    return train_records, validation_records



def assign_split(
    records: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    """
    Return copied records with canonical split metadata.

    Sentence and instance identifiers are preserved as stable source
    identifiers. Only the operational split field is changed.
    """
    assigned: list[dict[str, Any]] = []

    for record in records:
        updated = dict(record)
        updated["split"] = split
        assigned.append(updated)

    return assigned


def summarize(
    name: str,
    records: list[dict[str, Any]],
) -> None:
    sentence_ids = {
        record["sentence_id"] for record in records
    }
    labels = Counter(record["polarity"] for record in records)
    multi_aspect_sentence_ids = {
        record["sentence_id"]
        for record in records
        if record["is_multi_aspect"]
    }

    print(f"{name}:")
    print(f"  instances:              {len(records)}")
    print(f"  sentences:              {len(sentence_ids)}")
    print(
        "  multi-aspect sentences: "
        f"{len(multi_aspect_sentence_ids)}"
    )
    print(f"  labels:                 {dict(labels)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    args = parser.parse_args()

    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError(
            "--validation-ratio must be between 0 and 1."
        )

    for domain in ("laptops", "restaurants", "tweets"):
        domain_directory = args.processed_root / domain
        original_train_path = domain_directory / "train.jsonl"
        full_train_path = domain_directory / "train_full.jsonl"

        if full_train_path.exists():
            source_path = full_train_path
        else:
            source_path = original_train_path

        records = load_jsonl(source_path)

        test_path = domain_directory / "test.jsonl"
        test_records = load_jsonl(test_path)

        excluded_validation_texts = {
            normalize_tokens(record["tokens"])
            for record in test_records
        }

        train_records, validation_records = split_sentence_groups(
            records=records,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            excluded_validation_texts=excluded_validation_texts,
        )

        train_records = assign_split(
            train_records,
            "train",
        )
        validation_records = assign_split(
            validation_records,
            "validation",
        )

        if not full_train_path.exists():
            write_jsonl(records, full_train_path)

        write_jsonl(
            train_records,
            domain_directory / "train.jsonl",
        )
        write_jsonl(
            validation_records,
            domain_directory / "validation.jsonl",
        )

        print("=" * 80)
        print(f"Domain: {domain}")
        summarize("Full training data", records)
        summarize("Training subset", train_records)
        summarize("Validation subset", validation_records)


if __name__ == "__main__":
    main()
