from __future__ import annotations

import argparse
import json
import random
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


def sentence_signature(
    records: list[dict[str, Any]],
) -> tuple[int, int, int]:
    counts = Counter(record["polarity"] for record in records)

    return (
        counts.get("negative", 0),
        counts.get("neutral", 0),
        counts.get("positive", 0),
    )


def split_sentence_groups(
    records: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sentence_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        sentence_groups[record["sentence_id"]].append(record)

    signature_groups: dict[
        tuple[int, int, int],
        list[str],
    ] = defaultdict(list)

    for sentence_id, sentence_records in sentence_groups.items():
        signature = sentence_signature(sentence_records)
        signature_groups[signature].append(sentence_id)

    rng = random.Random(seed)
    validation_sentence_ids: set[str] = set()

    for sentence_ids in signature_groups.values():
        rng.shuffle(sentence_ids)

        raw_count = len(sentence_ids) * validation_ratio
        validation_count = int(round(raw_count))

        if len(sentence_ids) > 1:
            validation_count = max(1, validation_count)
            validation_count = min(
                validation_count,
                len(sentence_ids) - 1,
            )
        else:
            validation_count = 0

        validation_sentence_ids.update(
            sentence_ids[:validation_count]
        )

    train_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []

    for record in records:
        if record["sentence_id"] in validation_sentence_ids:
            validation_records.append(record)
        else:
            train_records.append(record)

    train_sentence_ids = {
        record["sentence_id"] for record in train_records
    }
    validation_sentence_ids_check = {
        record["sentence_id"] for record in validation_records
    }

    overlap = train_sentence_ids.intersection(
        validation_sentence_ids_check
    )

    if overlap:
        raise RuntimeError(
            f"Sentence leakage detected: {sorted(overlap)[:5]}"
        )

    return train_records, validation_records


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

        train_records, validation_records = split_sentence_groups(
            records=records,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
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
