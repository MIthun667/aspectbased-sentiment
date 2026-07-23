from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

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


def audit_file(path: Path) -> None:
    records = load_jsonl(path)

    labels = Counter()
    sentence_to_aspects: dict[str, list[dict]] = defaultdict(list)
    relation_counts = Counter()
    maximum_distance = 0
    maximum_length = 0

    for record in records:
        tokens = record["tokens"]
        start = record["aspect_start"]
        end = record["aspect_end"]
        aspect_tokens = record["aspect_tokens"]
        dependency_heads = record["dependency_heads"]
        dependency_relations = record["dependency_relations"]
        distances = record["aspect_distances"]

        if not (
            len(tokens)
            == len(record["pos_tags"])
            == len(dependency_heads)
            == len(dependency_relations)
            == len(distances)
        ):
            raise AssertionError(
                f"Length mismatch in {record['instance_id']}"
            )

        if tokens[start:end] != aspect_tokens:
            raise AssertionError(
                f"Aspect mismatch in {record['instance_id']}: "
                f"{tokens[start:end]} != {aspect_tokens}"
            )

        for token_index in range(start, end):
            if distances[token_index] != 0:
                raise AssertionError(
                    f"Aspect token distance is not zero in "
                    f"{record['instance_id']}"
                )

        labels[record["polarity"]] += 1
        sentence_to_aspects[record["sentence_id"]].append(record)
        relation_counts.update(dependency_relations)

        maximum_length = max(maximum_length, len(tokens))
        maximum_distance = max(
            maximum_distance,
            max(distances, default=0),
        )

    multi_aspect_sentences = {
        sentence_id: aspects
        for sentence_id, aspects in sentence_to_aspects.items()
        if len(aspects) > 1
    }

    differing_sentiment_sentences = 0
    eligible_cross_aspect_targets = 0

    for aspects in multi_aspect_sentences.values():
        polarities = {item["polarity"] for item in aspects}

        if len(polarities) > 1:
            differing_sentiment_sentences += 1

            for item in aspects:
                if any(
                    other["polarity"] != item["polarity"]
                    for other in aspects
                ):
                    eligible_cross_aspect_targets += 1

    print("=" * 90)
    print(path)
    print(f"Instances:                          {len(records)}")
    print(f"Unique sentences:                   {len(sentence_to_aspects)}")
    print(f"Multi-aspect sentences:             {len(multi_aspect_sentences)}")
    print(
        "Different-sentiment multi-aspect:   "
        f"{differing_sentiment_sentences}"
    )
    print(
        "Eligible CAIR target instances:     "
        f"{eligible_cross_aspect_targets}"
    )
    print(f"Label distribution:                 {dict(labels)}")
    print(f"Maximum sentence length:            {maximum_length}")
    print(f"Maximum dependency distance:        {maximum_distance}")
    print(f"Dependency relation types:          {len(relation_counts)}")
    print(
        "Most common dependency relations:  "
        f"{relation_counts.most_common(10)}"
    )


def main() -> None:
    processed_root = PROJECT_ROOT / "data" / "processed"

    paths = sorted(processed_root.glob("*/*.jsonl"))

    if not paths:
        raise FileNotFoundError(
            f"No JSONL files found under {processed_root}"
        )

    for path in paths:
        audit_file(path)


if __name__ == "__main__":
    main()
