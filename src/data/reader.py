from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.data.graph import (
    build_dependency_edges,
    convert_heads_to_zero_based,
    shortest_distances_from_aspect,
)
from src.data.schema import (
    ABSCInstance,
    POLARITY_TO_ID,
    VALID_POLARITIES,
)


@dataclass(slots=True)
class PreparationStatistics:
    domain: str
    split: str

    source_sentences: int = 0
    generated_instances: int = 0
    sentences_without_aspects: int = 0
    multi_aspect_sentences: int = 0
    invalid_sentences: int = 0
    invalid_aspects: int = 0

    label_counts: Counter[str] = field(default_factory=Counter)
    aspect_count_distribution: Counter[int] = field(
        default_factory=Counter
    )
    invalid_messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "split": self.split,
            "source_sentences": self.source_sentences,
            "generated_instances": self.generated_instances,
            "sentences_without_aspects": self.sentences_without_aspects,
            "multi_aspect_sentences": self.multi_aspect_sentences,
            "invalid_sentences": self.invalid_sentences,
            "invalid_aspects": self.invalid_aspects,
            "label_counts": dict(self.label_counts),
            "aspect_count_distribution": {
                str(key): value
                for key, value in sorted(
                    self.aspect_count_distribution.items()
                )
            },
            "invalid_messages": self.invalid_messages,
        }


def normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([.,!?;:])", r"\1", value)
    return value


def aspect_tokens_match(
    sentence_tokens: list[str],
    aspect_tokens: list[str],
    start: int,
    end: int,
) -> bool:
    span_tokens = sentence_tokens[start:end]

    span_text = normalize_text(" ".join(span_tokens))
    aspect_text = normalize_text(" ".join(aspect_tokens))

    return span_text == aspect_text


def validate_sentence_fields(
    record: dict[str, Any],
    *,
    sentence_index: int,
    source_path: Path,
) -> None:
    required_fields = {
        "token",
        "pos",
        "head",
        "deprel",
        "aspects",
    }

    missing_fields = required_fields.difference(record)

    if missing_fields:
        raise KeyError(
            f"{source_path}: sentence {sentence_index} missing fields: "
            f"{sorted(missing_fields)}"
        )

    tokens = record["token"]
    pos_tags = record["pos"]
    heads = record["head"]
    relations = record["deprel"]

    if not isinstance(tokens, list) or not tokens:
        raise ValueError(
            f"{source_path}: sentence {sentence_index} has no tokens."
        )

    lengths = {
        "token": len(tokens),
        "pos": len(pos_tags),
        "head": len(heads),
        "deprel": len(relations),
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"{source_path}: sentence {sentence_index} field-length "
            f"mismatch: {lengths}"
        )

    if not isinstance(record["aspects"], list):
        raise TypeError(
            f"{source_path}: sentence {sentence_index} aspects must be a "
            "list."
        )


def parse_aspect(
    aspect: dict[str, Any],
    *,
    sentence_tokens: list[str],
    sentence_index: int,
    aspect_index: int,
    source_path: Path,
) -> tuple[list[str], int, int, str]:
    required_fields = {"term", "from", "to", "polarity"}
    missing_fields = required_fields.difference(aspect)

    if missing_fields:
        raise KeyError(
            f"{source_path}: sentence {sentence_index}, aspect "
            f"{aspect_index} missing fields: {sorted(missing_fields)}"
        )

    aspect_tokens = aspect["term"]
    start = aspect["from"]
    end = aspect["to"]
    polarity = str(aspect["polarity"]).strip().lower()

    if not isinstance(aspect_tokens, list) or not aspect_tokens:
        raise ValueError(
            f"{source_path}: sentence {sentence_index}, aspect "
            f"{aspect_index} has empty term."
        )

    if not isinstance(start, int) or not isinstance(end, int):
        raise TypeError(
            f"{source_path}: sentence {sentence_index}, aspect "
            f"{aspect_index} span must contain integer indices."
        )

    if not 0 <= start < end <= len(sentence_tokens):
        raise ValueError(
            f"{source_path}: sentence {sentence_index}, aspect "
            f"{aspect_index} has invalid span [{start}, {end}) for "
            f"{len(sentence_tokens)} tokens."
        )

    if polarity not in VALID_POLARITIES:
        raise ValueError(
            f"{source_path}: sentence {sentence_index}, aspect "
            f"{aspect_index} has unsupported polarity {polarity!r}."
        )

    if not aspect_tokens_match(
        sentence_tokens=sentence_tokens,
        aspect_tokens=aspect_tokens,
        start=start,
        end=end,
    ):
        observed = sentence_tokens[start:end]

        raise ValueError(
            f"{source_path}: sentence {sentence_index}, aspect "
            f"{aspect_index} span mismatch. "
            f"term={aspect_tokens!r}, tokens[{start}:{end}]={observed!r}"
        )

    return aspect_tokens, start, end, polarity


def read_absc_instances(
    source_path: str | Path,
    *,
    domain: str,
    split: str,
    strict: bool = True,
) -> tuple[list[ABSCInstance], PreparationStatistics]:
    source_path = Path(source_path)

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    with source_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise TypeError(
            f"{source_path}: expected a top-level JSON list, found "
            f"{type(records).__name__}."
        )

    statistics = PreparationStatistics(
        domain=domain,
        split=split,
        source_sentences=len(records),
    )

    instances: list[ABSCInstance] = []

    for sentence_index, record in enumerate(records):
        sentence_id = f"{domain}:{split}:sentence-{sentence_index:06d}"

        try:
            validate_sentence_fields(
                record,
                sentence_index=sentence_index,
                source_path=source_path,
            )

            tokens = [str(token) for token in record["token"]]
            pos_tags = [str(tag) for tag in record["pos"]]
            raw_heads = [int(head) for head in record["head"]]
            dependency_relations = [
                str(relation) for relation in record["deprel"]
            ]
            aspects = record["aspects"]

            dependency_heads, root_indices = (
                convert_heads_to_zero_based(
                    heads=raw_heads,
                    number_of_tokens=len(tokens),
                )
            )

            graph_edges = [
                edge.to_list()
                for edge in build_dependency_edges(
                    dependency_heads,
                    dependency_relations,
                    add_reverse_edges=True,
                    add_self_loops=True,
                )
            ]

        except Exception as exc:
            statistics.invalid_sentences += 1
            statistics.invalid_messages.append(str(exc))

            if strict:
                raise

            continue

        number_of_aspects = len(aspects)
        statistics.aspect_count_distribution[number_of_aspects] += 1

        if number_of_aspects == 0:
            statistics.sentences_without_aspects += 1
            continue

        if number_of_aspects > 1:
            statistics.multi_aspect_sentences += 1

        for aspect_index, aspect in enumerate(aspects):
            try:
                (
                    aspect_tokens,
                    aspect_start,
                    aspect_end,
                    polarity,
                ) = parse_aspect(
                    aspect,
                    sentence_tokens=tokens,
                    sentence_index=sentence_index,
                    aspect_index=aspect_index,
                    source_path=source_path,
                )

                aspect_distances = shortest_distances_from_aspect(
                    dependency_heads=dependency_heads,
                    aspect_indices=range(aspect_start, aspect_end),
                )

                instance_id = (
                    f"{sentence_id}:aspect-{aspect_index:02d}"
                )

                instance = ABSCInstance(
                    instance_id=instance_id,
                    sentence_id=sentence_id,
                    domain=domain,
                    split=split,
                    tokens=tokens,
                    pos_tags=pos_tags,
                    dependency_heads=dependency_heads,
                    dependency_relations=dependency_relations,
                    aspect_text=" ".join(aspect_tokens),
                    aspect_tokens=aspect_tokens,
                    aspect_start=aspect_start,
                    aspect_end=aspect_end,
                    polarity=polarity,
                    label_id=POLARITY_TO_ID[polarity],
                    root_indices=root_indices,
                    graph_edges=graph_edges,
                    aspect_distances=aspect_distances,
                    number_of_aspects=number_of_aspects,
                    is_multi_aspect=number_of_aspects > 1,
                )

                instances.append(instance)
                statistics.generated_instances += 1
                statistics.label_counts[polarity] += 1

            except Exception as exc:
                statistics.invalid_aspects += 1
                statistics.invalid_messages.append(str(exc))

                if strict:
                    raise

    return instances, statistics


def write_jsonl(
    instances: Iterable[ABSCInstance],
    output_path: str | Path,
) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with output_path.open("w", encoding="utf-8") as file:
        for instance in instances:
            json.dump(
                instance.to_dict(),
                file,
                ensure_ascii=False,
            )
            file.write("\n")
            count += 1

    return count
