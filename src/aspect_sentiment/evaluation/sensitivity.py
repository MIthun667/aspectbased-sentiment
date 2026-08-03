from __future__ import annotations

import re
from typing import Any, Iterable


def normalize_text(
    text: str,
) -> str:
    normalized = text.casefold()
    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )
    return normalized.strip()


def record_text(
    record: dict[str, Any],
) -> str:
    tokens = record.get("tokens")

    if not isinstance(tokens, list):
        raise TypeError(
            "Canonical record tokens must be a list"
        )

    if not all(
        isinstance(token, str)
        for token in tokens
    ):
        raise TypeError(
            "Canonical record tokens must contain strings"
        )

    return " ".join(tokens)


def normalized_record_text(
    record: dict[str, Any],
) -> str:
    return normalize_text(
        record_text(record)
    )


def collect_normalized_texts(
    records: Iterable[dict[str, Any]],
) -> set[str]:
    return {
        normalized_record_text(record)
        for record in records
    }


def exclude_text_overlaps(
    evaluation_records: Iterable[dict[str, Any]],
    *,
    reference_records: Iterable[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    reference_texts = collect_normalized_texts(
        reference_records
    )

    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for record in evaluation_records:
        if normalized_record_text(record) in reference_texts:
            excluded.append(record)
        else:
            retained.append(record)

    return retained, excluded
