from __future__ import annotations

from pathlib import Path

from src.aspect_sentiment.models.llm import (
    LLMABSADataset,
)


def test_loads_laptop_validation_split() -> None:
    root = Path("data/processed")

    if not (
        root
        / "laptops"
        / "validation.jsonl"
    ).is_file():
        return

    dataset = LLMABSADataset.from_split(
        root,
        domain="laptops",
        split="validation",
    )

    assert len(dataset) == 354
    assert len(
        set(dataset.instance_ids())
    ) == 354

    first = dataset[0]

    assert first.domain == "laptops"
    assert first.split == "validation"
    assert first.tokens
    assert (
        first.tokens[
            first.aspect_start:
            first.aspect_end
        ]
    )
