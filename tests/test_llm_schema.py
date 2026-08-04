from __future__ import annotations

import pytest

from src.aspect_sentiment.models.llm import (
    LLMABSAInstance,
    StructuredABSAOutput,
)


def make_instance() -> LLMABSAInstance:
    return LLMABSAInstance(
        instance_id="instance-1",
        sentence_id="sentence-1",
        domain="laptops",
        split="validation",
        tokens=(
            "the",
            "battery",
            "lasts",
            "all",
            "day",
        ),
        aspect_text="battery",
        aspect_start=1,
        aspect_end=2,
        gold_sentiment="positive",
        gold_label_id=2,
        number_of_aspects=1,
        is_multi_aspect=False,
    )


def test_valid_instance() -> None:
    instance = make_instance()
    instance.validate()


def test_valid_structured_output() -> None:
    output = StructuredABSAOutput(
        sentiment="positive",
        evidence_indices=(2, 3, 4),
        confidence=0.91,
    )

    output.validate(
        number_of_tokens=5,
    )

    assert output.sentiment_id == 2
    assert output.evidence_tokens(
        make_instance().tokens
    ) == (
        "lasts",
        "all",
        "day",
    )


def test_duplicate_evidence_rejected() -> None:
    output = StructuredABSAOutput(
        sentiment="positive",
        evidence_indices=(2, 2),
        confidence=0.8,
    )

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        output.validate(
            number_of_tokens=5,
        )


def test_out_of_range_evidence_rejected() -> None:
    output = StructuredABSAOutput(
        sentiment="positive",
        evidence_indices=(5,),
        confidence=0.8,
    )

    with pytest.raises(
        ValueError,
        match="outside",
    ):
        output.validate(
            number_of_tokens=5,
        )
