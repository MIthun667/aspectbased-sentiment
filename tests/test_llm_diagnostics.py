from __future__ import annotations

from src.aspect_sentiment.models.llm import (
    LLMABSAInstance,
    StructuredABSAOutput,
    diagnose_evidence,
)


def make_instance() -> LLMABSAInstance:
    return LLMABSAInstance(
        instance_id="instance-1",
        sentence_id="sentence-1",
        domain="laptops",
        split="validation",
        tokens=(
            "the",
            "screen",
            "is",
            "excellent",
        ),
        aspect_text="screen",
        aspect_start=1,
        aspect_end=2,
        gold_sentiment="positive",
        gold_label_id=2,
        number_of_aspects=1,
        is_multi_aspect=False,
    )


def test_detects_target_only_evidence() -> None:
    diagnostics = diagnose_evidence(
        make_instance(),
        StructuredABSAOutput(
            sentiment="positive",
            evidence_indices=(1,),
            confidence=1.0,
        ),
    )

    assert diagnostics.evidence_target_only
    assert diagnostics.evidence_contains_target
    assert not (
        diagnostics.evidence_contains_non_target
    )
    assert diagnostics.evidence_tokens == (
        "screen",
    )


def test_detects_sentiment_context() -> None:
    diagnostics = diagnose_evidence(
        make_instance(),
        StructuredABSAOutput(
            sentiment="positive",
            evidence_indices=(1, 3),
            confidence=0.9,
        ),
    )

    assert not diagnostics.evidence_target_only
    assert diagnostics.evidence_contains_target
    assert (
        diagnostics.evidence_contains_non_target
    )
    assert diagnostics.evidence_tokens == (
        "screen",
        "excellent",
    )


def test_non_target_evidence_is_allowed() -> None:
    diagnostics = diagnose_evidence(
        make_instance(),
        StructuredABSAOutput(
            sentiment="positive",
            evidence_indices=(3,),
            confidence=0.9,
        ),
    )

    assert not diagnostics.evidence_target_only
    assert not diagnostics.evidence_contains_target
    assert (
        diagnostics.evidence_contains_non_target
    )
