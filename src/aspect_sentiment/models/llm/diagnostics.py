from __future__ import annotations

from dataclasses import asdict, dataclass

from .schema import (
    LLMABSAInstance,
    StructuredABSAOutput,
)


@dataclass(frozen=True, slots=True)
class EvidenceDiagnostics:
    evidence_count: int
    evidence_empty: bool
    evidence_target_only: bool
    evidence_contains_target: bool
    evidence_contains_non_target: bool
    evidence_indices_valid: bool
    evidence_tokens: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["evidence_tokens"] = list(
            self.evidence_tokens
        )
        return value


def diagnose_evidence(
    instance: LLMABSAInstance,
    output: StructuredABSAOutput,
) -> EvidenceDiagnostics:
    instance.validate()

    target_indices = set(
        range(
            instance.aspect_start,
            instance.aspect_end,
        )
    )

    evidence_indices = set(
        output.evidence_indices
    )

    indices_valid = all(
        0 <= index < len(instance.tokens)
        for index in output.evidence_indices
    )

    contains_target = bool(
        evidence_indices & target_indices
    )

    contains_non_target = bool(
        evidence_indices - target_indices
    )

    target_only = bool(
        evidence_indices
    ) and evidence_indices.issubset(
        target_indices
    )

    evidence_tokens = tuple(
        instance.tokens[index]
        for index in output.evidence_indices
        if 0 <= index < len(instance.tokens)
    )

    return EvidenceDiagnostics(
        evidence_count=len(
            output.evidence_indices
        ),
        evidence_empty=not bool(
            output.evidence_indices
        ),
        evidence_target_only=target_only,
        evidence_contains_target=(
            contains_target
        ),
        evidence_contains_non_target=(
            contains_non_target
        ),
        evidence_indices_valid=(
            indices_valid
        ),
        evidence_tokens=evidence_tokens,
    )
