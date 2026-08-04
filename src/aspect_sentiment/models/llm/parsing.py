from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .schema import (
    StructuredABSAOutput,
    SUPPORTED_PROMPT_MODES,
)


@dataclass(frozen=True, slots=True)
class StructuredOutputParseResult:
    valid: bool
    output: StructuredABSAOutput | None
    error_code: str | None
    error_message: str | None
    raw_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "output": (
                None
                if self.output is None
                else self.output.to_dict()
            ),
            "error_code": self.error_code,
            "error_message": (
                self.error_message
            ),
            "raw_text": self.raw_text,
        }


def expected_keys(
    prompt_mode: str,
) -> set[str]:
    if prompt_mode not in (
        SUPPORTED_PROMPT_MODES
    ):
        raise ValueError(
            "Unsupported prompt mode: "
            f"{prompt_mode!r}"
        )

    if prompt_mode == "sentiment_only":
        return {"sentiment"}

    if (
        prompt_mode
        == "evidence_then_sentiment"
    ):
        return {
            "sentiment",
            "evidence_indices",
        }

    return {
        "sentiment",
        "evidence_indices",
        "confidence",
    }


def parse_structured_output(
    raw_text: str,
    *,
    prompt_mode: str,
    number_of_tokens: int,
    strict_keys: bool = True,
) -> StructuredOutputParseResult:
    if not isinstance(raw_text, str):
        raise TypeError(
            "raw_text must be a string"
        )

    stripped = raw_text.strip()

    if not stripped:
        return StructuredOutputParseResult(
            valid=False,
            output=None,
            error_code="empty_output",
            error_message=(
                "Model output is empty"
            ),
            raw_text=raw_text,
        )

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as error:
        return StructuredOutputParseResult(
            valid=False,
            output=None,
            error_code="invalid_json",
            error_message=str(error),
            raw_text=raw_text,
        )

    if not isinstance(payload, dict):
        return StructuredOutputParseResult(
            valid=False,
            output=None,
            error_code="non_object_json",
            error_message=(
                "Output JSON must be an object"
            ),
            raw_text=raw_text,
        )

    required_keys = expected_keys(
        prompt_mode
    )

    actual_keys = set(payload)

    missing_keys = sorted(
        required_keys - actual_keys
    )

    if missing_keys:
        return StructuredOutputParseResult(
            valid=False,
            output=None,
            error_code="missing_keys",
            error_message=(
                "Missing required keys: "
                f"{missing_keys}"
            ),
            raw_text=raw_text,
        )

    if strict_keys:
        extra_keys = sorted(
            actual_keys - required_keys
        )

        if extra_keys:
            return StructuredOutputParseResult(
                valid=False,
                output=None,
                error_code="extra_keys",
                error_message=(
                    "Unexpected keys: "
                    f"{extra_keys}"
                ),
                raw_text=raw_text,
            )

    sentiment = payload.get(
        "sentiment"
    )

    if not isinstance(sentiment, str):
        return StructuredOutputParseResult(
            valid=False,
            output=None,
            error_code="invalid_sentiment_type",
            error_message=(
                "sentiment must be a string"
            ),
            raw_text=raw_text,
        )

    evidence_value = payload.get(
        "evidence_indices",
        [],
    )

    if not isinstance(
        evidence_value,
        list,
    ):
        return StructuredOutputParseResult(
            valid=False,
            output=None,
            error_code="invalid_evidence_type",
            error_message=(
                "evidence_indices must be a list"
            ),
            raw_text=raw_text,
        )

    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        for value in evidence_value
    ):
        return StructuredOutputParseResult(
            valid=False,
            output=None,
            error_code="invalid_evidence_index",
            error_message=(
                "Every evidence index must be "
                "an integer"
            ),
            raw_text=raw_text,
        )

    confidence = payload.get(
        "confidence"
    )

    output = StructuredABSAOutput(
        sentiment=sentiment.strip().lower(),
        evidence_indices=tuple(
            evidence_value
        ),
        confidence=confidence,
    )

    try:
        output.validate(
            number_of_tokens=(
                number_of_tokens
            ),
            require_evidence=(
                prompt_mode
                != "sentiment_only"
            ),
            require_confidence=(
                prompt_mode
                in {
                    (
                        "joint_evidence_"
                        "sentiment_confidence"
                    ),
                    (
                        "three_shot_joint_evidence_"
                        "sentiment_confidence"
                    ),
                }
            ),
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        return StructuredOutputParseResult(
            valid=False,
            output=None,
            error_code="schema_validation_error",
            error_message=str(error),
            raw_text=raw_text,
        )

    return StructuredOutputParseResult(
        valid=True,
        output=output,
        error_code=None,
        error_message=None,
        raw_text=raw_text,
    )


@dataclass(frozen=True, slots=True)
class RecoverableStructuredOutputResult:
    strict_result: StructuredOutputParseResult
    recoverable_valid: bool
    recovered_output: StructuredABSAOutput | None
    recovery_actions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strict_valid": (
                self.strict_result.valid
            ),
            "strict_error_code": (
                self.strict_result.error_code
            ),
            "recoverable_valid": (
                self.recoverable_valid
            ),
            "recovered_output": (
                None
                if self.recovered_output is None
                else self.recovered_output.to_dict()
            ),
            "recovery_actions": list(
                self.recovery_actions
            ),
            "raw_text": (
                self.strict_result.raw_text
            ),
        }


def parse_recoverable_structured_output(
    raw_text: str,
    *,
    prompt_mode: str,
    number_of_tokens: int,
    strict_keys: bool = True,
) -> RecoverableStructuredOutputResult:
    strict_result = parse_structured_output(
        raw_text,
        prompt_mode=prompt_mode,
        number_of_tokens=number_of_tokens,
        strict_keys=strict_keys,
    )

    if strict_result.valid:
        return RecoverableStructuredOutputResult(
            strict_result=strict_result,
            recoverable_valid=True,
            recovered_output=(
                strict_result.output
            ),
            recovery_actions=(),
        )

    try:
        payload = json.loads(
            raw_text.strip()
        )
    except (
        json.JSONDecodeError,
        TypeError,
    ):
        return RecoverableStructuredOutputResult(
            strict_result=strict_result,
            recoverable_valid=False,
            recovered_output=None,
            recovery_actions=(),
        )

    if not isinstance(payload, dict):
        return RecoverableStructuredOutputResult(
            strict_result=strict_result,
            recoverable_valid=False,
            recovered_output=None,
            recovery_actions=(),
        )

    required_keys = expected_keys(
        prompt_mode
    )

    actual_keys = set(payload)

    if required_keys - actual_keys:
        return RecoverableStructuredOutputResult(
            strict_result=strict_result,
            recoverable_valid=False,
            recovered_output=None,
            recovery_actions=(),
        )

    if (
        strict_keys
        and actual_keys - required_keys
    ):
        return RecoverableStructuredOutputResult(
            strict_result=strict_result,
            recoverable_valid=False,
            recovered_output=None,
            recovery_actions=(),
        )

    evidence_value = payload.get(
        "evidence_indices",
        [],
    )

    if not isinstance(
        evidence_value,
        list,
    ):
        return RecoverableStructuredOutputResult(
            strict_result=strict_result,
            recoverable_valid=False,
            recovered_output=None,
            recovery_actions=(),
        )

    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        for value in evidence_value
    ):
        return RecoverableStructuredOutputResult(
            strict_result=strict_result,
            recoverable_valid=False,
            recovered_output=None,
            recovery_actions=(),
        )

    recovery_actions = []

    normalized_indices = tuple(
        sorted(set(evidence_value))
    )

    if normalized_indices != tuple(
        evidence_value
    ):
        recovery_actions.append(
            "sort_and_deduplicate_evidence_indices"
        )

    sentiment = payload.get(
        "sentiment"
    )

    if not isinstance(sentiment, str):
        return RecoverableStructuredOutputResult(
            strict_result=strict_result,
            recoverable_valid=False,
            recovered_output=None,
            recovery_actions=tuple(
                recovery_actions
            ),
        )

    output = StructuredABSAOutput(
        sentiment=sentiment.strip().lower(),
        evidence_indices=(
            normalized_indices
        ),
        confidence=payload.get(
            "confidence"
        ),
    )

    try:
        output.validate(
            number_of_tokens=(
                number_of_tokens
            ),
            require_evidence=(
                prompt_mode
                != "sentiment_only"
            ),
            require_confidence=(
                prompt_mode
                in {
                    (
                        "joint_evidence_"
                        "sentiment_confidence"
                    ),
                    (
                        "three_shot_joint_evidence_"
                        "sentiment_confidence"
                    ),
                }
            ),
        )
    except (
        TypeError,
        ValueError,
    ):
        return RecoverableStructuredOutputResult(
            strict_result=strict_result,
            recoverable_valid=False,
            recovered_output=None,
            recovery_actions=tuple(
                recovery_actions
            ),
        )

    return RecoverableStructuredOutputResult(
        strict_result=strict_result,
        recoverable_valid=True,
        recovered_output=output,
        recovery_actions=tuple(
            recovery_actions
        ),
    )
