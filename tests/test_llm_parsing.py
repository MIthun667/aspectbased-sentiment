from __future__ import annotations

from src.aspect_sentiment.models.llm import (
    parse_structured_output,
)


MODE = (
    "joint_evidence_"
    "sentiment_confidence"
)


def test_valid_output_parses() -> None:
    result = parse_structured_output(
        (
            '{"sentiment":"positive",'
            '"evidence_indices":[2,3],'
            '"confidence":0.9}'
        ),
        prompt_mode=MODE,
        number_of_tokens=5,
    )

    assert result.valid
    assert result.output is not None
    assert result.output.sentiment == (
        "positive"
    )
    assert result.output.evidence_indices == (
        2,
        3,
    )


def test_markdown_fence_is_invalid() -> None:
    result = parse_structured_output(
        (
            '```json\n'
            '{"sentiment":"positive",'
            '"evidence_indices":[2],'
            '"confidence":0.9}\n'
            '```'
        ),
        prompt_mode=MODE,
        number_of_tokens=5,
    )

    assert not result.valid
    assert result.error_code == (
        "invalid_json"
    )


def test_extra_key_is_invalid() -> None:
    result = parse_structured_output(
        (
            '{"sentiment":"positive",'
            '"evidence_indices":[2],'
            '"confidence":0.9,'
            '"explanation":"good"}'
        ),
        prompt_mode=MODE,
        number_of_tokens=5,
    )

    assert not result.valid
    assert result.error_code == (
        "extra_keys"
    )


def test_unsorted_indices_are_invalid() -> None:
    result = parse_structured_output(
        (
            '{"sentiment":"positive",'
            '"evidence_indices":[3,2],'
            '"confidence":0.9}'
        ),
        prompt_mode=MODE,
        number_of_tokens=5,
    )

    assert not result.valid
    assert result.error_code == (
        "schema_validation_error"
    )


def test_sentiment_only_mode() -> None:
    result = parse_structured_output(
        '{"sentiment":"neutral"}',
        prompt_mode="sentiment_only",
        number_of_tokens=5,
    )

    assert result.valid
    assert result.output is not None
    assert result.output.confidence is None
    assert result.output.evidence_indices == ()
