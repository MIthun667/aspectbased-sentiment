from __future__ import annotations

import json

from .schema import (
    LLMABSAInstance,
    SUPPORTED_PROMPT_MODES,
)


SYSTEM_PROMPT = """
You are a precise aspect-based sentiment analysis system.

Analyze sentiment only toward the specified target aspect.
Do not use the overall sentence sentiment when it conflicts
with the target-specific sentiment.

Evidence must be extractive. Return only zero-based token
indices from the supplied indexed token list.

Return exactly one valid JSON object. Do not use Markdown,
code fences, explanations, or additional text.
""".strip()


def format_indexed_tokens(
    tokens: tuple[str, ...],
) -> str:
    if not tokens:
        raise ValueError(
            "tokens must not be empty"
        )

    return "\n".join(
        f"[{index}] {token}"
        for index, token in enumerate(tokens)
    )


def expected_output_example(
    prompt_mode: str,
) -> str:
    if prompt_mode not in (
        SUPPORTED_PROMPT_MODES
    ):
        raise ValueError(
            "Unsupported prompt mode: "
            f"{prompt_mode!r}"
        )

    if prompt_mode == "sentiment_only":
        payload = {
            "sentiment": "positive",
        }
    elif (
        prompt_mode
        == "evidence_then_sentiment"
    ):
        payload = {
            "evidence_indices": [2, 3],
            "sentiment": "positive",
        }
    else:
        payload = {
            "sentiment": "positive",
            "evidence_indices": [2, 3],
            "confidence": 0.85,
        }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_user_prompt(
    instance: LLMABSAInstance,
    *,
    prompt_mode: str,
) -> str:
    instance.validate()

    if prompt_mode not in (
        SUPPORTED_PROMPT_MODES
    ):
        raise ValueError(
            "Unsupported prompt mode: "
            f"{prompt_mode!r}"
        )

    indexed_tokens = format_indexed_tokens(
        instance.tokens
    )

    output_example = expected_output_example(
        prompt_mode
    )

    instructions = {
        "sentiment_only": (
            "Predict only the sentiment toward the "
            "target aspect."
        ),
        "evidence_then_sentiment": (
            "First identify the target-specific "
            "evidence indices, then predict sentiment."
        ),
        "joint_evidence_sentiment_confidence": (
            "Jointly predict target-specific evidence, "
            "sentiment, and confidence."
        ),
    }[prompt_mode]

    return f"""
Task:
{instructions}

Allowed sentiment labels:
negative, neutral, positive

Sentence tokens:
{indexed_tokens}

Target aspect:
{instance.aspect_text}

Target token span:
[{instance.aspect_start}, {instance.aspect_end})

Required JSON format:
{output_example}
""".strip()


def build_chat_messages(
    instance: LLMABSAInstance,
    *,
    prompt_mode: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_user_prompt(
                instance,
                prompt_mode=prompt_mode,
            ),
        },
    ]
