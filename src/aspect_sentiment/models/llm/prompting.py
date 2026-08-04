from __future__ import annotations

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

Evidence must contain the words that express, justify, or
contextualize the sentiment toward the specified target.
The target token alone is not sentiment evidence. The selected
indices must include at least one token outside the target span.
Use the smallest sufficient evidence set.

Return exactly one valid JSON object. Do not use Markdown,
code fences, explanations, or additional text.
""".strip()


THREE_SHOT_DEMONSTRATIONS = (
    {
        "tokens": (
            "the",
            "keyboard",
            "feels",
            "excellent",
        ),
        "aspect_text": "keyboard",
        "aspect_start": 1,
        "aspect_end": 2,
        "output": (
            '{"sentiment":"positive",'
            '"evidence_indices":[2,3],'
            '"confidence":0.90}'
        ),
    },
    {
        "tokens": (
            "the",
            "battery",
            "drains",
            "far",
            "too",
            "quickly",
        ),
        "aspect_text": "battery",
        "aspect_start": 1,
        "aspect_end": 2,
        "output": (
            '{"sentiment":"negative",'
            '"evidence_indices":[2,3,4,5],'
            '"confidence":0.90}'
        ),
    },
    {
        "tokens": (
            "the",
            "package",
            "includes",
            "a",
            "standard",
            "charger",
        ),
        "aspect_text": "charger",
        "aspect_start": 5,
        "aspect_end": 6,
        "output": (
            '{"sentiment":"neutral",'
            '"evidence_indices":[2,3,4],'
            '"confidence":0.75}'
        ),
    },
)


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


def expected_output_schema(
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
        return (
            "Required keys:\n"
            "- sentiment: one of negative, neutral, positive"
        )

    if (
        prompt_mode
        == "evidence_then_sentiment"
    ):
        return (
            "Required keys:\n"
            "- evidence_indices: sorted list of unique "
            "zero-based token indices\n"
            "- sentiment: one of negative, neutral, positive"
        )

    return (
        "Required keys:\n"
        "- sentiment: one of negative, neutral, positive\n"
        "- evidence_indices: sorted list of unique "
        "zero-based token indices\n"
        "- confidence: numeric value from 0.0 to 1.0"
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

    output_schema = expected_output_schema(
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
        (
            "three_shot_joint_evidence_"
            "sentiment_confidence"
        ): (
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

Evidence selection rules:
- Select the smallest token set that explains the sentiment.
- Do not return only the target token or target span.
- Include at least one sentiment-bearing or contextual token
  outside the target span.
- Do not select unrelated sentiment about another aspect.

Output requirements:
{output_schema}

Return one JSON object containing exactly those keys.
Choose all values from the current sentence and target.
""".strip()


def format_demonstration_user_prompt(
    demonstration: dict[str, object],
) -> str:
    tokens = tuple(
        str(token)
        for token in demonstration["tokens"]
    )

    return f"""
Task:
Jointly predict target-specific evidence, sentiment,
and confidence.

Allowed sentiment labels:
negative, neutral, positive

Sentence tokens:
{format_indexed_tokens(tokens)}

Target aspect:
{demonstration["aspect_text"]}

Target token span:
[{demonstration["aspect_start"]}, {demonstration["aspect_end"]})

Evidence selection rules:
- Select the smallest token set that explains the sentiment.
- Do not return only the target token or target span.
- Include at least one sentiment-bearing or contextual token
  outside the target span.
- Do not select unrelated sentiment about another aspect.

Return one JSON object.
""".strip()


def build_three_shot_messages(
    instance: LLMABSAInstance,
) -> list[dict[str, str]]:
    instance.validate()

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    for demonstration in (
        THREE_SHOT_DEMONSTRATIONS
    ):
        messages.append(
            {
                "role": "user",
                "content": (
                    format_demonstration_user_prompt(
                        demonstration
                    )
                ),
            }
        )

        messages.append(
            {
                "role": "assistant",
                "content": str(
                    demonstration["output"]
                ),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": build_user_prompt(
                instance,
                prompt_mode=(
                    "joint_evidence_"
                    "sentiment_confidence"
                ),
            ),
        }
    )

    return messages


def build_chat_messages(
    instance: LLMABSAInstance,
    *,
    prompt_mode: str,
) -> list[dict[str, str]]:
    if (
        prompt_mode
        == (
            "three_shot_joint_evidence_"
            "sentiment_confidence"
        )
    ):
        return build_three_shot_messages(
            instance
        )

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
