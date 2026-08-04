from __future__ import annotations

from src.aspect_sentiment.models.llm import (
    LLMABSAInstance,
    build_chat_messages,
    build_user_prompt,
)


def make_instance() -> LLMABSAInstance:
    return LLMABSAInstance(
        instance_id="instance-1",
        sentence_id="sentence-1",
        domain="laptops",
        split="validation",
        tokens=(
            "battery",
            "life",
            "is",
            "excellent",
        ),
        aspect_text="battery life",
        aspect_start=0,
        aspect_end=2,
        gold_sentiment="positive",
        gold_label_id=2,
        number_of_aspects=1,
        is_multi_aspect=False,
    )


def test_prompt_contains_indexed_tokens() -> None:
    prompt = build_user_prompt(
        make_instance(),
        prompt_mode=(
            "joint_evidence_"
            "sentiment_confidence"
        ),
    )

    assert "[0] battery" in prompt
    assert "[3] excellent" in prompt
    assert "battery life" in prompt
    assert "[0, 2)" in prompt
    assert "confidence" in prompt


def test_chat_messages_have_roles() -> None:
    messages = build_chat_messages(
        make_instance(),
        prompt_mode="sentiment_only",
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
