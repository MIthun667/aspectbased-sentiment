from src.data.reader import aspect_tokens_match
from src.data.schema import POLARITY_TO_ID


def test_aspect_span_match() -> None:
    tokens = ["The", "battery", "life", "is", "good", "."]

    assert aspect_tokens_match(
        sentence_tokens=tokens,
        aspect_tokens=["battery", "life"],
        start=1,
        end=3,
    )


def test_aspect_span_mismatch() -> None:
    tokens = ["The", "battery", "life", "is", "good", "."]

    assert not aspect_tokens_match(
        sentence_tokens=tokens,
        aspect_tokens=["battery"],
        start=1,
        end=3,
    )


def test_label_mapping() -> None:
    assert POLARITY_TO_ID["negative"] == 0
    assert POLARITY_TO_ID["neutral"] == 1
    assert POLARITY_TO_ID["positive"] == 2
