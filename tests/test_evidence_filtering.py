from __future__ import annotations

import pytest

from src.aspect_sentiment.evidence import (
    is_candidate_eligible,
)


@pytest.mark.parametrize(
    ("token", "pos_tag", "relation"),
    [
        ("the", "DT", "det"),
        ("it", "PRP", "dobj"),
        ("because", "IN", "mark"),
        ("and", "CC", "cc"),
        ("to", "TO", "mark"),
        ("does", "VBZ", "aux"),
        ("is", "VBZ", "cop"),
    ],
)
def test_function_words_are_ineligible(
    token: str,
    pos_tag: str,
    relation: str,
) -> None:
    assert not is_candidate_eligible(
        token=token,
        pos_tag=pos_tag,
        dependency_relation=relation,
        is_negation=False,
        is_intensifier=False,
    )


@pytest.mark.parametrize(
    ("token", "pos_tag", "relation"),
    [
        ("good", "JJ", "amod"),
        ("properly", "RB", "advmod"),
        ("overheat", "VB", "conj"),
        ("exchange", "NN", "dobj"),
    ],
)
def test_content_words_are_eligible(
    token: str,
    pos_tag: str,
    relation: str,
) -> None:
    assert is_candidate_eligible(
        token=token,
        pos_tag=pos_tag,
        dependency_relation=relation,
        is_negation=False,
        is_intensifier=False,
    )


@pytest.mark.parametrize(
    ("token", "pos_tag", "relation"),
    [
        ("not", "RB", "neg"),
        ("barely", "RB", "advmod"),
        ("very", "RB", "advmod"),
    ],
)
def test_modifiers_remain_eligible(
    token: str,
    pos_tag: str,
    relation: str,
) -> None:
    assert is_candidate_eligible(
        token=token,
        pos_tag=pos_tag,
        dependency_relation=relation,
        is_negation=token in {"not", "barely"},
        is_intensifier=token == "very",
    )


def test_unrelated_noun_is_ineligible() -> None:
    assert not is_candidate_eligible(
        token="laptop",
        pos_tag="NN",
        dependency_relation="compound",
        is_negation=False,
        is_intensifier=False,
    )
