from __future__ import annotations

import pytest

from src.aspect_sentiment.evidence import (
    dependency_ancestor_path,
    evaluate_clause_compatibility,
    find_aspect_clause_anchor,
    find_clause_anchor,
    is_predicate_conjunction,
    validate_dependency_structure,
)


def make_record() -> dict[str, object]:
    return {
        "tokens": [
            "Easy",
            "to",
            "start",
            "up",
            "and",
            "does",
            "not",
            "overheat",
            ".",
        ],
        "pos_tags": [
            "JJ",
            "TO",
            "VB",
            "RP",
            "CC",
            "VBZ",
            "RB",
            "VB",
            ".",
        ],
        "dependency_heads": [
            2,
            2,
            -1,
            2,
            7,
            7,
            7,
            2,
            2,
        ],
        "dependency_relations": [
            "advmod",
            "mark",
            "ROOT",
            "compound:prt",
            "cc",
            "aux",
            "neg",
            "conj",
            "punct",
        ],
        "aspect_start": 2,
        "aspect_end": 4,
    }


def test_validates_dependency_structure() -> None:
    heads, relations = (
        validate_dependency_structure(
            heads=[1, -1],
            relations=["nsubj", "ROOT"],
        )
    )

    assert heads == (1, -1)
    assert relations == (
        "nsubj",
        "ROOT",
    )


def test_rejects_invalid_head() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid dependency head",
    ):
        validate_dependency_structure(
            heads=[4, -1],
            relations=["nsubj", "ROOT"],
        )


def test_dependency_ancestor_path() -> None:
    assert dependency_ancestor_path(
        6,
        heads=make_record()[
            "dependency_heads"
        ],
    ) == (6, 7, 2)


def test_root_is_clause_anchor() -> None:
    record = make_record()

    assignment = find_clause_anchor(
        2,
        heads=record["dependency_heads"],
        relations=record[
            "dependency_relations"
        ],
        pos_tags=record["pos_tags"],
    )

    assert assignment.clause_anchor == 2


def test_verbal_conjunction_is_predicate() -> None:
    record = make_record()

    assert is_predicate_conjunction(
        7,
        heads=tuple(
            record["dependency_heads"]
        ),
        relations=tuple(
            record["dependency_relations"]
        ),
        pos_tags=tuple(
            record["pos_tags"]
        ),
    )


def test_verbal_conjunction_starts_clause() -> None:
    record = make_record()

    assignment = find_clause_anchor(
        7,
        heads=record["dependency_heads"],
        relations=record[
            "dependency_relations"
        ],
        pos_tags=record["pos_tags"],
    )

    assert assignment.clause_anchor == 7


def test_xcomp_does_not_always_start_clause() -> None:
    assignment = find_clause_anchor(
        2,
        heads=[1, -1, 1],
        relations=[
            "nsubj",
            "ROOT",
            "xcomp",
        ],
        pos_tags=[
            "PRP",
            "VB",
            "VB",
        ],
    )

    assert assignment.clause_anchor == 1


def test_relative_clause_does_not_always_split() -> None:
    assignment = find_clause_anchor(
        2,
        heads=[1, -1, 1],
        relations=[
            "det",
            "ROOT",
            "acl:relcl",
        ],
        pos_tags=[
            "DT",
            "NN",
            "JJ",
        ],
    )

    assert assignment.clause_anchor == 1


def test_aspect_uses_start_clause() -> None:
    assert (
        find_aspect_clause_anchor(
            make_record()
        )
        == 2
    )


def test_easy_is_same_clause() -> None:
    compatibility = (
        evaluate_clause_compatibility(
            make_record(),
            candidate_index=0,
        )
    )

    assert compatibility.is_same_clause


def test_overheat_is_different_clause() -> None:
    compatibility = (
        evaluate_clause_compatibility(
            make_record(),
            candidate_index=7,
        )
    )

    assert not compatibility.is_same_clause
    assert (
        compatibility.candidate_clause_anchor
        == 7
    )
