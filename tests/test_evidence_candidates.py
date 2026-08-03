from __future__ import annotations

import pytest

from src.aspect_sentiment.evidence import (
    generate_evidence_candidates,
    linear_distance_to_span,
)


def make_record() -> dict[str, object]:
    return {
        "tokens": [
            "The",
            "battery",
            "life",
            "is",
            "very",
            "good",
            ".",
        ],
        "pos_tags": [
            "DT",
            "NN",
            "NN",
            "VBZ",
            "RB",
            "JJ",
            ".",
        ],
        "dependency_heads": [
            2,
            2,
            5,
            5,
            5,
            -1,
            5,
        ],
        "dependency_relations": [
            "det",
            "compound",
            "nsubj",
            "cop",
            "advmod",
            "ROOT",
            "punct",
        ],
        "aspect_distances": [
            2,
            0,
            0,
            1,
            2,
            1,
            2,
        ],
        "aspect_start": 1,
        "aspect_end": 3,
    }


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, 1),
        (1, 0),
        (2, 0),
        (3, 1),
        (5, 3),
    ],
)
def test_linear_distance_to_span(
    index: int,
    expected: int,
) -> None:
    assert linear_distance_to_span(
        index,
        span_start=1,
        span_end=3,
    ) == expected


def test_candidate_generation_excludes_aspect() -> None:
    candidates = generate_evidence_candidates(
        make_record(),
        maximum_dependency_distance=3,
    )

    indices = {
        candidate.token_index
        for candidate in candidates
    }

    assert 1 not in indices
    assert 2 not in indices


def test_candidate_generation_includes_good() -> None:
    candidates = generate_evidence_candidates(
        make_record()
    )

    good = next(
        candidate
        for candidate in candidates
        if candidate.token == "good"
    )

    assert good.pos_tag == "JJ"
    assert good.dependency_distance == 1
    assert "opinion_pos" in good.candidate_sources


def test_candidate_generation_marks_intensifier() -> None:
    candidates = generate_evidence_candidates(
        make_record()
    )

    very = next(
        candidate
        for candidate in candidates
        if candidate.token == "very"
    )

    assert very.is_intensifier
    assert "intensifier" in very.candidate_sources


def test_punctuation_is_excluded() -> None:
    candidates = generate_evidence_candidates(
        make_record()
    )

    assert all(
        candidate.token != "."
        for candidate in candidates
    )


def test_invalid_alignment_is_rejected() -> None:
    record = make_record()
    record["pos_tags"] = ["NN"]

    with pytest.raises(
        ValueError,
        match="inconsistent",
    ):
        generate_evidence_candidates(
            record
        )


def test_invalid_radius_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        generate_evidence_candidates(
            make_record(),
            maximum_dependency_distance=-1,
        )
