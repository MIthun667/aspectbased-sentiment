from __future__ import annotations

import pytest

from src.aspect_sentiment.evidence import (
    CandidateAttribution,
    EvidenceCandidate,
    score_evidence_candidate,
    select_weak_evidence_indices,
)


def make_candidate(
    *,
    token_index: int,
    token: str = "good",
    pos_tag: str = "JJ",
    dependency_distance: int = 1,
    linear_distance: int = 1,
    relation: str = "amod",
    is_negation: bool = False,
    is_intensifier: bool = False,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        token_index=token_index,
        token=token,
        pos_tag=pos_tag,
        dependency_relation=relation,
        dependency_distance=dependency_distance,
        linear_distance=linear_distance,
        is_aspect_token=False,
        is_negation=is_negation,
        is_intensifier=is_intensifier,
        candidate_sources=(
            "dependency_radius",
            "opinion_pos",
        ),
    )


def make_attribution(
    *,
    token_index: int,
    unique: bool = True,
    tied: bool = False,
    competing: bool = False,
) -> CandidateAttribution:
    return CandidateAttribution(
        token_index=token_index,
        current_aspect_index=0,
        current_linear_distance=1,
        nearest_aspect_indices=(
            (0, 1)
            if tied
            else ((1,) if competing else (0,))
        ),
        nearest_distance=1,
        is_current_aspect_nearest=(
            not competing
        ),
        is_unique_to_current_aspect=unique,
        is_tied=tied,
        is_closer_to_competing_aspect=competing,
    )


def test_unique_candidate_scores_above_competing() -> None:
    candidate = make_candidate(
        token_index=5
    )

    unique_score = score_evidence_candidate(
        candidate,
        make_attribution(
            token_index=5,
            unique=True,
        ),
    )

    competing_score = score_evidence_candidate(
        candidate,
        make_attribution(
            token_index=5,
            unique=False,
            competing=True,
        ),
    )

    assert (
        unique_score.total_score
        > competing_score.total_score
    )


def test_negation_receives_modifier_score() -> None:
    candidate = make_candidate(
        token_index=3,
        token="not",
        pos_tag="RB",
        relation="neg",
        is_negation=True,
    )

    score = score_evidence_candidate(
        candidate,
        make_attribution(
            token_index=3
        ),
    )

    assert score.modifier_score == 0.7


def test_intensifier_receives_modifier_score() -> None:
    candidate = make_candidate(
        token_index=4,
        token="very",
        pos_tag="RB",
        relation="advmod",
        is_intensifier=True,
    )

    score = score_evidence_candidate(
        candidate,
        make_attribution(
            token_index=4
        ),
    )

    assert score.modifier_score == 0.4


def test_competing_candidate_is_not_selected() -> None:
    from src.aspect_sentiment.evidence import (
        ScoredEvidenceCandidate,
    )

    candidate = make_candidate(
        token_index=5
    )

    attribution = make_attribution(
        token_index=5,
        unique=False,
        competing=True,
    )

    scored = ScoredEvidenceCandidate(
        candidate=candidate,
        attribution=attribution,
        score=score_evidence_candidate(
            candidate,
            attribution,
        ),
    )

    selected = select_weak_evidence_indices(
        [scored],
        minimum_score=-10.0,
    )

    assert selected == ()


def test_selection_respects_maximum_tokens() -> None:
    from src.aspect_sentiment.evidence import (
        ScoredEvidenceCandidate,
    )

    scored = []

    for index in range(4):
        candidate = make_candidate(
            token_index=index
        )

        attribution = make_attribution(
            token_index=index
        )

        scored.append(
            ScoredEvidenceCandidate(
                candidate=candidate,
                attribution=attribution,
                score=score_evidence_candidate(
                    candidate,
                    attribution,
                ),
            )
        )

    selected = select_weak_evidence_indices(
        scored,
        minimum_score=0.0,
        maximum_tokens=2,
    )

    assert len(selected) == 2


def test_invalid_maximum_tokens_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        select_weak_evidence_indices(
            [],
            maximum_tokens=0,
        )
