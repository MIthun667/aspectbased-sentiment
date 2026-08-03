from __future__ import annotations

from src.aspect_sentiment.evidence import (
    CandidateAttribution,
    CandidateScore,
    EvidenceCandidate,
    ScoredEvidenceCandidate,
    is_competing_predicate_clause,
    select_clause_aware_weak_evidence_indices,
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


def make_scored_candidate(
    *,
    token_index: int,
    token: str,
    score: float,
) -> ScoredEvidenceCandidate:
    return ScoredEvidenceCandidate(
        candidate=EvidenceCandidate(
            token_index=token_index,
            token=token,
            pos_tag="VB",
            dependency_relation="conj",
            dependency_distance=1,
            linear_distance=1,
            is_aspect_token=False,
            is_negation=False,
            is_intensifier=False,
            candidate_sources=(
                "dependency_radius",
                "opinion_pos",
            ),
        ),
        attribution=CandidateAttribution(
            token_index=token_index,
            current_aspect_index=0,
            current_linear_distance=1,
            nearest_aspect_indices=(0,),
            nearest_distance=1,
            is_current_aspect_nearest=True,
            is_unique_to_current_aspect=True,
            is_tied=False,
            is_closer_to_competing_aspect=False,
        ),
        score=CandidateScore(
            token_index=token_index,
            total_score=score,
            dependency_score=0.5,
            linear_score=0.5,
            pos_score=1.0,
            relation_score=0.5,
            modifier_score=0.0,
            attribution_score=1.0,
        ),
    )


def test_overheat_is_competing_predicate_clause() -> None:
    assert is_competing_predicate_clause(
        make_record(),
        candidate_index=7,
    )


def test_easy_is_not_competing_predicate_clause() -> None:
    assert not is_competing_predicate_clause(
        make_record(),
        candidate_index=0,
    )


def test_clause_aware_selection_excludes_overheat() -> None:
    scored = [
        make_scored_candidate(
            token_index=7,
            token="overheat",
            score=3.5,
        ),
        make_scored_candidate(
            token_index=0,
            token="Easy",
            score=3.0,
        ),
    ]

    selected = (
        select_clause_aware_weak_evidence_indices(
            make_record(),
            scored,
            minimum_score=2.5,
            maximum_tokens=3,
        )
    )

    assert selected == (0,)


def test_clause_filter_can_be_disabled() -> None:
    scored = [
        make_scored_candidate(
            token_index=7,
            token="overheat",
            score=3.5,
        ),
        make_scored_candidate(
            token_index=0,
            token="Easy",
            score=3.0,
        ),
    ]

    selected = (
        select_clause_aware_weak_evidence_indices(
            make_record(),
            scored,
            minimum_score=2.5,
            maximum_tokens=3,
            exclude_competing_predicate_clauses=(
                False
            ),
        )
    )

    assert selected == (0, 7)
