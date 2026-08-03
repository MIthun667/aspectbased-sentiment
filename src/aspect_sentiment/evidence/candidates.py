from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


OPINION_POS_PREFIXES = (
    "JJ",
    "RB",
    "VB",
)

NEGATION_TOKENS = frozenset(
    {
        "not",
        "n't",
        "never",
        "no",
        "neither",
        "nor",
        "without",
        "hardly",
        "barely",
    }
)

INTENSIFIER_TOKENS = frozenset(
    {
        "very",
        "really",
        "extremely",
        "too",
        "so",
        "quite",
        "highly",
        "especially",
        "particularly",
        "remarkably",
        "incredibly",
        "absolutely",
        "completely",
        "slightly",
        "somewhat",
    }
)

PREFERRED_RELATIONS = frozenset(
    {
        "amod",
        "acomp",
        "advmod",
        "neg",
        "xcomp",
        "ccomp",
        "conj",
        "cop",
        "nsubj",
        "nsubjpass",
        "dobj",
        "dep",
    }
)

PUNCTUATION_POS = frozenset(
    {
        ".",
        ",",
        ":",
        "``",
        "''",
        "-LRB-",
        "-RRB-",
    }
)


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    token_index: int
    token: str
    pos_tag: str
    dependency_relation: str
    dependency_distance: int
    linear_distance: int
    is_aspect_token: bool
    is_negation: bool
    is_intensifier: bool
    candidate_sources: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "token_index": self.token_index,
            "token": self.token,
            "pos_tag": self.pos_tag,
            "dependency_relation": (
                self.dependency_relation
            ),
            "dependency_distance": (
                self.dependency_distance
            ),
            "linear_distance": self.linear_distance,
            "is_aspect_token": (
                self.is_aspect_token
            ),
            "is_negation": self.is_negation,
            "is_intensifier": (
                self.is_intensifier
            ),
            "candidate_sources": list(
                self.candidate_sources
            ),
        }


def validate_record_alignment(
    record: dict[str, Any],
) -> None:
    required_fields = (
        "tokens",
        "pos_tags",
        "dependency_heads",
        "dependency_relations",
        "aspect_distances",
        "aspect_start",
        "aspect_end",
    )

    for field_name in required_fields:
        if field_name not in record:
            raise KeyError(
                f"Missing required field: {field_name}"
            )

    lengths = {
        len(record["tokens"]),
        len(record["pos_tags"]),
        len(record["dependency_heads"]),
        len(record["dependency_relations"]),
        len(record["aspect_distances"]),
    }

    if len(lengths) != 1:
        raise ValueError(
            "Token-aligned fields have inconsistent lengths"
        )

    number_of_tokens = len(record["tokens"])
    aspect_start = int(record["aspect_start"])
    aspect_end = int(record["aspect_end"])

    if not (
        0
        <= aspect_start
        < aspect_end
        <= number_of_tokens
    ):
        raise ValueError(
            "Invalid aspect span"
        )


def linear_distance_to_span(
    token_index: int,
    *,
    span_start: int,
    span_end: int,
) -> int:
    if span_start <= token_index < span_end:
        return 0

    if token_index < span_start:
        return span_start - token_index

    return token_index - span_end + 1


def candidate_sources_for_token(
    *,
    token: str,
    pos_tag: str,
    dependency_relation: str,
    dependency_distance: int,
    maximum_dependency_distance: int,
) -> tuple[str, ...]:
    normalized_token = token.casefold()
    sources: list[str] = []

    if dependency_distance <= maximum_dependency_distance:
        sources.append(
            "dependency_radius"
        )

    if pos_tag.startswith(
        OPINION_POS_PREFIXES
    ):
        sources.append(
            "opinion_pos"
        )

    if dependency_relation in PREFERRED_RELATIONS:
        sources.append(
            "preferred_relation"
        )

    if normalized_token in NEGATION_TOKENS:
        sources.append(
            "negation"
        )

    if normalized_token in INTENSIFIER_TOKENS:
        sources.append(
            "intensifier"
        )

    return tuple(sources)


def generate_evidence_candidates(
    record: dict[str, Any],
    *,
    maximum_dependency_distance: int = 3,
    include_aspect_tokens: bool = False,
) -> tuple[EvidenceCandidate, ...]:
    validate_record_alignment(record)

    if maximum_dependency_distance < 0:
        raise ValueError(
            "maximum_dependency_distance must be non-negative"
        )

    tokens = record["tokens"]
    pos_tags = record["pos_tags"]
    relations = record["dependency_relations"]
    distances = record["aspect_distances"]

    aspect_start = int(record["aspect_start"])
    aspect_end = int(record["aspect_end"])

    candidates: list[EvidenceCandidate] = []

    for index, (
        token,
        pos_tag,
        relation,
        dependency_distance,
    ) in enumerate(
        zip(
            tokens,
            pos_tags,
            relations,
            distances,
            strict=True,
        )
    ):
        is_aspect_token = (
            aspect_start <= index < aspect_end
        )

        if (
            is_aspect_token
            and not include_aspect_tokens
        ):
            continue

        if pos_tag in PUNCTUATION_POS:
            continue

        sources = candidate_sources_for_token(
            token=token,
            pos_tag=pos_tag,
            dependency_relation=relation,
            dependency_distance=int(
                dependency_distance
            ),
            maximum_dependency_distance=(
                maximum_dependency_distance
            ),
        )

        if not sources:
            continue

        normalized_token = token.casefold()

        is_negation = (
            normalized_token
            in NEGATION_TOKENS
        )

        is_intensifier = (
            normalized_token
            in INTENSIFIER_TOKENS
        )

        if not is_candidate_eligible(
            token=token,
            pos_tag=pos_tag,
            dependency_relation=relation,
            is_negation=is_negation,
            is_intensifier=is_intensifier,
        ):
            continue

        candidates.append(
            EvidenceCandidate(
                token_index=index,
                token=token,
                pos_tag=pos_tag,
                dependency_relation=relation,
                dependency_distance=int(
                    dependency_distance
                ),
                linear_distance=(
                    linear_distance_to_span(
                        index,
                        span_start=aspect_start,
                        span_end=aspect_end,
                    )
                ),
                is_aspect_token=is_aspect_token,
                is_negation=is_negation,
                is_intensifier=is_intensifier,
                candidate_sources=sources,
            )
        )

    return tuple(candidates)


@dataclass(frozen=True, slots=True)
class AspectSpan:
    aspect_index: int
    start: int
    end: int
    text: str

    def validate(
        self,
        *,
        number_of_tokens: int,
    ) -> None:
        if self.aspect_index < 0:
            raise ValueError(
                "aspect_index must be non-negative"
            )

        if not (
            0
            <= self.start
            < self.end
            <= number_of_tokens
        ):
            raise ValueError(
                "Invalid competing aspect span"
            )


@dataclass(frozen=True, slots=True)
class CandidateAttribution:
    token_index: int
    current_aspect_index: int
    current_linear_distance: int
    nearest_aspect_indices: tuple[int, ...]
    nearest_distance: int
    is_current_aspect_nearest: bool
    is_unique_to_current_aspect: bool
    is_tied: bool
    is_closer_to_competing_aspect: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "token_index": self.token_index,
            "current_aspect_index": (
                self.current_aspect_index
            ),
            "current_linear_distance": (
                self.current_linear_distance
            ),
            "nearest_aspect_indices": list(
                self.nearest_aspect_indices
            ),
            "nearest_distance": (
                self.nearest_distance
            ),
            "is_current_aspect_nearest": (
                self.is_current_aspect_nearest
            ),
            "is_unique_to_current_aspect": (
                self.is_unique_to_current_aspect
            ),
            "is_tied": self.is_tied,
            "is_closer_to_competing_aspect": (
                self.is_closer_to_competing_aspect
            ),
        }


def aspect_spans_from_sentence_records(
    records: Iterable[dict[str, Any]],
) -> tuple[AspectSpan, ...]:
    materialized = list(records)

    if not materialized:
        raise ValueError(
            "Sentence records must not be empty"
        )

    sentence_ids = {
        str(record["sentence_id"])
        for record in materialized
    }

    if len(sentence_ids) != 1:
        raise ValueError(
            "All records must belong to one sentence"
        )

    token_sequences = {
        tuple(record["tokens"])
        for record in materialized
    }

    if len(token_sequences) != 1:
        raise ValueError(
            "Sentence records contain different tokens"
        )

    number_of_tokens = len(
        materialized[0]["tokens"]
    )

    ordered = sorted(
        materialized,
        key=lambda record: (
            int(record["aspect_start"]),
            int(record["aspect_end"]),
            str(record["instance_id"]),
        ),
    )

    spans: list[AspectSpan] = []

    for aspect_index, record in enumerate(ordered):
        span = AspectSpan(
            aspect_index=aspect_index,
            start=int(record["aspect_start"]),
            end=int(record["aspect_end"]),
            text=str(record["aspect_text"]),
        )

        span.validate(
            number_of_tokens=number_of_tokens
        )

        spans.append(span)

    return tuple(spans)


def identify_current_aspect_index(
    record: dict[str, Any],
    *,
    aspect_spans: tuple[AspectSpan, ...],
) -> int:
    matches = [
        span.aspect_index
        for span in aspect_spans
        if (
            span.start
            == int(record["aspect_start"])
            and span.end
            == int(record["aspect_end"])
            and span.text
            == str(record["aspect_text"])
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            "Current aspect could not be uniquely "
            "matched to sentence aspect spans"
        )

    return matches[0]


def attribute_candidate_to_aspects(
    *,
    token_index: int,
    current_aspect_index: int,
    aspect_spans: tuple[AspectSpan, ...],
) -> CandidateAttribution:
    if not aspect_spans:
        raise ValueError(
            "aspect_spans must not be empty"
        )

    aspect_indices = {
        span.aspect_index
        for span in aspect_spans
    }

    if current_aspect_index not in aspect_indices:
        raise ValueError(
            "current_aspect_index is not present"
        )

    distances = {
        span.aspect_index: linear_distance_to_span(
            token_index,
            span_start=span.start,
            span_end=span.end,
        )
        for span in aspect_spans
    }

    nearest_distance = min(
        distances.values()
    )

    nearest_aspect_indices = tuple(
        sorted(
            aspect_index
            for aspect_index, distance
            in distances.items()
            if distance == nearest_distance
        )
    )

    current_distance = distances[
        current_aspect_index
    ]

    current_is_nearest = (
        current_aspect_index
        in nearest_aspect_indices
    )

    tied = (
        current_is_nearest
        and len(nearest_aspect_indices) > 1
    )

    unique_to_current = (
        nearest_aspect_indices
        == (current_aspect_index,)
    )

    return CandidateAttribution(
        token_index=token_index,
        current_aspect_index=(
            current_aspect_index
        ),
        current_linear_distance=(
            current_distance
        ),
        nearest_aspect_indices=(
            nearest_aspect_indices
        ),
        nearest_distance=nearest_distance,
        is_current_aspect_nearest=(
            current_is_nearest
        ),
        is_unique_to_current_aspect=(
            unique_to_current
        ),
        is_tied=tied,
        is_closer_to_competing_aspect=(
            not current_is_nearest
        ),
    )


def attribute_candidates_for_record(
    record: dict[str, Any],
    *,
    sentence_records: Iterable[
        dict[str, Any]
    ],
    maximum_dependency_distance: int = 3,
) -> tuple[
    tuple[
        EvidenceCandidate,
        CandidateAttribution,
    ],
    ...,
]:
    aspect_spans = (
        aspect_spans_from_sentence_records(
            sentence_records
        )
    )

    current_aspect_index = (
        identify_current_aspect_index(
            record,
            aspect_spans=aspect_spans,
        )
    )

    candidates = generate_evidence_candidates(
        record,
        maximum_dependency_distance=(
            maximum_dependency_distance
        ),
    )

    return tuple(
        (
            candidate,
            attribute_candidate_to_aspects(
                token_index=(
                    candidate.token_index
                ),
                current_aspect_index=(
                    current_aspect_index
                ),
                aspect_spans=aspect_spans,
            ),
        )
        for candidate in candidates
    )


@dataclass(frozen=True, slots=True)
class CandidateScore:
    token_index: int
    total_score: float
    dependency_score: float
    linear_score: float
    pos_score: float
    relation_score: float
    modifier_score: float
    attribution_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "token_index": self.token_index,
            "total_score": self.total_score,
            "dependency_score": self.dependency_score,
            "linear_score": self.linear_score,
            "pos_score": self.pos_score,
            "relation_score": self.relation_score,
            "modifier_score": self.modifier_score,
            "attribution_score": self.attribution_score,
        }


@dataclass(frozen=True, slots=True)
class ScoredEvidenceCandidate:
    candidate: EvidenceCandidate
    attribution: CandidateAttribution
    score: CandidateScore

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "attribution": self.attribution.to_dict(),
            "score": self.score.to_dict(),
        }


def score_evidence_candidate(
    candidate: EvidenceCandidate,
    attribution: CandidateAttribution,
) -> CandidateScore:
    dependency_score = 1.0 / (
        1.0 + candidate.dependency_distance
    )

    linear_score = 1.0 / (
        1.0 + candidate.linear_distance
    )

    pos_score = 0.0

    if candidate.pos_tag.startswith("JJ"):
        pos_score = 1.0
    elif candidate.pos_tag.startswith("RB"):
        pos_score = 0.8
    elif candidate.pos_tag.startswith("VB"):
        pos_score = 0.6

    relation_score = (
        0.5
        if candidate.dependency_relation
        in PREFERRED_RELATIONS
        else 0.0
    )

    modifier_score = 0.0

    if candidate.is_negation:
        modifier_score += 0.7

    if candidate.is_intensifier:
        modifier_score += 0.4

    if attribution.is_unique_to_current_aspect:
        attribution_score = 1.0
    elif attribution.is_tied:
        attribution_score = 0.25
    else:
        attribution_score = -1.0

    total_score = (
        1.5 * dependency_score
        + 0.5 * linear_score
        + pos_score
        + relation_score
        + modifier_score
        + attribution_score
    )

    return CandidateScore(
        token_index=candidate.token_index,
        total_score=total_score,
        dependency_score=dependency_score,
        linear_score=linear_score,
        pos_score=pos_score,
        relation_score=relation_score,
        modifier_score=modifier_score,
        attribution_score=attribution_score,
    )


def score_candidates_for_record(
    record: dict[str, Any],
    *,
    sentence_records: Iterable[
        dict[str, Any]
    ],
    maximum_dependency_distance: int = 3,
) -> tuple[ScoredEvidenceCandidate, ...]:
    attributed = attribute_candidates_for_record(
        record,
        sentence_records=sentence_records,
        maximum_dependency_distance=(
            maximum_dependency_distance
        ),
    )

    scored = [
        ScoredEvidenceCandidate(
            candidate=candidate,
            attribution=attribution,
            score=score_evidence_candidate(
                candidate,
                attribution,
            ),
        )
        for candidate, attribution in attributed
    ]

    scored.sort(
        key=lambda item: (
            -item.score.total_score,
            item.candidate.dependency_distance,
            item.candidate.linear_distance,
            item.candidate.token_index,
        )
    )

    return tuple(scored)


def select_weak_evidence_indices(
    scored_candidates: Iterable[
        ScoredEvidenceCandidate
    ],
    *,
    minimum_score: float = 1.5,
    maximum_tokens: int = 5,
    allow_ties: bool = True,
) -> tuple[int, ...]:
    if maximum_tokens <= 0:
        raise ValueError(
            "maximum_tokens must be positive"
        )

    selected: list[int] = []

    for item in scored_candidates:
        if item.score.total_score < minimum_score:
            continue

        if (
            item.attribution.is_tied
            and not allow_ties
        ):
            continue

        if (
            item.attribution
            .is_closer_to_competing_aspect
        ):
            continue

        selected.append(
            item.candidate.token_index
        )

        if len(selected) >= maximum_tokens:
            break

    return tuple(sorted(selected))

EXCLUDED_POS_TAGS = frozenset(
    {
        "CC",
        "DT",
        "EX",
        "IN",
        "PDT",
        "POS",
        "PRP",
        "PRP$",
        "TO",
        "WDT",
        "WP",
        "WP$",
    }
)

AUXILIARY_TOKENS = frozenset(
    {
        "am",
        "are",
        "be",
        "been",
        "being",
        "can",
        "could",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "is",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "was",
        "were",
        "will",
        "would",
        "'m",
        "'re",
        "'s",
    }
)

SELECTED_NOUN_RELATIONS = frozenset(
    {
        "acomp",
        "attr",
        "conj",
        "dobj",
        "nmod",
        "xcomp",
    }
)


def is_candidate_eligible(
    *,
    token: str,
    pos_tag: str,
    dependency_relation: str,
    is_negation: bool,
    is_intensifier: bool,
) -> bool:
    normalized_token = token.casefold()

    if is_negation or is_intensifier:
        return True

    if pos_tag in PUNCTUATION_POS:
        return False

    if pos_tag in EXCLUDED_POS_TAGS:
        return False

    if normalized_token in AUXILIARY_TOKENS:
        return False

    if pos_tag.startswith("JJ"):
        return True

    if pos_tag.startswith("RB"):
        return True

    if pos_tag.startswith("VB"):
        return True

    if (
        pos_tag.startswith("NN")
        and dependency_relation
        in SELECTED_NOUN_RELATIONS
    ):
        return True

    return False



def select_clause_aware_weak_evidence_indices(
    record: dict[str, Any],
    scored_candidates: Iterable[
        ScoredEvidenceCandidate
    ],
    *,
    minimum_score: float = 2.5,
    maximum_tokens: int = 3,
    allow_ties: bool = False,
    exclude_competing_predicate_clauses: bool = True,
) -> tuple[int, ...]:
    from .clauses import (
        is_competing_predicate_clause,
    )

    if maximum_tokens <= 0:
        raise ValueError(
            "maximum_tokens must be positive"
        )

    selected: list[int] = []

    for item in scored_candidates:
        if item.score.total_score < minimum_score:
            continue

        if (
            item.attribution.is_tied
            and not allow_ties
        ):
            continue

        if (
            item.attribution
            .is_closer_to_competing_aspect
        ):
            continue

        token_index = (
            item.candidate.token_index
        )

        if (
            exclude_competing_predicate_clauses
            and is_competing_predicate_clause(
                record,
                candidate_index=token_index,
            )
        ):
            continue

        selected.append(token_index)

        if len(selected) >= maximum_tokens:
            break

    return tuple(sorted(selected))
