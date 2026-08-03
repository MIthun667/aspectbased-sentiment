from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


STRONG_CLAUSE_BOUNDARY_RELATIONS = frozenset(
    {
        "ROOT",
        "root",
        "advcl",
        "ccomp",
        "parataxis",
    }
)

PREDICATE_POS_PREFIXES = (
    "VB",
    "JJ",
)

SUBJECT_RELATIONS = frozenset(
    {
        "nsubj",
        "nsubjpass",
        "csubj",
        "csubjpass",
    }
)


@dataclass(frozen=True, slots=True)
class ClauseAssignment:
    token_index: int
    clause_anchor: int
    ancestor_path: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "token_index": self.token_index,
            "clause_anchor": self.clause_anchor,
            "ancestor_path": list(
                self.ancestor_path
            ),
        }


@dataclass(frozen=True, slots=True)
class ClauseCompatibility:
    candidate_index: int
    aspect_clause_anchor: int
    candidate_clause_anchor: int
    is_same_clause: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_index": self.candidate_index,
            "aspect_clause_anchor": (
                self.aspect_clause_anchor
            ),
            "candidate_clause_anchor": (
                self.candidate_clause_anchor
            ),
            "is_same_clause": self.is_same_clause,
        }


def validate_dependency_structure(
    *,
    heads: Iterable[int],
    relations: Iterable[str],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    materialized_heads = tuple(
        int(head)
        for head in heads
    )

    materialized_relations = tuple(
        str(relation)
        for relation in relations
    )

    if len(materialized_heads) != len(
        materialized_relations
    ):
        raise ValueError(
            "Dependency heads and relations "
            "have inconsistent lengths"
        )

    if not materialized_heads:
        raise ValueError(
            "Dependency structure must not be empty"
        )

    number_of_tokens = len(
        materialized_heads
    )

    for index, head in enumerate(
        materialized_heads
    ):
        if head == -1:
            continue

        if not 0 <= head < number_of_tokens:
            raise ValueError(
                f"Invalid dependency head at token {index}"
            )

        if head == index:
            raise ValueError(
                "Self-referential dependency head "
                f"at token {index}"
            )

    return (
        materialized_heads,
        materialized_relations,
    )


def dependency_ancestor_path(
    token_index: int,
    *,
    heads: Iterable[int],
) -> tuple[int, ...]:
    materialized_heads = tuple(
        int(head)
        for head in heads
    )

    if not 0 <= token_index < len(
        materialized_heads
    ):
        raise IndexError(
            "token_index is out of range"
        )

    path = [token_index]
    visited = {token_index}
    current = token_index

    while True:
        head = materialized_heads[current]

        if head == -1:
            break

        if head in visited:
            raise ValueError(
                "Dependency cycle detected"
            )

        visited.add(head)
        path.append(head)
        current = head

    return tuple(path)


def has_subject_dependent(
    token_index: int,
    *,
    heads: tuple[int, ...],
    relations: tuple[str, ...],
) -> bool:
    return any(
        head == token_index
        and relation in SUBJECT_RELATIONS
        for head, relation in zip(
            heads,
            relations,
            strict=True,
        )
    )


def is_predicate_conjunction(
    token_index: int,
    *,
    heads: tuple[int, ...],
    relations: tuple[str, ...],
    pos_tags: tuple[str, ...],
) -> bool:
    if relations[token_index] != "conj":
        return False

    pos_tag = pos_tags[token_index]

    if not pos_tag.startswith(
        PREDICATE_POS_PREFIXES
    ):
        return False

    if pos_tag.startswith("VB"):
        return True

    return has_subject_dependent(
        token_index,
        heads=heads,
        relations=relations,
    )


def find_clause_anchor(
    token_index: int,
    *,
    heads: Iterable[int],
    relations: Iterable[str],
    pos_tags: Iterable[str] | None = None,
) -> ClauseAssignment:
    (
        materialized_heads,
        materialized_relations,
    ) = validate_dependency_structure(
        heads=heads,
        relations=relations,
    )

    if pos_tags is None:
        materialized_pos_tags = tuple(
            ""
            for _ in materialized_heads
        )
    else:
        materialized_pos_tags = tuple(
            str(tag)
            for tag in pos_tags
        )

        if len(materialized_pos_tags) != len(
            materialized_heads
        ):
            raise ValueError(
                "POS tags and dependencies "
                "have inconsistent lengths"
            )

    path = dependency_ancestor_path(
        token_index,
        heads=materialized_heads,
    )

    for ancestor_index in path:
        relation = materialized_relations[
            ancestor_index
        ]

        if relation in (
            STRONG_CLAUSE_BOUNDARY_RELATIONS
        ):
            return ClauseAssignment(
                token_index=token_index,
                clause_anchor=ancestor_index,
                ancestor_path=path,
            )

        if is_predicate_conjunction(
            ancestor_index,
            heads=materialized_heads,
            relations=materialized_relations,
            pos_tags=materialized_pos_tags,
        ):
            return ClauseAssignment(
                token_index=token_index,
                clause_anchor=ancestor_index,
                ancestor_path=path,
            )

        if (
            materialized_heads[
                ancestor_index
            ]
            == -1
        ):
            return ClauseAssignment(
                token_index=token_index,
                clause_anchor=ancestor_index,
                ancestor_path=path,
            )

    raise RuntimeError(
        "No clause anchor could be identified"
    )


def find_aspect_clause_anchor(
    record: dict[str, Any],
) -> int:
    aspect_start = int(
        record["aspect_start"]
    )

    aspect_end = int(
        record["aspect_end"]
    )

    anchors = [
        find_clause_anchor(
            token_index,
            heads=record[
                "dependency_heads"
            ],
            relations=record[
                "dependency_relations"
            ],
            pos_tags=record["pos_tags"],
        ).clause_anchor
        for token_index in range(
            aspect_start,
            aspect_end,
        )
    ]

    if not anchors:
        raise ValueError(
            "Aspect span must not be empty"
        )

    counts = Counter(anchors)
    maximum_count = max(
        counts.values()
    )

    return min(
        anchor
        for anchor, count in counts.items()
        if count == maximum_count
    )


def evaluate_clause_compatibility(
    record: dict[str, Any],
    *,
    candidate_index: int,
) -> ClauseCompatibility:
    aspect_anchor = (
        find_aspect_clause_anchor(record)
    )

    candidate_anchor = find_clause_anchor(
        candidate_index,
        heads=record["dependency_heads"],
        relations=record[
            "dependency_relations"
        ],
        pos_tags=record["pos_tags"],
    ).clause_anchor

    return ClauseCompatibility(
        candidate_index=candidate_index,
        aspect_clause_anchor=aspect_anchor,
        candidate_clause_anchor=(
            candidate_anchor
        ),
        is_same_clause=(
            aspect_anchor == candidate_anchor
        ),
    )


def is_competing_predicate_clause(
    record: dict[str, Any],
    *,
    candidate_index: int,
) -> bool:
    compatibility = (
        evaluate_clause_compatibility(
            record,
            candidate_index=candidate_index,
        )
    )

    if compatibility.is_same_clause:
        return False

    candidate_anchor = (
        compatibility.candidate_clause_anchor
    )

    return is_predicate_conjunction(
        candidate_anchor,
        heads=tuple(
            int(head)
            for head in record[
                "dependency_heads"
            ]
        ),
        relations=tuple(
            str(relation)
            for relation in record[
                "dependency_relations"
            ]
        ),
        pos_tags=tuple(
            str(tag)
            for tag in record["pos_tags"]
        ),
    )
