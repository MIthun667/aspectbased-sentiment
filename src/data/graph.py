from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class DependencyEdge:
    source: int
    target: int
    relation: str
    direction: str

    def to_list(self) -> list[int | str]:
        return [
            self.source,
            self.target,
            self.relation,
            self.direction,
        ]


def convert_heads_to_zero_based(
    heads: list[int],
    number_of_tokens: int,
) -> tuple[list[int], list[int]]:
    """
    Convert Stanford CoreNLP dependency heads to zero-based indices.

    CoreNLP convention:
        head == 0  -> ROOT
        head >= 1  -> one-based governor token index

    Returned convention:
        head == -1 -> ROOT
        head >= 0  -> zero-based governor token index
    """
    if len(heads) != number_of_tokens:
        raise ValueError(
            "Dependency-head count does not match token count: "
            f"{len(heads)} != {number_of_tokens}"
        )

    converted: list[int] = []
    roots: list[int] = []

    for dependent_index, raw_head in enumerate(heads):
        if not isinstance(raw_head, int):
            raise TypeError(
                f"Dependency head at token {dependent_index} is not an integer: "
                f"{raw_head!r}"
            )

        if raw_head == 0:
            converted.append(-1)
            roots.append(dependent_index)
            continue

        zero_based_head = raw_head - 1

        if zero_based_head < 0 or zero_based_head >= number_of_tokens:
            raise ValueError(
                f"Invalid dependency head {raw_head} at token "
                f"{dependent_index}; sentence length={number_of_tokens}"
            )

        converted.append(zero_based_head)

    if not roots:
        raise ValueError("Dependency tree contains no ROOT node.")

    return converted, roots


def build_dependency_edges(
    dependency_heads: list[int],
    dependency_relations: list[str],
    *,
    add_reverse_edges: bool = True,
    add_self_loops: bool = True,
) -> list[DependencyEdge]:
    """
    Build directed dependency edges.

    Forward edge:
        governor -> dependent

    Reverse edge:
        dependent -> governor

    Self-loop:
        token -> token
    """
    if len(dependency_heads) != len(dependency_relations):
        raise ValueError(
            "Dependency-head and dependency-relation lengths differ: "
            f"{len(dependency_heads)} != {len(dependency_relations)}"
        )

    edges: list[DependencyEdge] = []

    for dependent, (governor, relation) in enumerate(
        zip(dependency_heads, dependency_relations)
    ):
        if governor == -1:
            continue

        normalized_relation = str(relation).strip().lower()

        edges.append(
            DependencyEdge(
                source=governor,
                target=dependent,
                relation=normalized_relation,
                direction="forward",
            )
        )

        if add_reverse_edges:
            edges.append(
                DependencyEdge(
                    source=dependent,
                    target=governor,
                    relation=f"{normalized_relation}__reverse",
                    direction="reverse",
                )
            )

    if add_self_loops:
        for token_index in range(len(dependency_heads)):
            edges.append(
                DependencyEdge(
                    source=token_index,
                    target=token_index,
                    relation="self_loop",
                    direction="self",
                )
            )

    return edges


def build_undirected_adjacency(
    number_of_tokens: int,
    dependency_heads: list[int],
) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(number_of_tokens)]

    for dependent, governor in enumerate(dependency_heads):
        if governor == -1:
            continue

        adjacency[dependent].append(governor)
        adjacency[governor].append(dependent)

    return adjacency


def shortest_distances_from_aspect(
    dependency_heads: list[int],
    aspect_indices: Iterable[int],
) -> list[int]:
    """
    Compute shortest undirected dependency distance from every token to the
    closest token in the target aspect span.
    """
    number_of_tokens = len(dependency_heads)
    aspect_nodes = sorted(set(aspect_indices))

    if not aspect_nodes:
        raise ValueError("Aspect span is empty.")

    for node in aspect_nodes:
        if node < 0 or node >= number_of_tokens:
            raise IndexError(
                f"Aspect index {node} outside sentence length "
                f"{number_of_tokens}"
            )

    adjacency = build_undirected_adjacency(
        number_of_tokens=number_of_tokens,
        dependency_heads=dependency_heads,
    )

    unreachable_value = number_of_tokens + 1
    distances = [unreachable_value] * number_of_tokens
    queue: deque[int] = deque()

    for node in aspect_nodes:
        distances[node] = 0
        queue.append(node)

    while queue:
        current = queue.popleft()

        for neighbor in adjacency[current]:
            candidate_distance = distances[current] + 1

            if candidate_distance < distances[neighbor]:
                distances[neighbor] = candidate_distance
                queue.append(neighbor)

    return distances


def graph_depth(
    dependency_heads: list[int],
    root_indices: list[int],
) -> int:
    """
    Return the maximum undirected distance from any root to any token.
    """
    number_of_tokens = len(dependency_heads)
    adjacency = build_undirected_adjacency(
        number_of_tokens,
        dependency_heads,
    )

    distances = [-1] * number_of_tokens
    queue: deque[int] = deque()

    for root in root_indices:
        distances[root] = 0
        queue.append(root)

    while queue:
        node = queue.popleft()

        for neighbor in adjacency[node]:
            if distances[neighbor] == -1:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)

    reachable = [distance for distance in distances if distance >= 0]
    return max(reachable, default=0)
