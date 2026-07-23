from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


VALID_POLARITIES = {"positive", "neutral", "negative"}

POLARITY_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID_TO_POLARITY = {
    value: key for key, value in POLARITY_TO_ID.items()
}


@dataclass(slots=True)
class ABSCInstance:
    instance_id: str
    sentence_id: str
    domain: str
    split: str

    tokens: list[str]
    pos_tags: list[str]

    # Canonical zero-based dependency heads.
    # -1 denotes the root.
    dependency_heads: list[int]
    dependency_relations: list[str]

    aspect_text: str
    aspect_tokens: list[str]
    aspect_start: int
    aspect_end: int

    polarity: str
    label_id: int

    root_indices: list[int]
    graph_edges: list[list[Any]]
    aspect_distances: list[int]

    number_of_aspects: int
    is_multi_aspect: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ABSCInstance":
        return cls(**value)
