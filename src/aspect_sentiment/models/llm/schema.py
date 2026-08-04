from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


VALID_SENTIMENTS = (
    "negative",
    "neutral",
    "positive",
)

SENTIMENT_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID_TO_SENTIMENT = {
    value: key
    for key, value in SENTIMENT_TO_ID.items()
}

SUPPORTED_PROMPT_MODES = (
    "sentiment_only",
    "evidence_then_sentiment",
    "joint_evidence_sentiment_confidence",
)


@dataclass(frozen=True, slots=True)
class StructuredABSAOutput:
    sentiment: str
    evidence_indices: tuple[int, ...] = ()
    confidence: float | None = None

    def validate(
        self,
        *,
        number_of_tokens: int,
        require_evidence: bool = True,
        require_confidence: bool = True,
    ) -> None:
        if self.sentiment not in VALID_SENTIMENTS:
            raise ValueError(
                "sentiment must be one of "
                f"{VALID_SENTIMENTS}, received "
                f"{self.sentiment!r}"
            )

        if not isinstance(
            number_of_tokens,
            int,
        ):
            raise TypeError(
                "number_of_tokens must be an integer"
            )

        if number_of_tokens <= 0:
            raise ValueError(
                "number_of_tokens must be positive"
            )

        if require_evidence and not (
            self.evidence_indices
        ):
            raise ValueError(
                "At least one evidence index is required"
            )

        if len(set(self.evidence_indices)) != len(
            self.evidence_indices
        ):
            raise ValueError(
                "evidence_indices must not contain "
                "duplicates"
            )

        if tuple(
            sorted(self.evidence_indices)
        ) != self.evidence_indices:
            raise ValueError(
                "evidence_indices must be sorted"
            )

        for index in self.evidence_indices:
            if isinstance(index, bool) or not isinstance(
                index,
                int,
            ):
                raise TypeError(
                    "Every evidence index must be "
                    "an integer"
                )

            if not (
                0 <= index < number_of_tokens
            ):
                raise ValueError(
                    "Evidence index is outside the "
                    f"token range: {index}"
                )

        if require_confidence:
            if self.confidence is None:
                raise ValueError(
                    "confidence is required"
                )

        if self.confidence is not None:
            if isinstance(
                self.confidence,
                bool,
            ) or not isinstance(
                self.confidence,
                (int, float),
            ):
                raise TypeError(
                    "confidence must be numeric"
                )

            if not (
                0.0
                <= float(self.confidence)
                <= 1.0
            ):
                raise ValueError(
                    "confidence must be within [0, 1]"
                )

    @property
    def sentiment_id(self) -> int:
        return SENTIMENT_TO_ID[
            self.sentiment
        ]

    def evidence_tokens(
        self,
        tokens: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(
            tokens[index]
            for index in self.evidence_indices
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)

        value["evidence_indices"] = list(
            self.evidence_indices
        )

        if value["confidence"] is not None:
            value["confidence"] = float(
                value["confidence"]
            )

        return value


@dataclass(frozen=True, slots=True)
class LLMABSAInstance:
    instance_id: str
    sentence_id: str
    domain: str
    split: str

    tokens: tuple[str, ...]

    aspect_text: str
    aspect_start: int
    aspect_end: int

    gold_sentiment: str
    gold_label_id: int

    number_of_aspects: int
    is_multi_aspect: bool

    def validate(self) -> None:
        if not self.instance_id:
            raise ValueError(
                "instance_id must not be empty"
            )

        if not self.sentence_id:
            raise ValueError(
                "sentence_id must not be empty"
            )

        if not self.tokens:
            raise ValueError(
                "tokens must not be empty"
            )

        if not (
            0
            <= self.aspect_start
            < self.aspect_end
            <= len(self.tokens)
        ):
            raise ValueError(
                "Invalid aspect span: "
                f"[{self.aspect_start}, "
                f"{self.aspect_end})"
            )

        if self.gold_sentiment not in (
            VALID_SENTIMENTS
        ):
            raise ValueError(
                "Invalid gold sentiment: "
                f"{self.gold_sentiment!r}"
            )

        expected_label = SENTIMENT_TO_ID[
            self.gold_sentiment
        ]

        if self.gold_label_id != expected_label:
            raise ValueError(
                "gold_label_id does not match "
                "gold_sentiment"
            )

    @classmethod
    def from_canonical_record(
        cls,
        record: Mapping[str, Any],
    ) -> "LLMABSAInstance":
        instance = cls(
            instance_id=str(
                record["instance_id"]
            ),
            sentence_id=str(
                record["sentence_id"]
            ),
            domain=str(record["domain"]),
            split=str(record["split"]),
            tokens=tuple(
                str(token)
                for token in record["tokens"]
            ),
            aspect_text=str(
                record["aspect_text"]
            ),
            aspect_start=int(
                record["aspect_start"]
            ),
            aspect_end=int(
                record["aspect_end"]
            ),
            gold_sentiment=str(
                record["polarity"]
            ),
            gold_label_id=int(
                record["label_id"]
            ),
            number_of_aspects=int(
                record.get(
                    "number_of_aspects",
                    1,
                )
            ),
            is_multi_aspect=bool(
                record.get(
                    "is_multi_aspect",
                    False,
                )
            ),
        )

        instance.validate()

        return instance
