from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


VALID_REPRESENTATIONS = {
    "sentence",
    "target_marked",
}


def sentence_text(
    record: dict[str, Any],
) -> str:
    tokens = record.get("tokens")

    if not isinstance(tokens, list):
        raise TypeError(
            "record tokens must be a list"
        )

    if not all(
        isinstance(token, str)
        for token in tokens
    ):
        raise TypeError(
            "record tokens must contain strings"
        )

    return " ".join(tokens)


def target_marked_text(
    record: dict[str, Any],
) -> str:
    tokens = record.get("tokens")
    aspect_start = record.get("aspect_start")
    aspect_end = record.get("aspect_end")

    if not isinstance(tokens, list):
        raise TypeError(
            "record tokens must be a list"
        )

    if not all(
        isinstance(token, str)
        for token in tokens
    ):
        raise TypeError(
            "record tokens must contain strings"
        )

    if not isinstance(aspect_start, int):
        raise TypeError(
            "aspect_start must be an integer"
        )

    if not isinstance(aspect_end, int):
        raise TypeError(
            "aspect_end must be an integer"
        )

    if not (
        0 <= aspect_start < aspect_end
        <= len(tokens)
    ):
        raise ValueError(
            "Invalid aspect span: "
            f"[{aspect_start}, {aspect_end})"
        )

    marked_tokens = [
        *tokens[:aspect_start],
        "<TARGET>",
        *tokens[aspect_start:aspect_end],
        "</TARGET>",
        *tokens[aspect_end:],
    ]

    return " ".join(marked_tokens)


def records_to_texts(
    records: Iterable[dict[str, Any]],
    *,
    representation: str,
) -> list[str]:
    if representation not in VALID_REPRESENTATIONS:
        raise ValueError(
            "Unsupported representation: "
            f"{representation!r}"
        )

    if representation == "sentence":
        return [
            sentence_text(record)
            for record in records
        ]

    return [
        target_marked_text(record)
        for record in records
    ]


@dataclass(slots=True)
class TfidfLogisticRegressionBaseline:
    representation: str
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression

    @classmethod
    def create(
        cls,
        *,
        representation: str,
        ngram_range: Sequence[int] = (1, 2),
        min_df: int = 2,
        max_features: int | None = 50000,
        sublinear_tf: bool = True,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 2026,
    ) -> "TfidfLogisticRegressionBaseline":
        if representation not in VALID_REPRESENTATIONS:
            raise ValueError(
                "Unsupported representation: "
                f"{representation!r}"
            )

        if len(ngram_range) != 2:
            raise ValueError(
                "ngram_range must contain two integers"
            )

        minimum_ngram = int(ngram_range[0])
        maximum_ngram = int(ngram_range[1])

        if minimum_ngram <= 0:
            raise ValueError(
                "minimum n-gram size must be positive"
            )

        if maximum_ngram < minimum_ngram:
            raise ValueError(
                "maximum n-gram size must be greater than "
                "or equal to minimum n-gram size"
            )

        if min_df <= 0:
            raise ValueError(
                "min_df must be positive"
            )

        if max_features is not None and max_features <= 0:
            raise ValueError(
                "max_features must be positive or None"
            )

        if C <= 0.0:
            raise ValueError(
                "C must be positive"
            )

        if max_iter <= 0:
            raise ValueError(
                "max_iter must be positive"
            )

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(
                minimum_ngram,
                maximum_ngram,
            ),
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            token_pattern=r"(?u)\b\w+\b|</?TARGET>",
        )

        classifier = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )

        return cls(
            representation=representation,
            vectorizer=vectorizer,
            classifier=classifier,
        )

    def fit(
        self,
        records: Sequence[dict[str, Any]],
        labels: Sequence[int],
    ) -> "TfidfLogisticRegressionBaseline":
        if not records:
            raise ValueError(
                "Cannot fit TF-IDF baseline on empty records"
            )

        if len(records) != len(labels):
            raise ValueError(
                "records and labels must have equal length"
            )

        invalid_labels = sorted(
            set(labels).difference({0, 1, 2})
        )

        if invalid_labels:
            raise ValueError(
                f"Invalid labels: {invalid_labels}"
            )

        if len(set(labels)) < 2:
            raise ValueError(
                "Training labels must contain at least "
                "two classes"
            )

        texts = records_to_texts(
            records,
            representation=self.representation,
        )

        features = self.vectorizer.fit_transform(
            texts
        )

        self.classifier.fit(
            features,
            labels,
        )

        return self

    def predict(
        self,
        records: Sequence[dict[str, Any]],
    ) -> np.ndarray:
        texts = records_to_texts(
            records,
            representation=self.representation,
        )

        features = self.vectorizer.transform(
            texts
        )

        return np.asarray(
            self.classifier.predict(features),
            dtype=np.int64,
        )

    def predict_proba(
        self,
        records: Sequence[dict[str, Any]],
    ) -> np.ndarray:
        texts = records_to_texts(
            records,
            representation=self.representation,
        )

        features = self.vectorizer.transform(
            texts
        )

        raw_probabilities = np.asarray(
            self.classifier.predict_proba(features),
            dtype=np.float64,
        )

        probabilities = np.zeros(
            (len(records), 3),
            dtype=np.float64,
        )

        for source_index, label_id in enumerate(
            self.classifier.classes_
        ):
            probabilities[
                :,
                int(label_id),
            ] = raw_probabilities[
                :,
                source_index,
            ]

        return probabilities

    def predict_records(
        self,
        records: Sequence[dict[str, Any]],
    ) -> np.ndarray:
        return self.predict(records)

    def predict_proba_records(
        self,
        records: Sequence[dict[str, Any]],
    ) -> np.ndarray:
        return self.predict_proba(records)

    @property
    def vocabulary_size(self) -> int:
        return len(
            self.vectorizer.vocabulary_
        )
