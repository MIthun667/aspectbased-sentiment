from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


VALID_LABEL_IDS = (0, 1, 2)


@dataclass(frozen=True, slots=True)
class MajorityClassBaseline:
    majority_label_id: int
    class_counts: tuple[int, int, int]
    class_probabilities: tuple[float, float, float]

    @classmethod
    def fit(
        cls,
        labels: Iterable[int],
    ) -> "MajorityClassBaseline":
        label_values = list(labels)

        if not label_values:
            raise ValueError(
                "Cannot fit majority baseline on empty labels"
            )

        invalid_labels = sorted(
            set(label_values).difference(
                VALID_LABEL_IDS
            )
        )

        if invalid_labels:
            raise ValueError(
                f"Invalid labels: {invalid_labels}"
            )

        counter = Counter(label_values)

        counts = tuple(
            counter.get(label_id, 0)
            for label_id in VALID_LABEL_IDS
        )

        majority_label_id = max(
            VALID_LABEL_IDS,
            key=lambda label_id: (
                counts[label_id],
                -label_id,
            ),
        )

        total = sum(counts)

        probabilities = tuple(
            count / total
            for count in counts
        )

        return cls(
            majority_label_id=majority_label_id,
            class_counts=counts,
            class_probabilities=probabilities,
        )

    def predict(
        self,
        number_of_instances: int,
    ) -> np.ndarray:
        if number_of_instances < 0:
            raise ValueError(
                "number_of_instances must be non-negative"
            )

        return np.full(
            number_of_instances,
            self.majority_label_id,
            dtype=np.int64,
        )

    def predict_proba(
        self,
        number_of_instances: int,
    ) -> np.ndarray:
        if number_of_instances < 0:
            raise ValueError(
                "number_of_instances must be non-negative"
            )

        return np.tile(
            np.asarray(
                self.class_probabilities,
                dtype=np.float64,
            ),
            (number_of_instances, 1),
        )

    def predict_from_items(
        self,
        items: Sequence[object],
    ) -> np.ndarray:
        return self.predict(len(items))
