from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizerBase,
)

from src.aspect_sentiment.data import (
    EvidenceAwareInstance,
)
from src.aspect_sentiment.evidence import (
    EvidenceIntervention,
    EvidenceInterventionResult,
    generate_evidence_intervention,
)

from .evidence_dataset import (
    EvidenceTransformerBatch,
    EvidenceTransformerBatchCollator,
    EvidenceTransformerDataset,
    EvidenceTransformerInstance,
)


@dataclass(frozen=True, slots=True)
class CounterfactualEvidenceInstance:
    selected: EvidenceTransformerInstance
    random_same_sentence: (
        EvidenceTransformerInstance
    )
    shuffled_cross_instance: (
        EvidenceTransformerInstance
    )


@dataclass(frozen=True, slots=True)
class CounterfactualEvidenceBatch:
    selected: EvidenceTransformerBatch
    random_same_sentence: (
        EvidenceTransformerBatch
    )
    shuffled_cross_instance: (
        EvidenceTransformerBatch
    )


class CounterfactualEvidenceTransformerDataset(
    Dataset[CounterfactualEvidenceInstance]
):
    """
    Three aligned evidence views for compatibility
    and counterfactual training.

    Positive view:
        selected weak evidence

    Negative views:
        random same-sentence evidence
        shuffled cross-instance evidence
    """

    def __init__(
        self,
        instances: Sequence[
            EvidenceAwareInstance
        ],
        *,
        tokenizer: PreTrainedTokenizerBase,
        maximum_length: int = 128,
        reject_truncation: bool = True,
        seed: int = 2026,
    ) -> None:
        if not instances:
            raise ValueError(
                "Dataset instances must not be empty"
            )

        source_instances = tuple(
            instances
        )

        random_result = (
            generate_evidence_intervention(
                source_instances,
                intervention=(
                    EvidenceIntervention
                    .RANDOM_SAME_SENTENCE
                ),
                seed=seed,
            )
        )

        cross_result = (
            generate_evidence_intervention(
                source_instances,
                intervention=(
                    EvidenceIntervention
                    .SHUFFLED_CROSS_INSTANCE
                ),
                seed=seed,
            )
        )

        self.seed = seed
        self.random_intervention = (
            random_result
        )
        self.cross_intervention = (
            cross_result
        )

        selected_dataset = (
            EvidenceTransformerDataset(
                source_instances,
                tokenizer=tokenizer,
                maximum_length=(
                    maximum_length
                ),
                reject_truncation=(
                    reject_truncation
                ),
            )
        )

        random_dataset = (
            EvidenceTransformerDataset(
                random_result.instances,
                tokenizer=tokenizer,
                maximum_length=(
                    maximum_length
                ),
                reject_truncation=(
                    reject_truncation
                ),
            )
        )

        cross_dataset = (
            EvidenceTransformerDataset(
                cross_result.instances,
                tokenizer=tokenizer,
                maximum_length=(
                    maximum_length
                ),
                reject_truncation=(
                    reject_truncation
                ),
            )
        )

        if not (
            len(selected_dataset)
            == len(random_dataset)
            == len(cross_dataset)
        ):
            raise RuntimeError(
                "Counterfactual evidence datasets "
                "have inconsistent lengths"
            )

        aligned: list[
            CounterfactualEvidenceInstance
        ] = []

        for index in range(
            len(selected_dataset)
        ):
            selected = selected_dataset[index]
            random_view = random_dataset[index]
            cross_view = cross_dataset[index]

            self._validate_alignment(
                selected,
                random_view,
                name=(
                    "random_same_sentence"
                ),
            )

            self._validate_alignment(
                selected,
                cross_view,
                name=(
                    "shuffled_cross_instance"
                ),
            )

            aligned.append(
                CounterfactualEvidenceInstance(
                    selected=selected,
                    random_same_sentence=(
                        random_view
                    ),
                    shuffled_cross_instance=(
                        cross_view
                    ),
                )
            )

        self._instances = tuple(aligned)

    @staticmethod
    def _validate_alignment(
        selected: EvidenceTransformerInstance,
        counterfactual: (
            EvidenceTransformerInstance
        ),
        *,
        name: str,
    ) -> None:
        if (
            counterfactual.instance_id
            != selected.instance_id
        ):
            raise ValueError(
                f"{name} changed instance_id"
            )

        if (
            counterfactual.sentence_id
            != selected.sentence_id
        ):
            raise ValueError(
                f"{name} changed sentence_id"
            )

        if (
            counterfactual.label_id
            != selected.label_id
        ):
            raise ValueError(
                f"{name} changed label_id"
            )

        for field_name in (
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "aspect_subword_mask",
            "sentence_subword_mask",
            "pair_aspect_subword_mask",
        ):
            selected_value = getattr(
                selected,
                field_name,
            )

            counterfactual_value = getattr(
                counterfactual,
                field_name,
            )

            if (
                counterfactual_value
                != selected_value
            ):
                raise ValueError(
                    f"{name} changed "
                    f"{field_name}"
                )

    @property
    def random_intervention_result(
        self,
    ) -> EvidenceInterventionResult:
        return self.random_intervention

    @property
    def cross_intervention_result(
        self,
    ) -> EvidenceInterventionResult:
        return self.cross_intervention

    def __len__(self) -> int:
        return len(self._instances)

    def __getitem__(
        self,
        index: int,
    ) -> CounterfactualEvidenceInstance:
        return self._instances[index]


@dataclass(slots=True)
class CounterfactualEvidenceBatchCollator:
    tokenizer: PreTrainedTokenizerBase
    _evidence_collator: (
        EvidenceTransformerBatchCollator
    ) = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._evidence_collator = (
            EvidenceTransformerBatchCollator(
                tokenizer=self.tokenizer
            )
        )

    @staticmethod
    def _validate_batch_alignment(
        selected: EvidenceTransformerBatch,
        counterfactual: (
            EvidenceTransformerBatch
        ),
        *,
        name: str,
    ) -> None:
        if (
            counterfactual.instance_ids
            != selected.instance_ids
        ):
            raise ValueError(
                f"{name} changed instance order"
            )

        if (
            counterfactual.sentence_ids
            != selected.sentence_ids
        ):
            raise ValueError(
                f"{name} changed sentence order"
            )

        if not torch.equal(
            counterfactual.labels,
            selected.labels,
        ):
            raise ValueError(
                f"{name} changed labels"
            )

        tensor_fields = (
            "input_ids",
            "attention_mask",
            "aspect_subword_mask",
            "sentence_subword_mask",
            "pair_aspect_subword_mask",
        )

        for field_name in tensor_fields:
            selected_value = getattr(
                selected,
                field_name,
            )

            counterfactual_value = getattr(
                counterfactual,
                field_name,
            )

            if not torch.equal(
                counterfactual_value,
                selected_value,
            ):
                raise ValueError(
                    f"{name} changed "
                    f"{field_name}"
                )

        if (
            selected.token_type_ids is None
            and counterfactual.token_type_ids
            is not None
        ):
            raise ValueError(
                f"{name} changed token_type_ids"
            )

        if (
            selected.token_type_ids is not None
            and counterfactual.token_type_ids
            is None
        ):
            raise ValueError(
                f"{name} changed token_type_ids"
            )

        if (
            selected.token_type_ids is not None
            and counterfactual.token_type_ids
            is not None
            and not torch.equal(
                selected.token_type_ids,
                counterfactual.token_type_ids,
            )
        ):
            raise ValueError(
                f"{name} changed token_type_ids"
            )

    def __call__(
        self,
        instances: Sequence[
            CounterfactualEvidenceInstance
        ],
    ) -> CounterfactualEvidenceBatch:
        if not instances:
            raise ValueError(
                "Cannot collate an empty "
                "counterfactual batch"
            )

        selected = self._evidence_collator(
            [
                instance.selected
                for instance in instances
            ]
        )

        random_same_sentence = (
            self._evidence_collator(
                [
                    instance
                    .random_same_sentence
                    for instance in instances
                ]
            )
        )

        shuffled_cross_instance = (
            self._evidence_collator(
                [
                    instance
                    .shuffled_cross_instance
                    for instance in instances
                ]
            )
        )

        self._validate_batch_alignment(
            selected,
            random_same_sentence,
            name="random_same_sentence",
        )

        self._validate_batch_alignment(
            selected,
            shuffled_cross_instance,
            name="shuffled_cross_instance",
        )

        return CounterfactualEvidenceBatch(
            selected=selected,
            random_same_sentence=(
                random_same_sentence
            ),
            shuffled_cross_instance=(
                shuffled_cross_instance
            ),
        )
