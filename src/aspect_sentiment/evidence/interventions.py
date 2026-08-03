from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from random import Random
from typing import Sequence

from src.aspect_sentiment.data import (
    EvidenceAwareInstance,
)


class EvidenceIntervention(str, Enum):
    ORIGINAL = "original"
    EMPTY = "empty"
    ASPECT_ONLY = "aspect_only"
    FULL_SENTENCE = "full_sentence"
    RANDOM_SAME_SENTENCE = (
        "random_same_sentence"
    )
    SHUFFLED_CROSS_INSTANCE = (
        "shuffled_cross_instance"
    )


SUPPORTED_EVIDENCE_INTERVENTIONS = frozenset(
    intervention.value
    for intervention in EvidenceIntervention
)


@dataclass(frozen=True, slots=True)
class EvidenceInterventionResult:
    intervention: EvidenceIntervention
    instances: tuple[
        EvidenceAwareInstance,
        ...
    ]
    seed: int | None
    number_changed: int
    number_unchanged: int
    number_empty: int


def _validated_indices(
    instance: EvidenceAwareInstance,
    indices: Sequence[int],
) -> tuple[int, ...]:
    normalized = tuple(
        sorted(set(indices))
    )

    for index in normalized:
        if not isinstance(index, int):
            raise TypeError(
                "Evidence indices must be integers"
            )

        if not (
            0 <= index < len(instance.tokens)
        ):
            raise ValueError(
                "Evidence index out of range for "
                f"{instance.instance_id!r}: {index}"
            )

    return normalized


def replace_instance_evidence(
    instance: EvidenceAwareInstance,
    *,
    selected_indices: Sequence[int],
    intervention: EvidenceIntervention,
) -> EvidenceAwareInstance:
    indices = _validated_indices(
        instance,
        selected_indices,
    )

    tokens = tuple(
        instance.tokens[index]
        for index in indices
    )

    mask = tuple(
        index in indices
        for index in range(
            len(instance.tokens)
        )
    )

    selection_config = dict(
        instance.selection_config
    )

    selection_config[
        "intervention"
    ] = intervention.value

    selection_config[
        "source_evidence_indices"
    ] = list(
        instance.selected_evidence_indices
    )

    return replace(
        instance,
        selected_evidence_indices=indices,
        selected_evidence_tokens=tokens,
        evidence_mask=mask,
        evidence_is_empty=(
            len(indices) == 0
        ),
        candidate_count=len(indices),
        selection_config=selection_config,
    )


def _aspect_indices(
    instance: EvidenceAwareInstance,
) -> tuple[int, ...]:
    return tuple(
        range(
            instance.aspect_start,
            instance.aspect_end,
        )
    )


def _full_sentence_indices(
    instance: EvidenceAwareInstance,
) -> tuple[int, ...]:
    return tuple(
        range(len(instance.tokens))
    )


def _random_same_sentence_indices(
    instance: EvidenceAwareInstance,
    *,
    random_generator: Random,
) -> tuple[int, ...]:
    evidence_count = len(
        instance.selected_evidence_indices
    )

    if evidence_count == 0:
        return ()

    original = set(
        instance.selected_evidence_indices
    )

    candidates = [
        index
        for index in range(
            len(instance.tokens)
        )
        if index not in original
    ]

    if not candidates:
        return (
            instance.selected_evidence_indices
        )

    sample_size = min(
        evidence_count,
        len(candidates),
    )

    return tuple(
        sorted(
            random_generator.sample(
                candidates,
                sample_size,
            )
        )
    )


def _cyclic_derangement(
    number_of_instances: int,
    *,
    random_generator: Random,
) -> tuple[int, ...]:
    if number_of_instances < 2:
        raise ValueError(
            "Cross-instance intervention requires "
            "at least two instances"
        )

    order = list(
        range(number_of_instances)
    )

    random_generator.shuffle(order)

    mapping = [0] * number_of_instances

    for position, source_index in enumerate(
        order
    ):
        donor_index = order[
            (position + 1)
            % number_of_instances
        ]

        mapping[source_index] = donor_index

    if any(
        source == donor
        for source, donor in enumerate(
            mapping
        )
    ):
        raise RuntimeError(
            "Failed to construct cross-instance "
            "derangement"
        )

    return tuple(mapping)


def _mapped_cross_instance_indices(
    recipient: EvidenceAwareInstance,
    donor: EvidenceAwareInstance,
) -> tuple[int, ...]:
    donor_indices = (
        donor.selected_evidence_indices
    )

    if not donor_indices:
        return ()

    donor_length = len(
        donor.tokens
    )

    recipient_length = len(
        recipient.tokens
    )

    if donor_length <= 0:
        raise ValueError(
            "Donor sentence must not be empty"
        )

    if recipient_length <= 0:
        raise ValueError(
            "Recipient sentence must not be empty"
        )

    mapped: list[int] = []

    for donor_index in donor_indices:
        relative_position = (
            donor_index
            / max(donor_length - 1, 1)
        )

        recipient_index = round(
            relative_position
            * max(recipient_length - 1, 0)
        )

        mapped.append(recipient_index)

    return tuple(
        sorted(set(mapped))
    )


def generate_evidence_intervention(
    instances: Sequence[
        EvidenceAwareInstance
    ],
    *,
    intervention: EvidenceIntervention
    | str,
    seed: int = 2026,
) -> EvidenceInterventionResult:
    if not instances:
        raise ValueError(
            "instances must not be empty"
        )

    try:
        resolved = (
            intervention
            if isinstance(
                intervention,
                EvidenceIntervention,
            )
            else EvidenceIntervention(
                intervention
            )
        )
    except ValueError as error:
        raise ValueError(
            "Unsupported evidence intervention: "
            f"{intervention!r}"
        ) from error

    source = tuple(instances)

    random_generator = Random(seed)

    donor_mapping: tuple[int, ...] | None = (
        None
    )

    if resolved is (
        EvidenceIntervention
        .SHUFFLED_CROSS_INSTANCE
    ):
        donor_mapping = _cyclic_derangement(
            len(source),
            random_generator=(
                random_generator
            ),
        )

    transformed: list[
        EvidenceAwareInstance
    ] = []

    for index, instance in enumerate(source):
        if resolved is (
            EvidenceIntervention.ORIGINAL
        ):
            transformed_instance = instance

        elif resolved is (
            EvidenceIntervention.EMPTY
        ):
            transformed_instance = (
                replace_instance_evidence(
                    instance,
                    selected_indices=(),
                    intervention=resolved,
                )
            )

        elif resolved is (
            EvidenceIntervention
            .ASPECT_ONLY
        ):
            transformed_instance = (
                replace_instance_evidence(
                    instance,
                    selected_indices=(
                        _aspect_indices(
                            instance
                        )
                    ),
                    intervention=resolved,
                )
            )

        elif resolved is (
            EvidenceIntervention
            .FULL_SENTENCE
        ):
            transformed_instance = (
                replace_instance_evidence(
                    instance,
                    selected_indices=(
                        _full_sentence_indices(
                            instance
                        )
                    ),
                    intervention=resolved,
                )
            )

        elif resolved is (
            EvidenceIntervention
            .RANDOM_SAME_SENTENCE
        ):
            transformed_instance = (
                replace_instance_evidence(
                    instance,
                    selected_indices=(
                        _random_same_sentence_indices(
                            instance,
                            random_generator=(
                                random_generator
                            ),
                        )
                    ),
                    intervention=resolved,
                )
            )

        elif resolved is (
            EvidenceIntervention
            .SHUFFLED_CROSS_INSTANCE
        ):
            assert donor_mapping is not None

            donor = source[
                donor_mapping[index]
            ]

            transformed_instance = (
                replace_instance_evidence(
                    instance,
                    selected_indices=(
                        _mapped_cross_instance_indices(
                            instance,
                            donor,
                        )
                    ),
                    intervention=resolved,
                )
            )

            updated_config = dict(
                transformed_instance
                .selection_config
            )

            updated_config[
                "donor_instance_id"
            ] = donor.instance_id

            transformed_instance = replace(
                transformed_instance,
                selection_config=(
                    updated_config
                ),
            )

        else:
            raise AssertionError(
                "Unhandled intervention"
            )

        transformed.append(
            transformed_instance
        )

    changed = sum(
        transformed_instance
        .selected_evidence_indices
        != original_instance
        .selected_evidence_indices
        for original_instance,
        transformed_instance in zip(
            source,
            transformed,
            strict=True,
        )
    )

    empty = sum(
        instance.evidence_is_empty
        for instance in transformed
    )

    return EvidenceInterventionResult(
        intervention=resolved,
        instances=tuple(transformed),
        seed=(
            None
            if resolved
            in {
                EvidenceIntervention.ORIGINAL,
                EvidenceIntervention.EMPTY,
                EvidenceIntervention.ASPECT_ONLY,
                EvidenceIntervention.FULL_SENTENCE,
            }
            else seed
        ),
        number_changed=changed,
        number_unchanged=(
            len(transformed) - changed
        ),
        number_empty=empty,
    )
