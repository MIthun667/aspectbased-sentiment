from __future__ import annotations

from src.aspect_sentiment.data import (
    EvidenceAwareDataset,
)
from src.aspect_sentiment.evidence import (
    EvidenceIntervention,
    generate_evidence_intervention,
    replace_instance_evidence,
)


def canonical_record(
    *,
    instance_id: str,
    tokens: list[str],
    aspect_start: int,
    aspect_end: int,
    label_id: int,
) -> dict[str, object]:
    polarity = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }[label_id]

    return {
        "instance_id": instance_id,
        "sentence_id": (
            f"{instance_id}:sentence"
        ),
        "domain": "laptops",
        "split": "test",
        "tokens": tokens,
        "aspect_text": " ".join(
            tokens[
                aspect_start:aspect_end
            ]
        ),
        "aspect_start": aspect_start,
        "aspect_end": aspect_end,
        "polarity": polarity,
        "label_id": label_id,
    }


def evidence_record(
    canonical: dict[str, object],
    *,
    selected_indices: list[int],
) -> dict[str, object]:
    tokens = canonical["tokens"]

    assert isinstance(tokens, list)

    return {
        "schema_version": 1,
        "instance_id": canonical[
            "instance_id"
        ],
        "sentence_id": canonical[
            "sentence_id"
        ],
        "domain": canonical["domain"],
        "split": canonical["split"],
        "tokens": tokens,
        "aspect_text": canonical[
            "aspect_text"
        ],
        "aspect_start": canonical[
            "aspect_start"
        ],
        "aspect_end": canonical[
            "aspect_end"
        ],
        "polarity": canonical[
            "polarity"
        ],
        "selected_evidence_indices": (
            selected_indices
        ),
        "selected_evidence_tokens": [
            tokens[index]
            for index in selected_indices
        ],
        "evidence_is_empty": (
            len(selected_indices) == 0
        ),
        "candidate_count": len(
            selected_indices
        ),
        "selection_config": {
            "maximum_tokens": 3,
        },
    }


def make_instances():
    records = [
        canonical_record(
            instance_id="one",
            tokens=[
                "The",
                "battery",
                "life",
                "is",
                "excellent",
            ],
            aspect_start=1,
            aspect_end=3,
            label_id=2,
        ),
        canonical_record(
            instance_id="two",
            tokens=[
                "The",
                "screen",
                "is",
                "barely",
                "acceptable",
                "today",
            ],
            aspect_start=1,
            aspect_end=2,
            label_id=1,
        ),
        canonical_record(
            instance_id="three",
            tokens=[
                "This",
                "keyboard",
                "feels",
                "very",
                "poor",
            ],
            aspect_start=1,
            aspect_end=2,
            label_id=0,
        ),
    ]

    evidence = [
        evidence_record(
            records[0],
            selected_indices=[4],
        ),
        evidence_record(
            records[1],
            selected_indices=[3, 4],
        ),
        evidence_record(
            records[2],
            selected_indices=[4],
        ),
    ]

    dataset = EvidenceAwareDataset(
        records,
        evidence,
    )

    return [
        dataset[index]
        for index in range(len(dataset))
    ]


def test_original_preserves_objects() -> None:
    instances = make_instances()

    result = generate_evidence_intervention(
        instances,
        intervention="original",
    )

    assert result.instances == tuple(
        instances
    )
    assert result.number_changed == 0
    assert result.seed is None


def test_empty_removes_all_evidence() -> None:
    result = generate_evidence_intervention(
        make_instances(),
        intervention=(
            EvidenceIntervention.EMPTY
        ),
    )

    assert result.number_empty == 3
    assert result.number_changed == 3

    for instance in result.instances:
        assert (
            instance.selected_evidence_indices
            == ()
        )
        assert (
            instance.selected_evidence_tokens
            == ()
        )
        assert not any(
            instance.evidence_mask
        )
        assert instance.evidence_is_empty


def test_aspect_only_uses_aspect_span() -> None:
    result = generate_evidence_intervention(
        make_instances(),
        intervention="aspect_only",
    )

    assert (
        result.instances[0]
        .selected_evidence_indices
        == (1, 2)
    )

    assert (
        result.instances[1]
        .selected_evidence_indices
        == (1,)
    )


def test_full_sentence_selects_every_token() -> None:
    result = generate_evidence_intervention(
        make_instances(),
        intervention="full_sentence",
    )

    for original, transformed in zip(
        make_instances(),
        result.instances,
        strict=True,
    ):
        assert (
            transformed
            .selected_evidence_indices
            == tuple(
                range(len(original.tokens))
            )
        )


def test_random_same_sentence_is_deterministic() -> None:
    instances = make_instances()

    first = generate_evidence_intervention(
        instances,
        intervention=(
            "random_same_sentence"
        ),
        seed=2026,
    )

    second = generate_evidence_intervention(
        instances,
        intervention=(
            "random_same_sentence"
        ),
        seed=2026,
    )

    assert first.instances == second.instances


def test_random_excludes_original_when_possible() -> None:
    instances = make_instances()

    result = generate_evidence_intervention(
        instances,
        intervention=(
            "random_same_sentence"
        ),
        seed=2026,
    )

    for original, transformed in zip(
        instances,
        result.instances,
        strict=True,
    ):
        assert set(
            transformed
            .selected_evidence_indices
        ).isdisjoint(
            original
            .selected_evidence_indices
        )


def test_cross_instance_is_deterministic() -> None:
    instances = make_instances()

    first = generate_evidence_intervention(
        instances,
        intervention=(
            "shuffled_cross_instance"
        ),
        seed=2027,
    )

    second = generate_evidence_intervention(
        instances,
        intervention=(
            "shuffled_cross_instance"
        ),
        seed=2027,
    )

    assert first.instances == second.instances


def test_cross_instance_uses_other_donor() -> None:
    instances = make_instances()

    result = generate_evidence_intervention(
        instances,
        intervention=(
            "shuffled_cross_instance"
        ),
        seed=2028,
    )

    for original, transformed in zip(
        instances,
        result.instances,
        strict=True,
    ):
        donor_id = (
            transformed.selection_config[
                "donor_instance_id"
            ]
        )

        assert donor_id != (
            original.instance_id
        )


def test_interventions_preserve_core_fields() -> None:
    originals = make_instances()

    for intervention in (
        "empty",
        "aspect_only",
        "full_sentence",
        "random_same_sentence",
        "shuffled_cross_instance",
    ):
        result = generate_evidence_intervention(
            originals,
            intervention=intervention,
            seed=2026,
        )

        for original, transformed in zip(
            originals,
            result.instances,
            strict=True,
        ):
            assert (
                transformed.instance_id
                == original.instance_id
            )
            assert (
                transformed.sentence_id
                == original.sentence_id
            )
            assert (
                transformed.tokens
                == original.tokens
            )
            assert (
                transformed.aspect_start
                == original.aspect_start
            )
            assert (
                transformed.aspect_end
                == original.aspect_end
            )
            assert (
                transformed.label_id
                == original.label_id
            )


def test_replacement_rebuilds_tokens_and_mask() -> None:
    instance = make_instances()[0]

    replaced = replace_instance_evidence(
        instance,
        selected_indices=(0, 3),
        intervention=(
            EvidenceIntervention
            .RANDOM_SAME_SENTENCE
        ),
    )

    assert (
        replaced.selected_evidence_indices
        == (0, 3)
    )

    assert (
        replaced.selected_evidence_tokens
        == ("The", "is")
    )

    assert replaced.evidence_mask == (
        True,
        False,
        False,
        True,
        False,
    )


def test_cross_instance_requires_two_instances() -> None:
    instance = make_instances()[0]

    try:
        generate_evidence_intervention(
            [instance],
            intervention=(
                "shuffled_cross_instance"
            ),
        )
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_invalid_intervention_rejected() -> None:
    try:
        generate_evidence_intervention(
            make_instances(),
            intervention="unsupported",
        )
    except ValueError as error:
        assert "Unsupported" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError"
        )
