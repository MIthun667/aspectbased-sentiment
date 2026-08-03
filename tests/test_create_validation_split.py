from __future__ import annotations

from scripts.create_validation_split import (
    assign_split,
    normalize_tokens,
    split_sentence_groups,
)


def make_record(
    instance_id: str,
    sentence_id: str,
    polarity: str,
    text: str | None = None,
) -> dict[str, object]:
    sentence_text = (
        text
        if text is not None
        else sentence_id
    )

    return {
        "instance_id": instance_id,
        "sentence_id": sentence_id,
        "polarity": polarity,
        "split": "train",
        "is_multi_aspect": False,
        "tokens": sentence_text.split(),
    }


def test_assign_split_updates_metadata_without_mutation() -> None:
    source = [
        make_record(
            "instance-1",
            "sentence-1",
            "positive",
        )
    ]

    assigned = assign_split(
        source,
        "validation",
    )

    assert assigned[0]["split"] == "validation"
    assert source[0]["split"] == "train"
    assert (
        assigned[0]["instance_id"]
        == source[0]["instance_id"]
    )
    assert (
        assigned[0]["sentence_id"]
        == source[0]["sentence_id"]
    )


def test_normalize_tokens_is_case_and_space_stable() -> None:
    first = normalize_tokens(
        ["Good", "food", "."]
    )
    second = normalize_tokens(
        ["good", "food", "."]
    )

    assert first == "good food."
    assert first == second


def test_grouped_split_keeps_sentence_instances_together() -> None:
    records = [
        make_record(
            "a0",
            "sentence-a",
            "positive",
            "great battery",
        ),
        make_record(
            "a1",
            "sentence-a",
            "negative",
            "great battery",
        ),
        make_record(
            "b0",
            "sentence-b",
            "positive",
            "good screen",
        ),
        make_record(
            "c0",
            "sentence-c",
            "negative",
            "bad keyboard",
        ),
        make_record(
            "d0",
            "sentence-d",
            "neutral",
            "the laptop exists",
        ),
        make_record(
            "e0",
            "sentence-e",
            "neutral",
            "the computer exists",
        ),
    ]

    train, validation = split_sentence_groups(
        records=records,
        validation_ratio=0.5,
        seed=2026,
    )

    train_ids = {
        record["sentence_id"]
        for record in train
    }
    validation_ids = {
        record["sentence_id"]
        for record in validation
    }

    assert not train_ids.intersection(
        validation_ids
    )


def test_duplicate_sentences_remain_in_same_split() -> None:
    records = [
        make_record(
            "a0",
            "sentence-a",
            "positive",
            "Good food .",
        ),
        make_record(
            "b0",
            "sentence-b",
            "positive",
            "good food .",
        ),
        make_record(
            "c0",
            "sentence-c",
            "negative",
            "bad food .",
        ),
        make_record(
            "d0",
            "sentence-d",
            "negative",
            "terrible service .",
        ),
        make_record(
            "e0",
            "sentence-e",
            "neutral",
            "average location .",
        ),
        make_record(
            "f0",
            "sentence-f",
            "neutral",
            "ordinary place .",
        ),
    ]

    train, validation = split_sentence_groups(
        records=records,
        validation_ratio=0.5,
        seed=2026,
    )

    locations = {}

    for split_name, split_records in (
        ("train", train),
        ("validation", validation),
    ):
        for record in split_records:
            locations[
                record["sentence_id"]
            ] = split_name

    assert (
        locations["sentence-a"]
        == locations["sentence-b"]
    )


def test_grouped_split_has_no_normalized_text_overlap() -> None:
    records = [
        make_record(
            f"instance-{index}",
            f"sentence-{index}",
            ["negative", "neutral", "positive"][
                index % 3
            ],
            f"unique example {index}",
        )
        for index in range(30)
    ]

    records.extend(
        [
            make_record(
                "duplicate-a",
                "duplicate-sentence-a",
                "positive",
                "same repeated sentence",
            ),
            make_record(
                "duplicate-b",
                "duplicate-sentence-b",
                "positive",
                "Same repeated sentence",
            ),
        ]
    )

    train, validation = split_sentence_groups(
        records=records,
        validation_ratio=0.2,
        seed=2026,
    )

    train_texts = {
        normalize_tokens(record["tokens"])
        for record in train
    }
    validation_texts = {
        normalize_tokens(record["tokens"])
        for record in validation
    }

    assert not train_texts.intersection(
        validation_texts
    )


def test_grouped_split_is_deterministic() -> None:
    records = [
        make_record(
            f"instance-{index}",
            f"sentence-{index}",
            ["negative", "neutral", "positive"][
                index % 3
            ],
            f"example sentence {index}",
        )
        for index in range(30)
    ]

    first = split_sentence_groups(
        records=records,
        validation_ratio=0.2,
        seed=2026,
    )
    second = split_sentence_groups(
        records=records,
        validation_ratio=0.2,
        seed=2026,
    )

    assert first == second


def test_official_test_duplicate_is_excluded_from_validation() -> None:
    records = [
        make_record(
            "duplicate-training",
            "sentence-duplicate",
            "neutral",
            "official duplicated text",
        ),
        make_record(
            "positive-a",
            "sentence-positive-a",
            "positive",
            "positive example a",
        ),
        make_record(
            "positive-b",
            "sentence-positive-b",
            "positive",
            "positive example b",
        ),
        make_record(
            "negative-a",
            "sentence-negative-a",
            "negative",
            "negative example a",
        ),
        make_record(
            "negative-b",
            "sentence-negative-b",
            "negative",
            "negative example b",
        ),
        make_record(
            "neutral-a",
            "sentence-neutral-a",
            "neutral",
            "neutral example a",
        ),
        make_record(
            "neutral-b",
            "sentence-neutral-b",
            "neutral",
            "neutral example b",
        ),
    ]

    excluded = {
        normalize_tokens(
            ["official", "duplicated", "text"]
        )
    }

    train, validation = split_sentence_groups(
        records=records,
        validation_ratio=0.5,
        seed=2026,
        excluded_validation_texts=excluded,
    )

    train_ids = {
        record["instance_id"]
        for record in train
    }
    validation_ids = {
        record["instance_id"]
        for record in validation
    }

    assert "duplicate-training" in train_ids
    assert "duplicate-training" not in validation_ids
