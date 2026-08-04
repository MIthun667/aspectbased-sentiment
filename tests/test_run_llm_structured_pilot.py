from __future__ import annotations

from collections import Counter

from scripts.run_llm_structured_pilot import (
    deterministic_stratified_subset,
)
from src.aspect_sentiment.models.llm import (
    LLMABSADataset,
    LLMABSAInstance,
)


def make_instance(
    index: int,
    sentiment: str,
    label_id: int,
) -> LLMABSAInstance:
    return LLMABSAInstance(
        instance_id=(
            f"{sentiment}-{index:02d}"
        ),
        sentence_id=(
            f"sentence-{sentiment}-{index:02d}"
        ),
        domain="laptops",
        split="validation",
        tokens=(
            "the",
            "screen",
            "is",
            "fine",
        ),
        aspect_text="screen",
        aspect_start=1,
        aspect_end=2,
        gold_sentiment=sentiment,
        gold_label_id=label_id,
        number_of_aspects=1,
        is_multi_aspect=False,
    )


def make_dataset() -> LLMABSADataset:
    instances = []

    for sentiment, label_id in (
        ("negative", 0),
        ("neutral", 1),
        ("positive", 2),
    ):
        instances.extend(
            make_instance(
                index,
                sentiment,
                label_id,
            )
            for index in range(10)
        )

    return LLMABSADataset(
        instances
    )


def test_subset_is_stratified() -> None:
    subset = (
        deterministic_stratified_subset(
            make_dataset(),
            instances_per_label=4,
            seed=2026,
        )
    )

    assert len(subset) == 12

    assert Counter(
        instance.gold_sentiment
        for instance in subset
    ) == {
        "negative": 4,
        "neutral": 4,
        "positive": 4,
    }


def test_subset_is_deterministic() -> None:
    dataset = make_dataset()

    first = (
        deterministic_stratified_subset(
            dataset,
            instances_per_label=4,
            seed=2026,
        )
    )

    second = (
        deterministic_stratified_subset(
            dataset,
            instances_per_label=4,
            seed=2026,
        )
    )

    assert [
        instance.instance_id
        for instance in first
    ] == [
        instance.instance_id
        for instance in second
    ]


def test_seed_changes_subset() -> None:
    dataset = make_dataset()

    first = (
        deterministic_stratified_subset(
            dataset,
            instances_per_label=4,
            seed=2026,
        )
    )

    second = (
        deterministic_stratified_subset(
            dataset,
            instances_per_label=4,
            seed=2027,
        )
    )

    assert [
        instance.instance_id
        for instance in first
    ] != [
        instance.instance_id
        for instance in second
    ]


def test_runner_help_works_as_direct_script() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_llm_structured_pilot.py",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert (
        "structured LLM ABSA pilot"
        in result.stdout
    )


def test_complete_split_selection() -> None:
    from scripts.run_llm_structured_pilot import (
        select_evaluation_instances,
    )

    dataset = make_dataset()

    selected, mode = (
        select_evaluation_instances(
            dataset,
            instances_per_label=None,
            seed=2026,
        )
    )

    assert mode == "complete_split"
    assert len(selected) == len(dataset)
    assert [
        instance.instance_id
        for instance in selected
    ] == [
        instance.instance_id
        for instance in dataset
    ]


def test_subset_selection_mode() -> None:
    from scripts.run_llm_structured_pilot import (
        select_evaluation_instances,
    )

    selected, mode = (
        select_evaluation_instances(
            make_dataset(),
            instances_per_label=4,
            seed=2026,
        )
    )

    assert (
        mode
        == "deterministic_stratified_subset"
    )
    assert len(selected) == 12
