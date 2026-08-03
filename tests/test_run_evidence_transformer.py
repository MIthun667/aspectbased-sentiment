from __future__ import annotations

from pathlib import Path

import pytest
from transformers import AutoTokenizer

from scripts.run_evidence_transformer import (
    create_evidence_transformer_data_loader,
)
from src.aspect_sentiment.config import (
    load_experiment_config,
)


CONFIG_ROOT = Path(
    "configs/evidence_transformer"
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-base",
        use_fast=True,
        local_files_only=True,
    )


@pytest.mark.parametrize(
    ("filename", "expected_mode"),
    [
        (
            "deberta_cls_laptops.yaml",
            "cls",
        ),
        (
            "deberta_cls_aspect_laptops.yaml",
            "cls_aspect",
        ),
        (
            (
                "deberta_cls_aspect_"
                "evidence_laptops.yaml"
            ),
            "cls_aspect_evidence",
        ),
    ],
)
def test_evidence_configs_load(
    filename: str,
    expected_mode: str,
) -> None:
    config = load_experiment_config(
        CONFIG_ROOT / filename
    )

    assert (
        config.model.name
        == "evidence_deberta_classifier"
    )

    assert (
        config.model.parameters["mode"]
        == expected_mode
    )

    assert (
        config.model.parameters[
            "maximum_length"
        ]
        == 128
    )

    assert (
        config.model.parameters[
            "reject_truncation"
        ]
        is True
    )

    assert (
        config.model.parameters[
            "evidence_root"
        ]
        == "data/derived/evidence"
    )


def test_experiment_names_are_unique() -> None:
    names = {
        load_experiment_config(path)
        .output.experiment_name
        for path in sorted(
            CONFIG_ROOT.glob("*.yaml")
        )
    }

    assert len(names) == 3


def test_loader_rejects_empty_instances(
    tokenizer,
) -> None:
    # The dataset constructor is expected to reject
    # empty evidence-aware datasets before DataLoader
    # construction.
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        create_evidence_transformer_data_loader(
            [],
            tokenizer=tokenizer,
            maximum_length=128,
            batch_size=4,
            shuffle=False,
            seed=2026,
            num_workers=0,
            reject_truncation=True,
        )
