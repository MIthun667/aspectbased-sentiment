from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.aspect_sentiment.models.transformer import (
    EvidenceAwareTransformerClassifier,
    masked_mean_pool,
)


class TinyEncoderBackbone(nn.Module):
    def __init__(
        self,
        *,
        vocabulary_size: int = 32,
        hidden_dimension: int = 8,
    ) -> None:
        super().__init__()

        self.config = SimpleNamespace(
            hidden_size=hidden_dimension
        )

        self.embedding = nn.Embedding(
            vocabulary_size,
            hidden_dimension,
        )

        self.projection = nn.Linear(
            hidden_dimension,
            hidden_dimension,
        )

        self.last_token_type_ids = None

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        token_type_ids=None,
    ):
        self.last_token_type_ids = (
            token_type_ids
        )

        hidden = self.embedding(
            input_ids
        )

        hidden = self.projection(
            hidden
        )

        return SimpleNamespace(
            last_hidden_state=hidden
        )


def make_inputs():
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 0],
            [5, 6, 7, 0, 0],
        ],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    aspect_mask = torch.tensor(
        [
            [0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    evidence_mask = torch.tensor(
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    return (
        input_ids,
        attention_mask,
        aspect_mask,
        evidence_mask,
    )


def build_model(
    mode: str,
) -> EvidenceAwareTransformerClassifier:
    return EvidenceAwareTransformerClassifier(
        backbone=TinyEncoderBackbone(),
        hidden_dimension=8,
        number_of_classes=3,
        mode=mode,
        dropout=0.0,
    )


def test_masked_mean_pool() -> None:
    hidden = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ]
    )

    mask = torch.tensor(
        [[False, True, True]]
    )

    pooled = masked_mean_pool(
        hidden,
        mask,
    )

    assert torch.allclose(
        pooled,
        torch.tensor([[4.0, 5.0]]),
    )


def test_masked_mean_pool_empty_mask() -> None:
    hidden = torch.randn(
        2,
        4,
        6,
    )

    mask = torch.zeros(
        2,
        4,
        dtype=torch.bool,
    )

    pooled = masked_mean_pool(
        hidden,
        mask,
    )

    assert pooled.shape == (2, 6)
    assert torch.equal(
        pooled,
        torch.zeros_like(pooled),
    )


@pytest.mark.parametrize(
    "mode",
    [
        "cls",
        "cls_aspect",
        "cls_aspect_evidence",
    ],
)
def test_forward_modes(mode) -> None:
    model = build_model(mode)

    (
        input_ids,
        attention_mask,
        aspect_mask,
        evidence_mask,
    ) = make_inputs()

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        aspect_subword_mask=(
            aspect_mask
            if mode != "cls"
            else None
        ),
        evidence_subword_mask=(
            evidence_mask
            if mode
            == "cls_aspect_evidence"
            else None
        ),
    )

    assert output.logits.shape == (
        2,
        3,
    )

    assert (
        output.cls_representation.shape
        == (2, 8)
    )

    if mode == "cls":
        assert (
            output.aspect_representation
            is None
        )
        assert (
            output.evidence_representation
            is None
        )

    if mode == "cls_aspect":
        assert (
            output.aspect_representation
            is not None
        )
        assert (
            output.evidence_representation
            is None
        )

    if mode == "cls_aspect_evidence":
        assert (
            output.aspect_representation
            is not None
        )
        assert (
            output.evidence_representation
            is not None
        )
        assert (
            output.evidence_available
            is not None
        )


def test_evidence_availability_indicator() -> None:
    model = build_model(
        "cls_aspect_evidence"
    )

    (
        input_ids,
        attention_mask,
        aspect_mask,
        evidence_mask,
    ) = make_inputs()

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        aspect_subword_mask=aspect_mask,
        evidence_subword_mask=(
            evidence_mask
        ),
    )

    assert (
        output.evidence_available
        is not None
    )

    assert (
        output.evidence_available
        .squeeze(-1)
        .tolist()
        == [1.0, 0.0]
    )

    assert (
        output.evidence_representation
        is not None
    )

    assert torch.equal(
        output.evidence_representation[
            1
        ],
        torch.zeros_like(
            output
            .evidence_representation[1]
        ),
    )


def test_classifier_input_dimensions() -> None:
    cls_model = build_model("cls")
    aspect_model = build_model(
        "cls_aspect"
    )
    evidence_model = build_model(
        "cls_aspect_evidence"
    )

    assert (
        cls_model.classifier.in_features
        == 8
    )

    assert (
        aspect_model.classifier.in_features
        == 16
    )

    assert (
        evidence_model.classifier.in_features
        == 25
    )


def test_backward_produces_gradients() -> None:
    model = build_model(
        "cls_aspect_evidence"
    )

    (
        input_ids,
        attention_mask,
        aspect_mask,
        evidence_mask,
    ) = make_inputs()

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        aspect_subword_mask=aspect_mask,
        evidence_subword_mask=(
            evidence_mask
        ),
    )

    loss = nn.CrossEntropyLoss()(
        output.logits,
        torch.tensor(
            [0, 2],
            dtype=torch.long,
        ),
    )

    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    )


def test_token_type_ids_are_forwarded() -> None:
    backbone = TinyEncoderBackbone()

    model = (
        EvidenceAwareTransformerClassifier(
            backbone=backbone,
            hidden_dimension=8,
            mode="cls",
        )
    )

    (
        input_ids,
        attention_mask,
        _,
        _,
    ) = make_inputs()

    token_type_ids = torch.zeros_like(
        input_ids
    )

    model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    assert backbone.last_token_type_ids is (
        token_type_ids
    )


def test_missing_aspect_mask_rejected() -> None:
    model = build_model(
        "cls_aspect"
    )

    input_ids, attention_mask, _, _ = (
        make_inputs()
    )

    with pytest.raises(
        ValueError,
        match="aspect_subword_mask",
    ):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


def test_missing_evidence_mask_rejected() -> None:
    model = build_model(
        "cls_aspect_evidence"
    )

    (
        input_ids,
        attention_mask,
        aspect_mask,
        _,
    ) = make_inputs()

    with pytest.raises(
        ValueError,
        match="evidence_subword_mask",
    ):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aspect_subword_mask=aspect_mask,
        )


def test_invalid_mode_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported evidence mode",
    ):
        EvidenceAwareTransformerClassifier(
            backbone=TinyEncoderBackbone(),
            hidden_dimension=8,
            mode="invalid",
        )


def test_configuration_property() -> None:
    model = build_model("cls")

    assert (
        model.configuration.hidden_size
        == 8
    )


def test_rejects_unsupported_dtype() -> None:
    with pytest.raises(
        ValueError,
        match="dtype",
    ):
        (
            EvidenceAwareTransformerClassifier
            .from_pretrained(
                "unused-model",
                dtype=torch.float64,
                local_files_only=True,
            )
        )
