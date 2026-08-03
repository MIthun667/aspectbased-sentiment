from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.aspect_sentiment.models.transformer import (
    EvidenceCompatibilityBindingTransformerClassifier,
)


class TinyEncoder(nn.Module):
    def __init__(
        self,
        hidden_dimension: int = 8,
    ) -> None:
        super().__init__()

        self.config = SimpleNamespace(
            hidden_size=hidden_dimension
        )

        self.embedding = nn.Embedding(
            64,
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

        hidden = self.projection(
            self.embedding(input_ids)
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
    *,
    dropout: float = 0.0,
):
    return (
        EvidenceCompatibilityBindingTransformerClassifier(
            backbone=TinyEncoder(),
            hidden_dimension=8,
            number_of_classes=3,
            dropout=dropout,
            compatibility_dimension=4,
        )
    )


def test_compatibility_forward_shapes() -> None:
    model = build_model()

    inputs = make_inputs()

    output = model(
        input_ids=inputs[0],
        attention_mask=inputs[1],
        aspect_subword_mask=inputs[2],
        evidence_subword_mask=inputs[3],
    )

    assert output.logits.shape == (2, 3)

    assert output.context_logits.shape == (
        2,
        3,
    )

    assert output.evidence_logits.shape == (
        2,
        3,
    )

    assert (
        output.compatibility_features.shape
        == (2, 33)
    )

    assert (
        output.compatibility_score.shape
        == (2, 1)
    )

    assert output.gate_values.shape == (
        2,
        1,
    )

    assert (
        output.projected_evidence_representation
        .shape
        == (2, 8)
    )


def test_scalar_gate_is_bounded() -> None:
    model = build_model()

    inputs = make_inputs()

    output = model(
        input_ids=inputs[0],
        attention_mask=inputs[1],
        aspect_subword_mask=inputs[2],
        evidence_subword_mask=inputs[3],
    )

    assert torch.all(
        output.gate_values >= 0.0
    )

    assert torch.all(
        output.gate_values <= 1.0
    )


def test_empty_evidence_forces_zero_gate() -> None:
    model = build_model()

    inputs = make_inputs()

    output = model(
        input_ids=inputs[0],
        attention_mask=inputs[1],
        aspect_subword_mask=inputs[2],
        evidence_subword_mask=inputs[3],
    )

    assert (
        output.evidence_available
        .squeeze(-1)
        .tolist()
        == [1.0, 0.0]
    )

    assert torch.equal(
        output.gate_values[1],
        torch.zeros_like(
            output.gate_values[1]
        ),
    )


def test_empty_evidence_zeroes_evidence_path() -> None:
    model = build_model()

    inputs = make_inputs()

    output = model(
        input_ids=inputs[0],
        attention_mask=inputs[1],
        aspect_subword_mask=inputs[2],
        evidence_subword_mask=inputs[3],
    )

    assert torch.equal(
        output.evidence_representation[1],
        torch.zeros_like(
            output.evidence_representation[1]
        ),
    )

    assert torch.equal(
        output.projected_evidence_representation[
            1
        ],
        torch.zeros_like(
            output
            .projected_evidence_representation[
                1
            ]
        ),
    )

    assert torch.allclose(
        output.fused_representation[1],
        output.context_representation[1],
    )


def test_nonempty_evidence_has_scalar_gate() -> None:
    model = build_model()

    inputs = make_inputs()

    output = model(
        input_ids=inputs[0],
        attention_mask=inputs[1],
        aspect_subword_mask=inputs[2],
        evidence_subword_mask=inputs[3],
    )

    assert output.gate_values[0].item() > (
        0.0
    )

    assert output.gate_values[0].item() < (
        1.0
    )


def test_compatibility_features_are_correct() -> None:
    model = build_model()

    inputs = make_inputs()

    output = model(
        input_ids=inputs[0],
        attention_mask=inputs[1],
        aspect_subword_mask=inputs[2],
        evidence_subword_mask=inputs[3],
    )

    hidden = model.hidden_dimension

    features = (
        output.compatibility_features
    )

    target = features[:, :hidden]

    evidence = features[
        :,
        hidden : hidden * 2,
    ]

    difference = features[
        :,
        hidden * 2 : hidden * 3,
    ]

    product = features[
        :,
        hidden * 3 : hidden * 4,
    ]

    availability = features[:, -1:]

    assert torch.allclose(
        target,
        output.aspect_representation,
    )

    assert torch.allclose(
        evidence,
        output.pooled_evidence_representation,
    )

    assert torch.allclose(
        difference,
        torch.abs(target - evidence),
    )

    assert torch.allclose(
        product,
        target * evidence,
    )

    assert torch.equal(
        availability,
        output.evidence_available,
    )


def test_backward_reaches_compatibility_network() -> None:
    model = build_model()

    inputs = make_inputs()

    output = model(
        input_ids=inputs[0],
        attention_mask=inputs[1],
        aspect_subword_mask=inputs[2],
        evidence_subword_mask=inputs[3],
    )

    loss = (
        output.logits.sum()
        + output.context_logits.sum()
        + output.evidence_logits.sum()
        + output.compatibility_score.sum()
    )

    loss.backward()

    modules = (
        model.context_projection,
        model.evidence_projection,
        model.compatibility_network,
        model.context_classifier,
        model.evidence_classifier,
        model.combined_classifier,
    )

    for module in modules:
        assert any(
            parameter.grad is not None
            for parameter in (
                module.parameters()
            )
        )


def test_compatibility_score_controls_gate() -> None:
    model = build_model()

    inputs = make_inputs()

    output = model(
        input_ids=inputs[0],
        attention_mask=inputs[1],
        aspect_subword_mask=inputs[2],
        evidence_subword_mask=inputs[3],
    )

    expected_gate = torch.sigmoid(
        output.compatibility_score
    ) * output.evidence_available

    assert torch.allclose(
        output.gate_values,
        expected_gate,
    )


def test_token_type_ids_are_forwarded() -> None:
    backbone = TinyEncoder()

    model = (
        EvidenceCompatibilityBindingTransformerClassifier(
            backbone=backbone,
            hidden_dimension=8,
            dropout=0.0,
        )
    )

    inputs = make_inputs()

    token_type_ids = torch.zeros_like(
        inputs[0]
    )

    model(
        input_ids=inputs[0],
        attention_mask=inputs[1],
        aspect_subword_mask=inputs[2],
        evidence_subword_mask=inputs[3],
        token_type_ids=token_type_ids,
    )

    assert backbone.last_token_type_ids is (
        token_type_ids
    )


def test_invalid_compatibility_dimension_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="compatibility_dimension",
    ):
        EvidenceCompatibilityBindingTransformerClassifier(
            backbone=TinyEncoder(),
            hidden_dimension=8,
            compatibility_dimension=0,
        )


def test_missing_evidence_mask_rejected() -> None:
    model = build_model()

    inputs = make_inputs()

    with pytest.raises(
        ValueError,
        match="evidence_subword_mask",
    ):
        model(
            input_ids=inputs[0],
            attention_mask=inputs[1],
            aspect_subword_mask=inputs[2],
            evidence_subword_mask=None,
        )


def test_configuration_property() -> None:
    model = build_model()

    assert model.configuration.hidden_size == 8
