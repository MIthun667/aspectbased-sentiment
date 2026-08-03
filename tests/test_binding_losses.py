from __future__ import annotations

import pytest
import torch

from src.aspect_sentiment.models.transformer import (
    EvidenceBindingLoss,
    EvidenceBindingTransformerOutput,
)


def make_output():
    logits = torch.tensor(
        [
            [2.0, 0.5, 0.1],
            [0.2, 1.5, 0.3],
        ],
        requires_grad=True,
    )

    context_logits = torch.tensor(
        [
            [1.8, 0.6, 0.2],
            [0.1, 1.4, 0.5],
        ],
        requires_grad=True,
    )

    evidence_logits = torch.tensor(
        [
            [1.7, 0.4, 0.3],
            [0.3, 0.8, 0.4],
        ],
        requires_grad=True,
    )

    batch_size = 2
    hidden_dimension = 4
    sequence_length = 3

    return EvidenceBindingTransformerOutput(
        logits=logits,
        context_logits=context_logits,
        evidence_logits=evidence_logits,
        context_representation=torch.zeros(
            batch_size,
            hidden_dimension,
        ),
        evidence_representation=torch.zeros(
            batch_size,
            hidden_dimension,
        ),
        projected_evidence_representation=(
            torch.zeros(
                batch_size,
                hidden_dimension,
            )
        ),
        fused_representation=torch.zeros(
            batch_size,
            hidden_dimension,
        ),
        gate_values=torch.zeros(
            batch_size,
            hidden_dimension,
        ),
        evidence_available=torch.tensor(
            [[1.0], [0.0]]
        ),
        cls_representation=torch.zeros(
            batch_size,
            hidden_dimension,
        ),
        aspect_representation=torch.zeros(
            batch_size,
            hidden_dimension,
        ),
        pooled_evidence_representation=(
            torch.zeros(
                batch_size,
                hidden_dimension,
            )
        ),
        last_hidden_state=torch.zeros(
            batch_size,
            sequence_length,
            hidden_dimension,
        ),
    )


def test_loss_is_finite() -> None:
    objective = EvidenceBindingLoss()

    result = objective(
        make_output(),
        torch.tensor(
            [0, 1],
            dtype=torch.long,
        ),
        evidence_is_empty=torch.tensor(
            [False, True]
        ),
    )

    assert torch.isfinite(result.loss)
    assert result.number_with_evidence == 1


def test_empty_examples_excluded_from_evidence_loss() -> None:
    objective = EvidenceBindingLoss(
        combined_weight=0.0,
        context_weight=0.0,
        evidence_weight=1.0,
        agreement_weight=0.0,
    )

    output = make_output()

    first = objective(
        output,
        torch.tensor([0, 1]),
        evidence_is_empty=torch.tensor(
            [False, True]
        ),
    )

    modified = make_output()

    modified.evidence_logits.data[1] = (
        torch.tensor(
            [50.0, -50.0, -50.0]
        )
    )

    second = objective(
        modified,
        torch.tensor([0, 1]),
        evidence_is_empty=torch.tensor(
            [False, True]
        ),
    )

    assert torch.allclose(
        first.evidence_loss,
        second.evidence_loss,
    )


def test_all_empty_returns_zero_auxiliary_losses() -> None:
    objective = EvidenceBindingLoss()

    result = objective(
        make_output(),
        torch.tensor([0, 1]),
        evidence_is_empty=torch.tensor(
            [True, True]
        ),
    )

    assert result.number_with_evidence == 0

    assert result.evidence_loss.item() == (
        pytest.approx(0.0)
    )

    assert result.agreement_loss.item() == (
        pytest.approx(0.0)
    )


def test_total_matches_weighted_components() -> None:
    objective = EvidenceBindingLoss(
        combined_weight=1.0,
        context_weight=0.2,
        evidence_weight=0.5,
        agreement_weight=0.1,
    )

    result = objective(
        make_output(),
        torch.tensor([0, 1]),
        evidence_is_empty=torch.tensor(
            [False, True]
        ),
    )

    expected = (
        result.combined_loss
        + 0.2 * result.context_loss
        + 0.5 * result.evidence_loss
        + 0.1 * result.agreement_loss
    )

    assert torch.allclose(
        result.loss,
        expected,
    )


def test_backward_reaches_logits() -> None:
    objective = EvidenceBindingLoss(
        context_weight=0.2,
    )

    output = make_output()

    result = objective(
        output,
        torch.tensor([0, 1]),
        evidence_is_empty=torch.tensor(
            [False, True]
        ),
    )

    result.loss.backward()

    assert output.logits.grad is not None
    assert (
        output.context_logits.grad
        is not None
    )
    assert (
        output.evidence_logits.grad
        is not None
    )


def test_negative_weight_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="evidence_weight",
    ):
        EvidenceBindingLoss(
            evidence_weight=-0.1
        )


def test_zero_temperature_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="temperature",
    ):
        EvidenceBindingLoss(
            agreement_temperature=0.0
        )


def test_label_shape_rejected() -> None:
    objective = EvidenceBindingLoss()

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        objective(
            make_output(),
            torch.tensor([[0, 1]]),
            evidence_is_empty=torch.tensor(
                [False, True]
            ),
        )
