from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.aspect_sentiment.models.transformer import (
    CounterfactualBindingLoss,
    masked_gold_probability_margin_loss,
    masked_margin_ranking_loss,
)


def test_ranking_loss_zero_when_margin_satisfied() -> None:
    loss = masked_margin_ranking_loss(
        positive_scores=torch.tensor(
            [[1.0], [0.8]]
        ),
        negative_scores=torch.tensor(
            [[0.2], [0.4]]
        ),
        valid_mask=torch.tensor(
            [True, True]
        ),
        margin=0.2,
    )

    assert loss.item() == pytest.approx(
        0.0
    )


def test_ranking_loss_penalizes_wrong_order() -> None:
    loss = masked_margin_ranking_loss(
        positive_scores=torch.tensor(
            [[0.2]]
        ),
        negative_scores=torch.tensor(
            [[0.5]]
        ),
        valid_mask=torch.tensor(
            [True]
        ),
        margin=0.2,
    )

    assert loss.item() == pytest.approx(
        0.5
    )


def test_ranking_loss_respects_mask() -> None:
    first = masked_margin_ranking_loss(
        positive_scores=torch.tensor(
            [[1.0], [-10.0]]
        ),
        negative_scores=torch.tensor(
            [[0.0], [10.0]]
        ),
        valid_mask=torch.tensor(
            [True, False]
        ),
        margin=0.2,
    )

    second = masked_margin_ranking_loss(
        positive_scores=torch.tensor(
            [[1.0], [100.0]]
        ),
        negative_scores=torch.tensor(
            [[0.0], [-100.0]]
        ),
        valid_mask=torch.tensor(
            [True, False]
        ),
        margin=0.2,
    )

    assert torch.allclose(
        first,
        second,
    )


def test_all_invalid_ranking_returns_zero() -> None:
    positive = torch.tensor(
        [[0.1], [0.2]],
        requires_grad=True,
    )

    negative = torch.tensor(
        [[0.9], [0.8]],
        requires_grad=True,
    )

    loss = masked_margin_ranking_loss(
        positive_scores=positive,
        negative_scores=negative,
        valid_mask=torch.tensor(
            [False, False]
        ),
    )

    assert loss.item() == pytest.approx(
        0.0
    )

    loss.backward()

    assert positive.grad is not None
    assert negative.grad is not None


def test_probability_margin_zero_when_satisfied() -> None:
    selected = torch.tensor(
        [[0.0, 0.0, 4.0]]
    )

    corrupted = torch.tensor(
        [[2.0, 0.0, 0.0]]
    )

    loss = (
        masked_gold_probability_margin_loss(
            selected_logits=selected,
            corrupted_logits=corrupted,
            labels=torch.tensor([2]),
            valid_mask=torch.tensor([True]),
            margin=0.05,
        )
    )

    assert loss.item() == pytest.approx(
        0.0
    )


def test_probability_margin_penalizes_corruption() -> None:
    selected = torch.tensor(
        [[1.0, 0.0, 0.0]]
    )

    corrupted = torch.tensor(
        [[0.0, 0.0, 3.0]]
    )

    loss = (
        masked_gold_probability_margin_loss(
            selected_logits=selected,
            corrupted_logits=corrupted,
            labels=torch.tensor([2]),
            valid_mask=torch.tensor([True]),
            margin=0.05,
        )
    )

    assert loss.item() > 0.0


def test_probability_margin_has_gradients() -> None:
    selected = torch.tensor(
        [[1.0, 0.0, 0.0]],
        requires_grad=True,
    )

    corrupted = torch.tensor(
        [[0.0, 0.0, 3.0]],
        requires_grad=True,
    )

    loss = (
        masked_gold_probability_margin_loss(
            selected_logits=selected,
            corrupted_logits=corrupted,
            labels=torch.tensor([2]),
            valid_mask=torch.tensor([True]),
        )
    )

    loss.backward()

    assert selected.grad is not None
    assert corrupted.grad is not None


def make_output(
    *,
    score: float,
    logits: list[float],
):
    return SimpleNamespace(
        compatibility_score=torch.tensor(
            [[score]],
            requires_grad=True,
        ),
        logits=torch.tensor(
            [logits],
            requires_grad=True,
        ),
    )


def test_counterfactual_objective_combines_losses() -> None:
    objective = CounterfactualBindingLoss(
        ranking_weight=0.2,
        probability_margin_weight=0.3,
        ranking_margin=0.2,
        probability_margin=0.05,
    )

    selected = make_output(
        score=0.1,
        logits=[1.0, 0.0, 0.0],
    )

    corrupted = make_output(
        score=0.5,
        logits=[0.0, 0.0, 3.0],
    )

    result = objective(
        selected_output=selected,
        corrupted_output=corrupted,
        labels=torch.tensor([2]),
        valid_mask=torch.tensor([True]),
    )

    expected = (
        0.2 * result.ranking_loss
        + 0.3
        * result.probability_margin_loss
    )

    assert torch.allclose(
        result.loss,
        expected,
    )

    assert result.number_valid == 1


def test_counterfactual_objective_backward() -> None:
    objective = CounterfactualBindingLoss()

    selected = make_output(
        score=0.1,
        logits=[1.0, 0.0, 0.0],
    )

    corrupted = make_output(
        score=0.5,
        logits=[0.0, 0.0, 3.0],
    )

    result = objective(
        selected_output=selected,
        corrupted_output=corrupted,
        labels=torch.tensor([2]),
        valid_mask=torch.tensor([True]),
    )

    result.loss.backward()

    assert (
        selected.compatibility_score.grad
        is not None
    )

    assert (
        corrupted.compatibility_score.grad
        is not None
    )

    assert selected.logits.grad is not None
    assert corrupted.logits.grad is not None


def test_negative_counterfactual_weight_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="ranking_weight",
    ):
        CounterfactualBindingLoss(
            ranking_weight=-0.1
        )


def test_invalid_score_shape_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="shape",
    ):
        masked_margin_ranking_loss(
            positive_scores=torch.zeros(
                2,
                3,
            ),
            negative_scores=torch.zeros(
                2,
                3,
            ),
            valid_mask=torch.tensor(
                [True, True]
            ),
        )
