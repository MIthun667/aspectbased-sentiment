from __future__ import annotations

import pytest
import torch

from src.aspect_sentiment.models.neural import (
    TargetAwareBiLSTM,
)


def make_model() -> TargetAwareBiLSTM:
    return TargetAwareBiLSTM(
        vocabulary_size=50,
        embedding_dimension=16,
        hidden_dimension=12,
        number_of_layers=1,
        dropout=0.2,
        number_of_classes=3,
        padding_index=0,
    )


def test_forward_shapes() -> None:
    model = make_model()

    input_ids = torch.tensor(
        [
            [2, 5, 3, 8, 0],
            [2, 7, 9, 3, 4],
        ],
        dtype=torch.long,
    )

    attention_mask = (
        input_ids != 0
    )

    output = model(
        input_ids,
        attention_mask,
    )

    assert output.logits.shape == (2, 3)
    assert (
        output.pooled_representation.shape
        == (2, 48)
    )


def test_forward_is_finite() -> None:
    model = make_model()

    input_ids = torch.tensor(
        [
            [2, 5, 3],
            [2, 6, 3],
        ],
        dtype=torch.long,
    )

    attention_mask = torch.ones_like(
        input_ids,
        dtype=torch.bool,
    )

    output = model(
        input_ids,
        attention_mask,
    )

    assert torch.isfinite(
        output.logits
    ).all()

    assert torch.isfinite(
        output.pooled_representation
    ).all()


def test_padding_does_not_change_representation() -> None:
    model = make_model()
    model.eval()

    short_ids = torch.tensor(
        [[2, 5, 3]],
        dtype=torch.long,
    )

    short_mask = torch.tensor(
        [[True, True, True]],
        dtype=torch.bool,
    )

    padded_ids = torch.tensor(
        [[2, 5, 3, 0, 0]],
        dtype=torch.long,
    )

    padded_mask = torch.tensor(
        [[True, True, True, False, False]],
        dtype=torch.bool,
    )

    with torch.no_grad():
        short_output = model(
            short_ids,
            short_mask,
        )

        padded_output = model(
            padded_ids,
            padded_mask,
        )

    assert torch.allclose(
        short_output.pooled_representation,
        padded_output.pooled_representation,
        atol=1e-6,
    )

    assert torch.allclose(
        short_output.logits,
        padded_output.logits,
        atol=1e-6,
    )


def test_backward_pass_produces_gradients() -> None:
    model = make_model()

    input_ids = torch.tensor(
        [
            [2, 5, 3, 0],
            [2, 7, 8, 3],
        ],
        dtype=torch.long,
    )

    attention_mask = (
        input_ids != 0
    )

    labels = torch.tensor(
        [0, 2],
        dtype=torch.long,
    )

    output = model(
        input_ids,
        attention_mask,
    )

    loss = torch.nn.functional.cross_entropy(
        output.logits,
        labels,
    )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert gradients
    assert all(
        gradient is not None
        for gradient in gradients
    )


def test_invalid_attention_mask_dtype_is_rejected() -> None:
    model = make_model()

    input_ids = torch.tensor(
        [[2, 5, 3]],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [[1, 1, 1]],
        dtype=torch.long,
    )

    with pytest.raises(
        TypeError,
        match="bool",
    ):
        model(
            input_ids,
            attention_mask,
        )


def test_empty_sequence_is_rejected() -> None:
    model = make_model()

    input_ids = torch.tensor(
        [[0, 0, 0]],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [[False, False, False]],
        dtype=torch.bool,
    )

    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        model(
            input_ids,
            attention_mask,
        )


@pytest.mark.parametrize(
    (
        "argument",
        "value",
        "message",
    ),
    [
        (
            "vocabulary_size",
            0,
            "vocabulary_size",
        ),
        (
            "embedding_dimension",
            0,
            "embedding_dimension",
        ),
        (
            "hidden_dimension",
            0,
            "hidden_dimension",
        ),
        (
            "number_of_layers",
            0,
            "number_of_layers",
        ),
        (
            "dropout",
            1.0,
            "dropout",
        ),
    ],
)
def test_invalid_constructor_arguments(
    argument: str,
    value: int | float,
    message: str,
) -> None:
    arguments = {
        "vocabulary_size": 50,
        "embedding_dimension": 16,
        "hidden_dimension": 12,
        "number_of_layers": 1,
        "dropout": 0.2,
        "number_of_classes": 3,
        "padding_index": 0,
    }

    arguments[argument] = value

    with pytest.raises(
        ValueError,
        match=message,
    ):
        TargetAwareBiLSTM(**arguments)
