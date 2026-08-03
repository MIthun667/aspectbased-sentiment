from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.aspect_sentiment.models.transformer import (
    TransformerAspectClassifier,
)


class FakeTransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        vocabulary_size: int = 32,
        hidden_dimension: int = 8,
        number_of_classes: int = 3,
    ) -> None:
        super().__init__()

        self.config = SimpleNamespace(
            num_labels=number_of_classes
        )

        self.embedding = nn.Embedding(
            vocabulary_size,
            hidden_dimension,
        )

        self.classifier = nn.Linear(
            hidden_dimension,
            number_of_classes,
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

        embedded = self.embedding(
            input_ids
        )

        mask = (
            attention_mask
            .unsqueeze(-1)
            .to(embedded.dtype)
        )

        pooled = (
            (embedded * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1.0)
        )

        logits = self.classifier(
            pooled
        )

        return SimpleNamespace(
            logits=logits,
            hidden_states=None,
        )


def build_model() -> (
    TransformerAspectClassifier
):
    return TransformerAspectClassifier(
        backbone=FakeTransformerBackbone()
    )


def test_forward_returns_expected_shape() -> None:
    model = build_model()

    output = model(
        input_ids=torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 0],
            ],
            dtype=torch.long,
        ),
        attention_mask=torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 0],
            ],
            dtype=torch.long,
        ),
    )

    assert output.logits.shape == (2, 3)
    assert torch.isfinite(
        output.logits
    ).all()


def test_forward_passes_token_type_ids() -> None:
    backbone = FakeTransformerBackbone()

    model = TransformerAspectClassifier(
        backbone=backbone
    )

    token_type_ids = torch.tensor(
        [
            [0, 0, 1],
            [0, 1, 1],
        ],
        dtype=torch.long,
    )

    model(
        input_ids=torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        ),
        attention_mask=torch.ones(
            2,
            3,
            dtype=torch.long,
        ),
        token_type_ids=token_type_ids,
    )

    assert backbone.last_token_type_ids is (
        token_type_ids
    )


def test_backward_produces_gradients() -> None:
    model = build_model()

    output = model(
        input_ids=torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        ),
        attention_mask=torch.ones(
            2,
            3,
            dtype=torch.long,
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


def test_configuration_property() -> None:
    model = build_model()

    assert model.configuration.num_labels == 3


def test_rejects_invalid_input_rank() -> None:
    model = build_model()

    with pytest.raises(
        ValueError,
        match="input_ids",
    ):
        model(
            input_ids=torch.tensor(
                [1, 2, 3]
            ),
            attention_mask=torch.tensor(
                [1, 1, 1]
            ),
        )


def test_rejects_attention_mask_mismatch() -> None:
    model = build_model()

    with pytest.raises(
        ValueError,
        match="attention_mask",
    ):
        model(
            input_ids=torch.ones(
                2,
                3,
                dtype=torch.long,
            ),
            attention_mask=torch.ones(
                2,
                2,
                dtype=torch.long,
            ),
        )


def test_rejects_token_type_mismatch() -> None:
    model = build_model()

    with pytest.raises(
        ValueError,
        match="token_type_ids",
    ):
        model(
            input_ids=torch.ones(
                2,
                3,
                dtype=torch.long,
            ),
            attention_mask=torch.ones(
                2,
                3,
                dtype=torch.long,
            ),
            token_type_ids=torch.ones(
                2,
                2,
                dtype=torch.long,
            ),
        )


def test_rejects_unsupported_pretrained_dtype() -> None:
    with pytest.raises(
        ValueError,
        match="dtype",
    ):
        TransformerAspectClassifier.from_pretrained(
            "unused-model",
            dtype=torch.float64,
            local_files_only=True,
        )
