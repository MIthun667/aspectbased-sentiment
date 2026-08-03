from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.aspect_sentiment.utils.reproducibility import (
    seed_everything,
)


def generate_values() -> tuple[
    float,
    np.ndarray,
    torch.Tensor,
]:
    return (
        random.random(),
        np.random.rand(4),
        torch.rand(4),
    )


def test_seed_everything_reproduces_random_values() -> None:
    first_state = seed_everything(2026)
    first_values = generate_values()

    second_state = seed_everything(2026)
    second_values = generate_values()

    assert first_state.seed == 2026
    assert second_state.seed == 2026

    assert first_values[0] == pytest.approx(
        second_values[0]
    )
    assert np.array_equal(
        first_values[1],
        second_values[1],
    )
    assert torch.equal(
        first_values[2],
        second_values[2],
    )


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        seed_everything(-1)
