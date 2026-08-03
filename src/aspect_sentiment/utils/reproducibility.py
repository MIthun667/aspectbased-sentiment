from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class ReproducibilityState:
    seed: int
    deterministic_algorithms: bool
    cuda_available: bool
    cuda_device_count: int
    torch_version: str
    numpy_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_everything(
    seed: int,
    *,
    deterministic_algorithms: bool = True,
) -> ReproducibilityState:
    """
    Seed Python, NumPy, and PyTorch for reproducible experiments.

    Deterministic PyTorch algorithms are enabled in warning mode so that
    unsupported operations are reported without silently changing the
    experiment configuration.
    """
    if seed < 0:
        raise ValueError(
            f"seed must be non-negative, received {seed}"
        )

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        deterministic_algorithms,
        warn_only=True,
    )

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = (
            deterministic_algorithms
        )
        torch.backends.cudnn.benchmark = False

    return ReproducibilityState(
        seed=seed,
        deterministic_algorithms=(
            deterministic_algorithms
        ),
        cuda_available=torch.cuda.is_available(),
        cuda_device_count=torch.cuda.device_count(),
        torch_version=torch.__version__,
        numpy_version=np.__version__,
    )
