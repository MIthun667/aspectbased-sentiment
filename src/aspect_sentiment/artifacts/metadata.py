from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .schema import RunMetadata


def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _run_git_command(
    arguments: list[str],
    *,
    repository_root: Path,
) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None

    value = result.stdout.strip()
    return value or None


def collect_git_metadata(
    repository_root: str | Path = ".",
) -> dict[str, object]:
    root = Path(repository_root).resolve()

    commit = _run_git_command(
        ["rev-parse", "HEAD"],
        repository_root=root,
    )

    branch = _run_git_command(
        ["branch", "--show-current"],
        repository_root=root,
    )

    status = _run_git_command(
        ["status", "--porcelain"],
        repository_root=root,
    )

    dirty = None if commit is None else bool(status)

    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": dirty,
    }


def collect_run_metadata(
    *,
    experiment_name: str,
    model_name: str,
    seed: int,
    status: str,
    started_at_utc: str,
    completed_at_utc: str | None = None,
    command: str | None = None,
    notes: str = "",
    repository_root: str | Path = ".",
) -> RunMetadata:
    git_metadata = collect_git_metadata(
        repository_root
    )

    cuda_device_name = None

    if torch.cuda.is_available():
        cuda_device_name = torch.cuda.get_device_name(0)

    return RunMetadata(
        schema_version="1.0",
        experiment_name=experiment_name,
        model_name=model_name,
        seed=seed,
        status=status,
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        git_commit=git_metadata["git_commit"],
        git_branch=git_metadata["git_branch"],
        git_dirty=git_metadata["git_dirty"],
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_device_count=torch.cuda.device_count(),
        cuda_device_name=cuda_device_name,
        command=command,
        notes=notes,
    )
