from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def get_git_commit_sha() -> str | None:
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        return commit
    except Exception:
        return None


def save_experiment_metadata(
    output_dir: Path,
    experiment_name: str,
    model_name: str,
    domain: str,
    seed: int,
    hyperparameters: dict[str, Any],
    best_checkpoint: str,
    best_validation_metric: float,
    best_epoch: float,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"

    data = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "domain": domain,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "git_commit_sha": get_git_commit_sha(),
        "hyperparameters": hyperparameters,
        "best_checkpoint": str(best_checkpoint),
        "best_validation_metric": float(best_validation_metric),
        "best_epoch": float(best_epoch),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return metadata_path


def load_best_checkpoint_path(run_dir: Path) -> Path:
    trainer_states = list(run_dir.glob("**/trainer_state.json"))
    if not trainer_states:
        raise FileNotFoundError(f"No trainer_state.json found in {run_dir}")

    # Read the latest trainer_state.json
    best_path_str = None
    for state_file in trainer_states:
        with state_file.open("r", encoding="utf-8") as f:
            state = json.load(f)
            if "best_model_checkpoint" in state and state["best_model_checkpoint"]:
                best_path_str = state["best_model_checkpoint"]
                break

    if not best_path_str:
        raise RuntimeError(f"Could not resolve best_model_checkpoint from {run_dir}")

    best_path = Path(best_path_str)
    if not best_path.exists():
        # Fallback to local path relative to run_dir if absolute path moved
        best_path = run_dir / best_path.name

    return best_path
