from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_WIDTH = 80


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)

    return str(value)


def print_rule(
    character: str = "-",
    *,
    width: int = DEFAULT_WIDTH,
) -> None:
    print(character * width)


def print_header(
    title: str,
    *,
    width: int = DEFAULT_WIDTH,
) -> None:
    print_rule("=", width=width)
    print(title)
    print_rule("=", width=width)


def print_section(
    title: str,
    *,
    width: int = DEFAULT_WIDTH,
) -> None:
    print()
    print(title)
    print_rule("-", width=width)


def print_key_values(
    values: Mapping[str, Any],
    *,
    label_width: int = 22,
) -> None:
    for label, value in values.items():
        formatted_label = f"{label}:"
        print(
            f"{formatted_label:<{label_width}}"
            f"{format_value(value)}"
        )


def print_experiment_summary(
    *,
    experiment_name: str,
    model_name: str,
    train_domain: str,
    evaluation_domains: Sequence[str],
    seed: int,
    selection_metric: str,
    output_directory: str | Path,
) -> None:
    print_header(
        f"EXPERIMENT: {experiment_name}"
    )

    print_key_values(
        {
            "Model": model_name,
            "Training domain": train_domain,
            "Evaluation domains": evaluation_domains,
            "Seed": seed,
            "Selection metric": selection_metric,
            "Output directory": output_directory,
        }
    )

    print_rule("-")
    print()


@dataclass(frozen=True, slots=True)
class EpochReport:
    epoch: int
    total_epochs: int
    train_loss: float
    validation_metrics: Mapping[str, float]
    learning_rate: float | None = None


def print_epoch_report(
    report: EpochReport,
) -> None:
    components = [
        (
            f"Epoch {report.epoch:>{len(str(report.total_epochs))}}"
            f"/{report.total_epochs}"
        ),
        f"Train Loss {report.train_loss:.4f}",
    ]

    preferred_order = (
        "macro_f1",
        "accuracy",
        "balanced_accuracy",
        "negative_log_likelihood",
    )

    displayed = set()

    for metric_name in preferred_order:
        if metric_name not in report.validation_metrics:
            continue

        components.append(
            f"Val {metric_display_name(metric_name)} "
            f"{report.validation_metrics[metric_name]:.4f}"
        )
        displayed.add(metric_name)

    for metric_name, value in (
        report.validation_metrics.items()
    ):
        if metric_name in displayed:
            continue

        components.append(
            f"Val {metric_display_name(metric_name)} "
            f"{value:.4f}"
        )

    if report.learning_rate is not None:
        components.append(
            f"LR {report.learning_rate:.2e}"
        )

    print(" | ".join(components))


def metric_display_name(name: str) -> str:
    replacements = {
        "accuracy": "Acc",
        "balanced_accuracy": "Bal-Acc",
        "macro_f1": "Macro-F1",
        "weighted_f1": "Weighted-F1",
        "negative_log_likelihood": "NLL",
        "brier_score": "Brier",
    }

    if name in replacements:
        return replacements[name]

    return name.replace("_", " ").title()


def print_evaluation_summary(
    *,
    domain: str,
    split: str,
    metrics: Mapping[str, Any],
) -> None:
    print_section(
        f"EVALUATION: {domain} / {split}"
    )

    selected_metrics = {
        "Instances": metrics.get(
            "number_of_instances",
            "N/A",
        ),
        "Accuracy": metrics.get(
            "accuracy",
            "N/A",
        ),
        "Balanced accuracy": metrics.get(
            "balanced_accuracy",
            "N/A",
        ),
        "Macro-F1": metrics.get(
            "macro_f1",
            "N/A",
        ),
        "Weighted-F1": metrics.get(
            "weighted_f1",
            "N/A",
        ),
    }

    if "negative_log_likelihood" in metrics:
        selected_metrics["NLL"] = metrics[
            "negative_log_likelihood"
        ]

    if "brier_score" in metrics:
        selected_metrics["Brier score"] = metrics[
            "brier_score"
        ]

    print_key_values(selected_metrics)


def print_training_complete(
    *,
    status: str,
    best_epoch: int | None,
    best_metric_name: str,
    best_metric_value: float | None,
    output_directory: str | Path,
) -> None:
    print()
    print_rule("=")
    print("TRAINING SUMMARY")
    print_rule("=")

    values: dict[str, Any] = {
        "Status": status,
    }

    if best_epoch is not None:
        values["Best epoch"] = best_epoch

    if best_metric_value is not None:
        values[
            f"Best {metric_display_name(best_metric_name)}"
        ] = best_metric_value

    values["Artifacts"] = output_directory

    print_key_values(values)
    print_rule("=")
