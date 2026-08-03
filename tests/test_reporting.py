from __future__ import annotations

from src.aspect_sentiment.utils.reporting import (
    EpochReport,
    format_value,
    metric_display_name,
    print_epoch_report,
    print_evaluation_summary,
    print_experiment_summary,
    print_training_complete,
)


def test_format_value_formats_floats() -> None:
    assert format_value(0.812345) == "0.8123"


def test_format_value_formats_sequences() -> None:
    assert format_value(
        ("laptops", "restaurants")
    ) == "laptops, restaurants"


def test_metric_display_names() -> None:
    assert metric_display_name(
        "macro_f1"
    ) == "Macro-F1"

    assert metric_display_name(
        "balanced_accuracy"
    ) == "Bal-Acc"


def test_experiment_summary_is_readable(
    capsys,
) -> None:
    print_experiment_summary(
        experiment_name="majority_laptops",
        model_name="majority_class",
        train_domain="laptops",
        evaluation_domains=("laptops",),
        seed=2026,
        selection_metric="macro_f1",
        output_directory=(
            "artifacts/experiments/"
            "majority_laptops/seed_2026"
        ),
    )

    output = capsys.readouterr().out

    assert "EXPERIMENT: majority_laptops" in output
    assert "Model:" in output
    assert "majority_class" in output
    assert "{" not in output
    assert '"model"' not in output


def test_epoch_report_is_single_line(
    capsys,
) -> None:
    print_epoch_report(
        EpochReport(
            epoch=2,
            total_epochs=20,
            train_loss=0.69123,
            validation_metrics={
                "macro_f1": 0.74281,
                "accuracy": 0.80114,
            },
            learning_rate=2e-5,
        )
    )

    output = capsys.readouterr().out.strip()

    assert output.count("\n") == 0
    assert "Epoch  2/20" in output
    assert "Train Loss 0.6912" in output
    assert "Val Macro-F1 0.7428" in output
    assert "Val Acc 0.8011" in output
    assert "LR 2.00e-05" in output


def test_evaluation_summary_is_not_json(
    capsys,
) -> None:
    print_evaluation_summary(
        domain="laptops",
        split="validation",
        metrics={
            "number_of_instances": 354,
            "accuracy": 0.81,
            "balanced_accuracy": 0.77,
            "macro_f1": 0.75,
            "weighted_f1": 0.80,
            "negative_log_likelihood": 0.62,
        },
    )

    output = capsys.readouterr().out

    assert "EVALUATION: laptops / validation" in output
    assert "Macro-F1:" in output
    assert "0.7500" in output
    assert "{" not in output


def test_training_complete_summary(
    capsys,
) -> None:
    print_training_complete(
        status="COMPLETE",
        best_epoch=3,
        best_metric_name="macro_f1",
        best_metric_value=0.7516,
        output_directory="artifacts/run",
    )

    output = capsys.readouterr().out

    assert "TRAINING SUMMARY" in output
    assert "Status:" in output
    assert "COMPLETE" in output
    assert "Best epoch:" in output
    assert "3" in output
    assert "Best Macro-F1:" in output
