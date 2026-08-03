from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from src.aspect_sentiment.artifacts import (
    ExperimentArtifactWriter,
    PredictionRecord,
    collect_run_metadata,
    utc_now_iso,
)
from src.aspect_sentiment.config import (
    ExperimentConfig,
    load_experiment_config,
)
from src.aspect_sentiment.data import (
    load_canonical_split,
)
from src.aspect_sentiment.evaluation import (
    ID_TO_POLARITY,
    compute_classification_metrics,
    exclude_text_overlaps,
)
from src.aspect_sentiment.models.baselines import (
    MajorityClassBaseline,
    TfidfLogisticRegressionBaseline,
)
from src.aspect_sentiment.utils import (
    print_evaluation_summary,
    print_experiment_summary,
    print_training_complete,
    seed_everything,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible baseline experiment "
            "from a YAML configuration."
        )
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to the experiment YAML file.",
    )

    return parser.parse_args()


def labels_from_records(
    records: list[dict[str, Any]],
) -> list[int]:
    return [
        int(record["label_id"])
        for record in records
    ]


def prediction_records(
    records: list[dict[str, Any]],
    predictions,
    probabilities,
) -> list[PredictionRecord]:
    output: list[PredictionRecord] = []

    for record, prediction, probability in zip(
        records,
        predictions,
        probabilities,
        strict=True,
    ):
        predicted_label_id = int(prediction)

        output.append(
            PredictionRecord(
                instance_id=str(
                    record["instance_id"]
                ),
                sentence_id=str(
                    record["sentence_id"]
                ),
                domain=str(
                    record["domain"]
                ),
                split=str(
                    record["split"]
                ),
                target=str(
                    record["aspect_text"]
                ),
                gold_label=str(
                    record["polarity"]
                ),
                gold_label_id=int(
                    record["label_id"]
                ),
                predicted_label=(
                    ID_TO_POLARITY[
                        predicted_label_id
                    ]
                ),
                predicted_label_id=(
                    predicted_label_id
                ),
                probabilities=tuple(
                    float(value)
                    for value in probability
                ),
            )
        )

    return output


def build_model(
    *,
    config: ExperimentConfig,
    training_records: list[dict[str, Any]],
    training_labels: list[int],
):
    if config.model.name == "majority_class":
        model = MajorityClassBaseline.fit(
            training_labels
        )

        print(
            "Fitted majority label:  "
            f"{ID_TO_POLARITY[model.majority_label_id]}"
        )
        print(
            "Training class counts:  "
            f"negative={model.class_counts[0]}, "
            f"neutral={model.class_counts[1]}, "
            f"positive={model.class_counts[2]}"
        )

        return model

    if (
        config.model.name
        == "tfidf_logistic_regression"
    ):
        parameters = dict(
            config.model.parameters
        )

        representation = str(
            parameters.pop(
                "representation",
                "sentence",
            )
        )

        ngram_range_value = parameters.pop(
            "ngram_range",
            (1, 2),
        )

        if not isinstance(
            ngram_range_value,
            (list, tuple),
        ):
            raise TypeError(
                "model.parameters.ngram_range "
                "must be a sequence"
            )

        ngram_range = tuple(
            int(value)
            for value in ngram_range_value
        )

        max_features_value = parameters.pop(
            "max_features",
            50000,
        )

        max_features = (
            None
            if max_features_value is None
            else int(max_features_value)
        )

        model = (
            TfidfLogisticRegressionBaseline.create(
                representation=representation,
                ngram_range=ngram_range,
                min_df=int(
                    parameters.pop(
                        "min_df",
                        2,
                    )
                ),
                max_features=max_features,
                sublinear_tf=bool(
                    parameters.pop(
                        "sublinear_tf",
                        True,
                    )
                ),
                C=float(
                    parameters.pop(
                        "C",
                        1.0,
                    )
                ),
                max_iter=int(
                    parameters.pop(
                        "max_iter",
                        1000,
                    )
                ),
                random_state=(
                    config.training.seed
                ),
            )
        )

        if parameters:
            raise ValueError(
                "Unsupported TF-IDF model parameters: "
                f"{sorted(parameters)}"
            )

        model.fit(
            training_records,
            training_labels,
        )

        print(
            "Representation:         "
            f"{representation}"
        )
        print(
            "TF-IDF vocabulary:      "
            f"{model.vocabulary_size}"
        )
        print(
            "Training instances:     "
            f"{len(training_records)}"
        )

        return model

    raise ValueError(
        "Unsupported baseline model: "
        f"{config.model.name!r}"
    )


def evaluate_records(
    *,
    config: ExperimentConfig,
    model,
    records: list[dict[str, Any]],
    domain: str,
    split: str,
    writer: ExperimentArtifactWriter,
) -> dict[str, Any]:
    if not records:
        raise ValueError(
            f"Cannot evaluate empty record set: {domain}/{split}"
        )

    labels = labels_from_records(records)

    predictions = model.predict_records(
        records
    )

    probabilities = model.predict_proba_records(
        records
    )

    metrics = compute_classification_metrics(
        labels,
        predictions,
        probabilities=probabilities,
    )

    writer.write_metrics(
        domain=domain,
        split=split,
        metrics=metrics,
    )

    if config.evaluation.save_predictions:
        writer.write_predictions(
            domain=domain,
            split=split,
            predictions=prediction_records(
                records,
                predictions,
                probabilities,
            ),
        )

    print_evaluation_summary(
        domain=domain,
        split=split,
        metrics=metrics,
    )

    return metrics


def evaluate_split(
    *,
    config: ExperimentConfig,
    model,
    domain: str,
    split: str,
    writer: ExperimentArtifactWriter,
) -> dict[str, Any]:
    records = load_canonical_split(
        config.data.processed_root,
        domain=domain,
        split=split,
    )

    return evaluate_records(
        config=config,
        model=model,
        records=records,
        domain=domain,
        split=split,
        writer=writer,
    )

def run_experiment(
    config: ExperimentConfig,
    *,
    command: str | None = None,
) -> Path:
    started_at = utc_now_iso()

    seed_everything(
        config.training.seed
    )

    print_experiment_summary(
        experiment_name=(
            config.output.experiment_name
        ),
        model_name=config.model.name,
        train_domain=config.data.train_domain,
        evaluation_domains=(
            config.data.evaluation_domains
        ),
        seed=config.training.seed,
        selection_metric=(
            config.training.selection_metric
        ),
        output_directory=(
            config.output_directory()
        ),
    )

    writer = ExperimentArtifactWriter(config)
    output_directory = writer.prepare()

    writer.write_resolved_config()

    training_records = load_canonical_split(
        config.data.processed_root,
        domain=config.data.train_domain,
        split=config.data.train_split,
    )

    training_labels = labels_from_records(
        training_records
    )

    model = build_model(
        config=config,
        training_records=training_records,
        training_labels=training_labels,
    )

    print()

    best_metric_value: float | None = None

    try:
        for domain in config.data.evaluation_domains:
            validation_metrics = evaluate_split(
                config=config,
                model=model,
                domain=domain,
                split=config.data.validation_split,
                writer=writer,
            )

            if domain == config.data.train_domain:
                value = validation_metrics[
                    config.training.selection_metric
                ]

                best_metric_value = float(value)

            test_records = load_canonical_split(
                config.data.processed_root,
                domain=domain,
                split=config.data.test_split,
            )

            evaluate_records(
                config=config,
                model=model,
                records=test_records,
                domain=domain,
                split=config.data.test_split,
                writer=writer,
            )

            if (
                config.evaluation
                .duplicate_excluded_sensitivity
            ):
                domain_training_records = (
                    load_canonical_split(
                        config.data.processed_root,
                        domain=domain,
                        split=config.data.train_split,
                    )
                )

                retained_records, excluded_records = (
                    exclude_text_overlaps(
                        test_records,
                        reference_records=(
                            domain_training_records
                        ),
                    )
                )

                print()
                print(
                    "Duplicate sensitivity: "
                    f"excluded={len(excluded_records)}, "
                    f"retained={len(retained_records)}"
                )

                if retained_records:
                    evaluate_records(
                        config=config,
                        model=model,
                        records=retained_records,
                        domain=domain,
                        split=(
                            "test_duplicate_excluded"
                        ),
                        writer=writer,
                    )

        completed_at = utc_now_iso()

        writer.write_run_metadata(
            collect_run_metadata(
                experiment_name=(
                    config.output.experiment_name
                ),
                model_name=config.model.name,
                seed=config.training.seed,
                status="COMPLETE",
                started_at_utc=started_at,
                completed_at_utc=completed_at,
                command=command,
                repository_root=".",
            )
        )

        print_training_complete(
            status="COMPLETE",
            best_epoch=1,
            best_metric_name=(
                config.training.selection_metric
            ),
            best_metric_value=(
                best_metric_value
            ),
            output_directory=output_directory,
        )

        return output_directory

    except Exception:
        writer.write_run_metadata(
            collect_run_metadata(
                experiment_name=(
                    config.output.experiment_name
                ),
                model_name=config.model.name,
                seed=config.training.seed,
                status="FAILED",
                started_at_utc=started_at,
                completed_at_utc=utc_now_iso(),
                command=command,
                repository_root=".",
            )
        )

        raise


def main() -> None:
    arguments = parse_arguments()

    config = load_experiment_config(
        arguments.config
    )

    command = " ".join(sys.argv)

    run_experiment(
        config,
        command=command,
    )


if __name__ == "__main__":
    main()
