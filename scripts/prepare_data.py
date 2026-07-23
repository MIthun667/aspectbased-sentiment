from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.reader import read_absc_instances, write_jsonl


DEFAULT_DOMAINS = {
    "laptops": "Laptops_corenlp",
    "restaurants": "Restaurants_corenlp",
    "tweets": "Tweets_corenlp",
}


def print_statistics(stats: dict) -> None:
    print("-" * 80)
    print(f"Domain:                   {stats['domain']}")
    print(f"Split:                    {stats['split']}")
    print(f"Source sentences:         {stats['source_sentences']}")
    print(f"Generated instances:      {stats['generated_instances']}")
    print(f"Multi-aspect sentences:   {stats['multi_aspect_sentences']}")
    print(f"Sentences without aspect: {stats['sentences_without_aspects']}")
    print(f"Invalid sentences:        {stats['invalid_sentences']}")
    print(f"Invalid aspects:          {stats['invalid_aspects']}")
    print(f"Label counts:             {stats['label_counts']}")
    print(
        "Aspect-count distribution: "
        f"{stats['aspect_count_distribution']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize CoreNLP SemEval files for ABSC."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    global_summary: dict[str, dict] = {}

    for short_name, directory_name in DEFAULT_DOMAINS.items():
        global_summary[short_name] = {}

        for split in ("train", "test"):
            source_path = (
                args.root
                / directory_name
                / f"{split}.json"
            )

            output_directory = args.output_root / short_name
            output_path = output_directory / f"{split}.jsonl"
            statistics_path = (
                output_directory / f"{split}_statistics.json"
            )

            print("=" * 80)
            print(f"Reading {source_path}")

            instances, statistics = read_absc_instances(
                source_path,
                domain=short_name,
                split=split,
                strict=args.strict,
            )

            written = write_jsonl(instances, output_path)

            if written != len(instances):
                raise RuntimeError(
                    f"Write-count mismatch for {output_path}: "
                    f"{written} != {len(instances)}"
                )

            statistics_dict = statistics.to_dict()

            statistics_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with statistics_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    statistics_dict,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            global_summary[short_name][split] = statistics_dict

            print_statistics(statistics_dict)
            print(f"Saved instances:          {output_path}")
            print(f"Saved statistics:         {statistics_path}")

    summary_path = args.output_root / "dataset_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            global_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 80)
    print(f"Complete summary saved to {summary_path}")


if __name__ == "__main__":
    main()
