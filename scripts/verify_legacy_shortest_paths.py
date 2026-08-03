from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.graph import (
    all_pairs_shortest_distances,
    convert_heads_to_zero_based,
)


DEFAULT_PAIRS = {
    "laptops": {
        "train": (
            "Laptops_corenlp/train.json",
            "Laptops_corenlp/train_write.json",
        ),
        "test": (
            "Laptops_corenlp/test.json",
            "Laptops_corenlp/test_write.json",
        ),
    },
    "restaurants": {
        "train": (
            "Restaurants_corenlp/train.json",
            "Restaurants_corenlp/train_write.json",
        ),
        "test": (
            "Restaurants_corenlp/test.json",
            "Restaurants_corenlp/test_write.json",
        ),
    },
    "tweets": {
        "train": (
            "Tweets_corenlp/train.json",
            "Tweets_corenlp/train_write.json",
        ),
        "test": (
            "Tweets_corenlp/test.json",
            "Tweets_corenlp/test_write.json",
        ),
    },
}


@dataclass
class VerificationResult:
    domain: str
    split: str
    base_records: int
    write_records: int
    compared_records: int
    matching_records: int
    mismatching_records: int
    skipped: bool
    status: str
    messages: list[str]


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, list):
        raise TypeError(
            f"{path}: expected a JSON list, "
            f"found {type(value).__name__}"
        )

    return value


def verify_pair(
    *,
    domain: str,
    split: str,
    base_path: Path,
    write_path: Path,
    maximum_distance: int,
) -> VerificationResult:
    base_records = load_json(base_path)
    write_records = load_json(write_path)

    messages: list[str] = []

    if len(base_records) != len(write_records):
        messages.append(
            "Pair is unusable because record counts differ: "
            f"base={len(base_records)}, write={len(write_records)}."
        )

        return VerificationResult(
            domain=domain,
            split=split,
            base_records=len(base_records),
            write_records=len(write_records),
            compared_records=0,
            matching_records=0,
            mismatching_records=0,
            skipped=True,
            status="UNUSABLE",
            messages=messages,
        )

    matching_records = 0
    mismatching_records = 0

    for record_index, (base_record, write_record) in enumerate(
        zip(base_records, write_records)
    ):
        base_without_short = {
            key: value
            for key, value in base_record.items()
            if key != "short"
        }
        write_without_short = {
            key: value
            for key, value in write_record.items()
            if key != "short"
        }

        if base_without_short != write_without_short:
            mismatching_records += 1
            messages.append(
                f"record={record_index}: base/write content mismatch"
            )
            continue

        tokens = base_record["token"]
        raw_heads = base_record["head"]

        dependency_heads, _ = convert_heads_to_zero_based(
            heads=raw_heads,
            number_of_tokens=len(tokens),
        )

        generated = all_pairs_shortest_distances(
            dependency_heads,
            maximum_distance=maximum_distance,
        )

        legacy = write_record.get("short")

        if generated == legacy:
            matching_records += 1
        else:
            mismatching_records += 1

            if len(messages) < 25:
                messages.append(
                    f"record={record_index}: generated matrix "
                    "does not match legacy short matrix"
                )

    status = (
        "PASS"
        if mismatching_records == 0
        else "FAIL"
    )

    return VerificationResult(
        domain=domain,
        split=split,
        base_records=len(base_records),
        write_records=len(write_records),
        compared_records=len(base_records),
        matching_records=matching_records,
        mismatching_records=mismatching_records,
        skipped=False,
        status=status,
        messages=messages,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that dependency heads reproduce legacy shortest-path "
            "matrices."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--maximum-distance",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "audits"
            / "legacy_shortest_path_verification.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    results: list[VerificationResult] = []

    for domain, splits in DEFAULT_PAIRS.items():
        for split, relative_paths in splits.items():
            base_relative, write_relative = relative_paths

            result = verify_pair(
                domain=domain,
                split=split,
                base_path=root / base_relative,
                write_path=root / write_relative,
                maximum_distance=args.maximum_distance,
            )

            results.append(result)

    report = {
        "schema_version": "1.0",
        "maximum_distance": args.maximum_distance,
        "graph_source_policy": (
            "Dependency heads are authoritative. Legacy short matrices "
            "are used only for verification."
        ),
        "results": [
            asdict(result)
            for result in results
        ],
        "summary": {
            "passed_pairs": sum(
                result.status == "PASS"
                for result in results
            ),
            "failed_pairs": sum(
                result.status == "FAIL"
                for result in results
            ),
            "unusable_pairs": sum(
                result.status == "UNUSABLE"
                for result in results
            ),
            "matching_records": sum(
                result.matching_records
                for result in results
            ),
            "mismatching_records": sum(
                result.mismatching_records
                for result in results
            ),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    print("=" * 88)
    print("LEGACY SHORTEST-PATH VERIFICATION")
    print("=" * 88)

    for result in results:
        print(
            f"{result.domain:12s} "
            f"{result.split:5s} "
            f"base={result.base_records:6d} "
            f"write={result.write_records:6d} "
            f"matched={result.matching_records:6d} "
            f"mismatched={result.mismatching_records:6d} "
            f"status={result.status}"
        )

        for message in result.messages:
            print(f"  {message}")

    print("-" * 88)
    print(
        f"Passed pairs:       "
        f"{report['summary']['passed_pairs']}"
    )
    print(
        f"Failed pairs:       "
        f"{report['summary']['failed_pairs']}"
    )
    print(
        f"Unusable pairs:     "
        f"{report['summary']['unusable_pairs']}"
    )
    print(
        f"Matching records:   "
        f"{report['summary']['matching_records']}"
    )
    print(
        f"Mismatching records:"
        f" {report['summary']['mismatching_records']}"
    )
    try:
        displayed_output = args.output.resolve().relative_to(root)
    except ValueError:
        displayed_output = args.output

    print(f"Report:             {displayed_output}")
    print("-" * 88)


if __name__ == "__main__":
    main()
