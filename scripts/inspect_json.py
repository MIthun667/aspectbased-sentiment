from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def describe_value(value: Any, depth: int = 0, max_depth: int = 4) -> Any:
    if depth >= max_depth:
        return type(value).__name__

    if isinstance(value, dict):
        return {
            key: describe_value(item, depth + 1, max_depth)
            for key, item in list(value.items())[:20]
        }

    if isinstance(value, list):
        if not value:
            return []

        sample = value[:3]
        return [
            describe_value(item, depth + 1, max_depth)
            for item in sample
        ]

    return {
        "type": type(value).__name__,
        "example": value,
    }


def collect_top_level_keys(data: Any) -> Counter[str]:
    counter: Counter[str] = Counter()

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                counter.update(item.keys())

    elif isinstance(data, dict):
        counter.update(data.keys())

        for value in data.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        counter.update(f"nested::{key}" for key in item.keys())

    return counter


def inspect_file(path: Path) -> None:
    print("=" * 100)
    print(f"FILE: {path}")
    print("=" * 100)

    if not path.exists():
        print("ERROR: File does not exist")
        return

    print(f"Size: {path.stat().st_size / 1024:.2f} KB")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Top-level type: {type(data).__name__}")

    if isinstance(data, list):
        print(f"Number of top-level items: {len(data)}")
        sample = data[0] if data else None

    elif isinstance(data, dict):
        print(f"Top-level keys: {list(data.keys())}")
        sample = data

        for key, value in data.items():
            if isinstance(value, list):
                print(f"List field '{key}': {len(value)} items")
    else:
        sample = data

    print("\nKey frequencies:")
    for key, count in collect_top_level_keys(data).most_common():
        print(f"  {key}: {count}")

    print("\nFirst-item structure:")
    print(json.dumps(describe_value(sample), indent=2, ensure_ascii=False))

    print("\nRaw first item:")
    if isinstance(data, list) and data:
        print(json.dumps(data[0], indent=2, ensure_ascii=False)[:10000])
    elif isinstance(data, dict):
        print(json.dumps(data, indent=2, ensure_ascii=False)[:10000])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.paths:
        try:
            inspect_file(path)
        except Exception as exc:
            print(f"FAILED TO READ {path}: {exc}")


if __name__ == "__main__":
    main()
