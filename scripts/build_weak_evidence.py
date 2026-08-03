#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.aspect_sentiment.evidence import (  # noqa: E402
    build_domain_manifest,
    build_evidence_split,
    load_weak_evidence_config,
    utc_timestamp,
)


DEFAULT_DOMAINS = (
    "laptops",
    "restaurants",
    "tweets",
)

DEFAULT_SPLITS = (
    "train",
    "validation",
    "test",
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic weak-evidence "
            "artifacts for processed ABSA data."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/evidence/default.yaml"
        ),
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/processed"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/derived/evidence"
        ),
    )

    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DEFAULT_DOMAINS),
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args(argv)


def require_source_file(
    path: Path,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing processed split: {path}"
        )


def require_output_permission(
    *,
    paths: Sequence[Path],
    overwrite: bool,
) -> None:
    existing = [
        path
        for path in paths
        if path.exists()
    ]

    if existing and not overwrite:
        formatted = "\n".join(
            f"  - {path}"
            for path in existing
        )

        raise FileExistsError(
            "Output files already exist. "
            "Use --overwrite to replace them:\n"
            f"{formatted}"
        )


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    config = load_weak_evidence_config(
        args.config
    )

    generation_timestamp = utc_timestamp()

    print(
        "Weak-evidence configuration:",
        config.to_dict(),
    )

    for domain in args.domains:
        domain_input_root = (
            args.input_root / domain
        )

        domain_output_root = (
            args.output_root / domain
        )

        split_paths = {}

        for split in args.splits:
            source_path = (
                domain_input_root
                / f"{split}.jsonl"
            )

            output_path = (
                domain_output_root
                / f"{split}.jsonl"
            )

            manifest_path = (
                domain_output_root
                / f"{split}.manifest.json"
            )

            require_source_file(
                source_path
            )

            split_paths[split] = (
                source_path,
                output_path,
                manifest_path,
            )

        domain_manifest_path = (
            domain_output_root
            / "manifest.json"
        )

        output_paths = [
            path
            for _, output_path, manifest_path
            in split_paths.values()
            for path in (
                output_path,
                manifest_path,
            )
        ]

        output_paths.append(
            domain_manifest_path
        )

        require_output_permission(
            paths=output_paths,
            overwrite=args.overwrite,
        )

        split_manifests = {}

        for split in args.splits:
            (
                source_path,
                output_path,
                manifest_path,
            ) = split_paths[split]

            print(
                f"Building {domain}/{split}..."
            )

            manifest = build_evidence_split(
                source_path=source_path,
                output_path=output_path,
                manifest_path=manifest_path,
                config=config,
                generated_at_utc=(
                    generation_timestamp
                ),
            )

            split_manifests[
                split
            ] = manifest

            statistics = manifest[
                "statistics"
            ]

            print(
                "  records:",
                statistics["record_count"],
            )

            print(
                "  coverage:",
                f'{statistics["coverage"]:.4f}',
            )

            print(
                "  mean selected:",
                (
                    f'{statistics["mean_selected_tokens"]:.2f}'
                ),
            )

        build_domain_manifest(
            domain=domain,
            split_manifests=split_manifests,
            config=config,
            output_path=(
                domain_manifest_path
            ),
            generated_at_utc=(
                generation_timestamp
            ),
        )

        print(
            f"Completed domain: {domain}"
        )

    print(
        "Weak-evidence artifact generation passed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
