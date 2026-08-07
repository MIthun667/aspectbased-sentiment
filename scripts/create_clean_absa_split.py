from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def norm(x: str) -> str:
    x = str(x).lower().strip()
    return re.sub(r"\s+", " ", x)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--domain",
        default="Laptop",
        choices=["Laptop", "Restaurants", "Twitter"],
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "TPHCA" / "unFixedData",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=25,
    )

    args = parser.parse_args()

    source = (
        args.data_root
        / args.domain
        / "train.csv"
    )

    df = pd.read_csv(source).reset_index(drop=True)

    df["context_norm"] = (
        df["context"]
        .map(norm)
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.val_size,
        random_state=args.seed,
    )

    train_idx, val_idx = next(
        splitter.split(
            df,
            groups=df["context_norm"],
        )
    )

    train_df = (
        df.iloc[train_idx]
        .reset_index(drop=True)
        .copy()
    )

    val_df = (
        df.iloc[val_idx]
        .reset_index(drop=True)
        .copy()
    )

    train_contexts = set(
        train_df["context_norm"]
    )

    val_contexts = set(
        val_df["context_norm"]
    )

    overlap = (
        train_contexts
        & val_contexts
    )

    if overlap:
        raise RuntimeError(
            f"Sentence leakage detected: {len(overlap)} overlaps"
        )

    output_dir = (
        Path("data")
        / "clean_splits"
        / args.domain.lower()
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        output_dir
        / "train.csv"
    )

    val_path = (
        output_dir
        / "val.csv"
    )

    train_df.to_csv(
        train_path,
        index=False,
    )

    val_df.to_csv(
        val_path,
        index=False,
    )

    summary = {
        "domain": args.domain,
        "seed": args.seed,
        "val_size": args.val_size,
        "source_rows": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_unique_sentences": (
            train_df["context_norm"].nunique()
        ),
        "val_unique_sentences": (
            val_df["context_norm"].nunique()
        ),
        "sentence_overlap": len(overlap),
        "train_label_counts": (
            train_df["polarity"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "val_label_counts": (
            val_df["polarity"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
    }

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print()
    print("Saved:")
    print(train_path)
    print(val_path)


if __name__ == "__main__":
    main()
