from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.target_evidence_trrl2 import (
    TargetEvidenceTRRL2,
)


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def entropy_bits(probabilities: np.ndarray) -> float:
    p = probabilities[
        probabilities > 0
    ]

    if len(p) == 0:
        return 0.0

    return float(
        -(p * np.log2(p)).sum()
    )


def tvd(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    return float(
        0.5
        * np.abs(
            p - q
        ).sum()
    )


def jsd(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """
    Jensen-Shannon divergence in bits.
    """
    p = np.asarray(
        p,
        dtype=np.float64,
    )

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    p = p / max(
        p.sum(),
        1e-12,
    )

    q = q / max(
        q.sum(),
        1e-12,
    )

    m = 0.5 * (
        p + q
    )

    def kl(a, b):
        mask = a > 0

        return float(
            np.sum(
                a[mask]
                * np.log2(
                    a[mask]
                    / np.clip(
                        b[mask],
                        1e-12,
                        None,
                    )
                )
            )
        )

    return (
        0.5 * kl(p, m)
        +
        0.5 * kl(q, m)
    )


def find_target_char_span(
    row: pd.Series,
):
    """
    Prefer supplied termFrom/termTo if valid.

    Otherwise fall back to case-insensitive string search.
    """

    sentence = str(
        row["context"]
    )

    target = str(
        row["term"]
    )

    if (
        "termFrom" in row
        and
        "termTo" in row
    ):
        try:
            start = int(
                row["termFrom"]
            )

            end = int(
                row["termTo"]
            )

            if (
                0 <= start < end <= len(sentence)
            ):
                extracted = sentence[
                    start:end
                ]

                if (
                    normalize_text(
                        extracted
                    )
                    ==
                    normalize_text(
                        target
                    )
                ):
                    return (
                        start,
                        end,
                    )

        except Exception:
            pass

    match = re.search(
        re.escape(
            target
        ),
        sentence,
        flags=re.IGNORECASE,
    )

    if match:
        return (
            match.start(),
            match.end(),
        )

    return (
        None,
        None,
    )


class EvidenceAnalyzer:

    def __init__(
        self,
        model,
        tokenizer,
        device,
        max_length=128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    @torch.no_grad()
    def analyze_row(
        self,
        row: pd.Series,
    ):
        sentence = str(
            row["context"]
        )

        target = str(
            row["term"]
        )

        encoded = self.tokenizer(
            sentence,
            target,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        sequence_ids = (
            encoded.sequence_ids(
                0
            )
        )

        word_ids = (
            encoded.word_ids(
                0
            )
        )

        sentence_mask = torch.tensor(
            [
                1 if sid == 0 else 0
                for sid in sequence_ids
            ],
            dtype=torch.long,
        ).unsqueeze(0)

        target_mask = torch.tensor(
            [
                1 if sid == 1 else 0
                for sid in sequence_ids
            ],
            dtype=torch.long,
        ).unsqueeze(0)

        model_inputs = {
            "input_ids": encoded[
                "input_ids"
            ].to(
                self.device
            ),

            "attention_mask": encoded[
                "attention_mask"
            ].to(
                self.device
            ),

            "sentence_mask": sentence_mask.to(
                self.device
            ),

            "target_mask": target_mask.to(
                self.device
            ),
        }

        (
            relation_embedding,
            attention,
        ) = self.model(
            **model_inputs,
            return_attention=True,
        )

        attention = (
            attention[0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        offsets = (
            encoded[
                "offset_mapping"
            ][0]
            .cpu()
            .numpy()
            .tolist()
        )

        input_ids = (
            encoded[
                "input_ids"
            ][0]
            .cpu()
            .tolist()
        )

        tokens = (
            self.tokenizer
            .convert_ids_to_tokens(
                input_ids
            )
        )

        # ----------------------------------------
        # Sentence-level attention only.
        # ----------------------------------------
        sentence_positions = [
            i
            for i, sid
            in enumerate(
                sequence_ids
            )
            if sid == 0
        ]

        sentence_attention = np.array(
            [
                attention[i]
                for i
                in sentence_positions
            ],
            dtype=np.float64,
        )

        sentence_attention = (
            sentence_attention
            /
            max(
                sentence_attention.sum(),
                1e-12,
            )
        )

        # ----------------------------------------
        # Target occurrence mass in original
        # sentence.
        # ----------------------------------------
        target_start, target_end = (
            find_target_char_span(
                row
            )
        )

        target_sentence_mass = 0.0

        target_sentence_positions = []

        if (
            target_start is not None
            and target_end is not None
        ):
            for i in sentence_positions:
                start, end = (
                    offsets[i]
                )

                if end <= start:
                    continue

                overlaps = (
                    start < target_end
                    and
                    end > target_start
                )

                if overlaps:
                    target_sentence_mass += float(
                        attention[i]
                    )

                    target_sentence_positions.append(
                        i
                    )

        # ----------------------------------------
        # Aggregate subwords into tokenizer words.
        # ----------------------------------------
        word_groups = defaultdict(
            lambda: {
                "attention": 0.0,
                "starts": [],
                "ends": [],
                "positions": [],
            }
        )

        for i in sentence_positions:

            wid = word_ids[i]

            if wid is None:
                continue

            start, end = offsets[i]

            if end <= start:
                continue

            group = word_groups[
                int(wid)
            ]

            group[
                "attention"
            ] += float(
                attention[i]
            )

            group[
                "starts"
            ].append(
                start
            )

            group[
                "ends"
            ].append(
                end
            )

            group[
                "positions"
            ].append(
                i
            )

        words = []

        for wid in sorted(
            word_groups
        ):
            group = word_groups[
                wid
            ]

            start = min(
                group["starts"]
            )

            end = max(
                group["ends"]
            )

            text = sentence[
                start:end
            ]

            overlaps_target = False

            if (
                target_start
                is not None
                and
                target_end
                is not None
            ):
                overlaps_target = (
                    start < target_end
                    and
                    end > target_start
                )

            words.append(
                {
                    "word_id": wid,
                    "text": text,
                    "start": start,
                    "end": end,
                    "attention": float(
                        group[
                            "attention"
                        ]
                    ),
                    "is_target": bool(
                        overlaps_target
                    ),
                }
            )

        words = sorted(
            words,
            key=lambda x:
            x["attention"],
            reverse=True,
        )

        return {
            "relation_embedding": (
                relation_embedding[
                    0
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
            ),

            "full_attention": (
                attention
            ),

            "sentence_positions": (
                sentence_positions
            ),

            "sentence_attention": (
                sentence_attention
            ),

            "target_sentence_mass": float(
                target_sentence_mass
            ),

            "attention_entropy_bits": (
                entropy_bits(
                    sentence_attention
                )
            ),

            "top_words": words,

            "tokens": tokens,
            "sequence_ids": sequence_ids,
        }


def compute_retrieval_correctness(
    embeddings,
    dataframe,
):
    """
    Same held-out relation retrieval criterion used during
    TRRL-2 evaluation.

    Returns:
        dict[row_index] -> True / False
    """

    df = (
        dataframe
        .reset_index(
            drop=True
        )
        .copy()
    )

    df["target_norm"] = (
        df["term"]
        .map(
            normalize_text
        )
    )

    df["context_norm"] = (
        df["context"]
        .map(
            normalize_text
        )
    )

    similarity = (
        embeddings
        @ embeddings.T
    )

    correctness = {}

    for i, row in df.iterrows():

        same_target = np.array(
            df["target_norm"]
            ==
            row["target_norm"]
        )

        same_context = np.array(
            df["context_norm"]
            ==
            row["context_norm"]
        )

        same_polarity = np.array(
            df["polarity"]
            ==
            row["polarity"]
        )

        candidates = (
            same_target
            & ~same_context
        )

        positives = (
            candidates
            & same_polarity
        )

        negatives = (
            candidates
            & ~same_polarity
        )

        if (
            positives.sum() == 0
            or
            negatives.sum() == 0
        ):
            continue

        scores = (
            similarity[
                i
            ].copy()
        )

        scores[
            ~candidates
        ] = -1e9

        nearest = int(
            np.argmax(
                scores
            )
        )

        correctness[
            i
        ] = bool(
            positives[
                nearest
            ]
        )

    return correctness


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "trrl2_heldout"
            / "laptop"
            / "seed_25"
        ),
    )

    parser.add_argument(
        "--model",
        default="roberta-base",
    )

    parser.add_argument(
        "--relation-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--attention-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    experiment_dir = (
        args.experiment_dir
    )

    checkpoint_path = (
        experiment_dir
        / "best_relation_encoder.pt"
    )

    heldout_path = (
        experiment_dir
        / "heldout_split.csv"
    )

    embedding_path = (
        experiment_dir
        / "heldout_embeddings.pt"
    )

    for path in [
        checkpoint_path,
        heldout_path,
        embedding_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    df = pd.read_csv(
        heldout_path
    ).reset_index(
        drop=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.model,
            use_fast=True,
        )
    )

    model = (
        TargetEvidenceTRRL2(
            backbone_name=(
                args.model
            ),
            relation_dim=(
                args.relation_dim
            ),
            attention_dim=(
                args.attention_dim
            ),
            dropout=0.1,
            freeze_backbone=True,
        )
    )

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(
        state
    )

    model.to(
        device
    )

    model.eval()

    analyzer = EvidenceAnalyzer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=(
            args.max_length
        ),
    )

    print(
        "=" * 90
    )

    print(
        "TRRL-2 HELD-OUT EVIDENCE ANALYSIS"
    )

    print(
        "=" * 90
    )

    print(
        "Held-out rows:",
        len(df),
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    # ------------------------------------------
    # Analyze every held-out example.
    # ------------------------------------------
    analyses = []

    embeddings = []

    for idx, row in df.iterrows():

        result = (
            analyzer
            .analyze_row(
                row
            )
        )

        embeddings.append(
            result[
                "relation_embedding"
            ]
        )

        analyses.append(
            {
                "row_index": int(
                    idx
                ),
                "target": str(
                    row["term"]
                ),
                "polarity": int(
                    row["polarity"]
                ),
                "context": str(
                    row["context"]
                ),
                "target_mass": (
                    result[
                        "target_sentence_mass"
                    ]
                ),
                "entropy_bits": (
                    result[
                        "attention_entropy_bits"
                    ]
                ),
                "top_words": (
                    result[
                        "top_words"
                    ][
                        :args.top_k
                    ]
                ),
                "_attention": (
                    result[
                        "sentence_attention"
                    ]
                ),
                "_sentence_positions": (
                    result[
                        "sentence_positions"
                    ]
                ),
            }
        )

    embeddings = np.stack(
        embeddings
    )

    # ------------------------------------------
    # Correct vs incorrect relation retrieval.
    # ------------------------------------------
    retrieval_correctness = (
        compute_retrieval_correctness(
            embeddings,
            df,
        )
    )

    correct_rows = []
    incorrect_rows = []

    for analysis in analyses:

        idx = analysis[
            "row_index"
        ]

        if idx not in retrieval_correctness:
            continue

        if retrieval_correctness[
            idx
        ]:
            correct_rows.append(
                analysis
            )
        else:
            incorrect_rows.append(
                analysis
            )

    def summarize_group(
        rows,
    ):
        if not rows:
            return {
                "n": 0,
                "mean_target_mass": 0.0,
                "mean_entropy_bits": 0.0,
            }

        return {
            "n": len(
                rows
            ),

            "mean_target_mass": float(
                np.mean(
                    [
                        x[
                            "target_mass"
                        ]
                        for x in rows
                    ]
                )
            ),

            "mean_entropy_bits": float(
                np.mean(
                    [
                        x[
                            "entropy_bits"
                        ]
                        for x in rows
                    ]
                )
            ),
        }

    correct_summary = (
        summarize_group(
            correct_rows
        )
    )

    incorrect_summary = (
        summarize_group(
            incorrect_rows
        )
    )

    # ------------------------------------------
    # Target-switch analysis.
    #
    # Same sentence appearing with >1 target
    # in held-out split.
    # ------------------------------------------
    context_groups = defaultdict(
        list
    )

    for i, row in df.iterrows():

        context_groups[
            normalize_text(
                row[
                    "context"
                ]
            )
        ].append(
            i
        )

    switch_pairs = []

    analysis_by_idx = {
        x["row_index"]: x
        for x in analyses
    }

    for indices in (
        context_groups.values()
    ):
        if len(
            indices
        ) < 2:
            continue

        for a_pos in range(
            len(indices)
        ):
            for b_pos in range(
                a_pos + 1,
                len(indices),
            ):

                a = indices[
                    a_pos
                ]

                b = indices[
                    b_pos
                ]

                row_a = df.iloc[
                    a
                ]

                row_b = df.iloc[
                    b
                ]

                if (
                    normalize_text(
                        row_a[
                            "term"
                        ]
                    )
                    ==
                    normalize_text(
                        row_b[
                            "term"
                        ]
                    )
                ):
                    continue

                att_a = (
                    analysis_by_idx[
                        a
                    ][
                        "_attention"
                    ]
                )

                att_b = (
                    analysis_by_idx[
                        b
                    ][
                        "_attention"
                    ]
                )

                # Sentence tokenization should be
                # identical for same raw context.
                if (
                    len(att_a)
                    !=
                    len(att_b)
                ):
                    continue

                switch_pairs.append(
                    {
                        "row_a": int(
                            a
                        ),
                        "row_b": int(
                            b
                        ),
                        "target_a": str(
                            row_a[
                                "term"
                            ]
                        ),
                        "target_b": str(
                            row_b[
                                "term"
                            ]
                        ),
                        "polarity_a": int(
                            row_a[
                                "polarity"
                            ]
                        ),
                        "polarity_b": int(
                            row_b[
                                "polarity"
                            ]
                        ),
                        "context": str(
                            row_a[
                                "context"
                            ]
                        ),
                        "tvd": tvd(
                            att_a,
                            att_b,
                        ),
                        "jsd_bits": jsd(
                            att_a,
                            att_b,
                        ),
                    }
                )

    switch_summary = {
        "pairs": len(
            switch_pairs
        ),

        "mean_tvd": float(
            np.mean(
                [
                    x[
                        "tvd"
                    ]
                    for x
                    in switch_pairs
                ]
            )
        )
        if switch_pairs
        else 0.0,

        "std_tvd": float(
            np.std(
                [
                    x[
                        "tvd"
                    ]
                    for x
                    in switch_pairs
                ]
            )
        )
        if switch_pairs
        else 0.0,

        "mean_jsd_bits": float(
            np.mean(
                [
                    x[
                        "jsd_bits"
                    ]
                    for x
                    in switch_pairs
                ]
            )
        )
        if switch_pairs
        else 0.0,

        "std_jsd_bits": float(
            np.std(
                [
                    x[
                        "jsd_bits"
                    ]
                    for x
                    in switch_pairs
                ]
            )
        )
        if switch_pairs
        else 0.0,
    }

    # ------------------------------------------
    # Overall summaries.
    # ------------------------------------------
    overall_target_mass = float(
        np.mean(
            [
                x[
                    "target_mass"
                ]
                for x
                in analyses
            ]
        )
    )

    overall_entropy = float(
        np.mean(
            [
                x[
                    "entropy_bits"
                ]
                for x
                in analyses
            ]
        )
    )

    summary = {
        "heldout_rows": int(
            len(df)
        ),

        "overall_mean_target_mass": (
            overall_target_mass
        ),

        "overall_mean_attention_entropy_bits": (
            overall_entropy
        ),

        "retrieval_correct": (
            correct_summary
        ),

        "retrieval_incorrect": (
            incorrect_summary
        ),

        "target_switch": (
            switch_summary
        ),
    }

    # ------------------------------------------
    # Human-readable report.
    # ------------------------------------------
    output_dir = (
        experiment_dir
        / "evidence_analysis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_dir
        / "evidence_summary.json"
    )

    serializable_analyses = []

    for item in analyses:
        item = dict(
            item
        )

        item.pop(
            "_attention",
            None,
        )

        item.pop(
            "_sentence_positions",
            None,
        )

        serializable_analyses.append(
            item
        )

    payload = {
        "summary": summary,
        "target_switch_pairs": (
            switch_pairs
        ),
        "examples": (
            serializable_analyses
        ),
    }

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_path = (
        output_dir
        / "evidence_report.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "TRRL-2 HELD-OUT EVIDENCE ANALYSIS\n"
        )

        f.write(
            "=" * 90
            + "\n\n"
        )

        f.write(
            json.dumps(
                summary,
                indent=2,
            )
        )

        f.write(
            "\n\n"
        )

        # Most target-sensitive pairs.
        f.write(
            "MOST TARGET-SENSITIVE SAME-SENTENCE PAIRS\n"
        )

        f.write(
            "=" * 90
            + "\n"
        )

        ranked_switches = sorted(
            switch_pairs,
            key=lambda x:
            x["tvd"],
            reverse=True,
        )

        for pair in ranked_switches[
            :20
        ]:

            f.write(
                "\n"
                + "-" * 90
                + "\n"
            )

            f.write(
                f"Target A: "
                f"{pair['target_a']} "
                f"(label={pair['polarity_a']})\n"
            )

            f.write(
                f"Target B: "
                f"{pair['target_b']} "
                f"(label={pair['polarity_b']})\n"
            )

            f.write(
                f"TVD: "
                f"{pair['tvd']:.4f}\n"
            )

            f.write(
                f"JSD(bits): "
                f"{pair['jsd_bits']:.4f}\n"
            )

            f.write(
                f"Sentence: "
                f"{pair['context']}\n"
            )

        f.write(
            "\n\nQUALITATIVE EVIDENCE EXAMPLES\n"
        )

        f.write(
            "=" * 90
            + "\n"
        )

        shown = 0

        # Prefer examples that participate
        # in relation retrieval evaluation.
        ordered_examples = (
            correct_rows[
                :args.examples // 2
            ]
            +
            incorrect_rows[
                :args.examples // 2
            ]
        )

        for item in ordered_examples:

            if shown >= args.examples:
                break

            shown += 1

            idx = item[
                "row_index"
            ]

            status = (
                "CORRECT"
                if retrieval_correctness.get(
                    idx,
                    False,
                )
                else "INCORRECT"
            )

            f.write(
                "\n"
                + "-" * 90
                + "\n"
            )

            f.write(
                f"Retrieval: {status}\n"
            )

            f.write(
                f"Target: "
                f"{item['target']}\n"
            )

            f.write(
                f"Polarity: "
                f"{item['polarity']}\n"
            )

            f.write(
                f"Target attention mass: "
                f"{item['target_mass']:.4f}\n"
            )

            f.write(
                f"Entropy(bits): "
                f"{item['entropy_bits']:.4f}\n"
            )

            f.write(
                "Top evidence:\n"
            )

            for word in item[
                "top_words"
            ]:

                marker = (
                    " [TARGET]"
                    if word[
                        "is_target"
                    ]
                    else ""
                )

                f.write(
                    f"  "
                    f"{word['text']!r:20s} "
                    f"{word['attention']:.4f}"
                    f"{marker}\n"
                )

            f.write(
                f"Sentence: "
                f"{item['context']}\n"
            )

    print(
        "\nSUMMARY"
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\nSaved:"
    )

    print(
        json_path
    )

    print(
        report_path
    )


if __name__ == "__main__":
    main()
