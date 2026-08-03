# Canonical Dataset Split Policy

## Canonical instance representation

Each processed record represents one target aspect and one sentiment label.

All target aspects originating from the same source sentence share the same
`sentence_id` and must remain in the same operational split.

The canonical processed record contains:

- tokenized sentence text;
- part-of-speech tags;
- dependency heads and relations;
- target aspect text and token span;
- sentiment polarity and numeric label;
- dependency graph edges;
- target-to-token dependency distances;
- source sentence and instance identifiers;
- operational split metadata.

## Official data partitions

The original training and test files are treated as authoritative source
partitions.

The official test data is never modified or used to construct the validation
set.

The validation set is derived only from the original training data.

## Validation split

Validation splitting uses:

- validation ratio: `0.15`;
- random seed: `2026`;
- sentence-level grouping;
- duplicate-aware normalized-text grouping;
- sentiment-signature stratification.

All aspects from one source sentence remain together.

Exact repeated normalized sentences also remain together, even when they have
different source sentence identifiers.

## Split metadata

Operational split metadata is explicitly assigned after partitioning:

- training records use `split="train"`;
- validation records use `split="validation"`;
- official test records use `split="test"`;
- `train_full.jsonl` retains the original `split="train"` metadata.

Stable `instance_id` and `sentence_id` values are preserved as source
identifiers when records move into validation.

## Official-test duplicate protection

Any normalized training sentence that also occurs in the official test set is
excluded from validation selection.

Such records remain in training so that official test examples cannot
indirectly influence model selection or early stopping through the validation
set.

## Duplicate findings

Five normalized sentence groups occur in both the official training and test
partitions:

- one Laptop group;
- one Restaurant group;
- three Twitter groups.

Four of these groups also share the same target signature.

No polarity conflicts were found among the duplicated target examples.

These official train-test duplicates are retained for benchmark compatibility.
They must also be reported through a duplicate-excluded sensitivity
evaluation during final experimentation.

## Final operational counts

| Domain | Training | Validation | Official test |
|---|---:|---:|---:|
| Laptops | 1,928 | 354 | 632 |
| Restaurants | 3,022 | 586 | 1,119 |
| Tweets | 5,144 | 907 | 677 |

The full training-source counts remain:

- Laptops: 2,282 aspect instances;
- Restaurants: 3,608 aspect instances;
- Tweets: 6,051 aspect instances.

## Integrity requirements

A valid generated split must satisfy all of the following:

1. Every processed record passes the canonical schema checks.
2. `train + validation` reconstructs `train_full`.
3. No `sentence_id` appears in more than one operational split.
4. No normalized sentence text crosses train and validation.
5. No normalized sentence text crosses validation and official test.
6. Validation records contain `split="validation"`.
7. Generated outputs are deterministic under seed `2026`.
8. Official raw and test files remain unchanged.
