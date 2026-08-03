# Dependency Graph Source Policy

## Authoritative representation

The dependency-head sequence in each base CoreNLP JSON file is the
authoritative source for graph construction.

Stanford CoreNLP heads use the following convention:

- `0` denotes the root;
- positive values are one-based governor indices.

The preprocessing pipeline converts these values to:

- `-1` for the root;
- zero-based governor indices for all non-root tokens.

## Shortest-path reconstruction

Undirected token-to-token shortest-path distances are reconstructed directly
from dependency heads.

For compatibility with the legacy `short` matrices, distances are clipped at
a maximum value of `5`.

## Verification result

Generated shortest-path matrices were compared against every usable legacy
base/`_write` pair.

The following five pairs matched exactly:

- Laptop training
- Laptop test
- Restaurant test
- Twitter training
- Twitter test

A total of 9,190 records matched with zero matrix mismatches.

## Restaurant training exception

`Restaurants_corenlp/train_write.json` is unusable because:

- the base training file contains 1,980 records;
- the corresponding `_write` file contains only 1 record;
- that record does not match record 0 in the base training file.

The raw file is retained unchanged for provenance but is never used as a graph
or distance source.

Restaurant training distances are reconstructed from the complete dependency
heads in `Restaurants_corenlp/train.json`.

## Research policy

The project must not depend on legacy `short` matrices during model training,
validation, or testing.

Legacy matrices may be used only for:

- integrity verification;
- reproducibility checks;
- confirming the clipping convention.

This policy ensures that all three domains use one consistent and reproducible
graph-construction procedure.
