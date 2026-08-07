# Aspect-Based Sentiment Analysis Research Framework

This repository provides a controlled experimental environment for studying
target-conditioned sentiment reasoning in Aspect-Based Sentiment Analysis
(ABSA).

## Research Objective

The project investigates how neural language models associate sentiment
information with the correct aspect or target, particularly in sentences
containing multiple competing sentiment signals.

The repository intentionally separates:

1. benchmark preparation,
2. a strong sequence-pair Transformer baseline,
3. future experimental architectures.

No proposed architecture is treated as correct by default.

## Datasets

Three standard ABSA benchmarks are supported:

- SemEval Laptop
- SemEval Restaurant
- Twitter Target-Dependent Sentiment

The processed benchmark sizes are:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| Laptop | 1933 | 349 | 632 |
| Restaurant | 3022 | 586 | 1119 |
| Twitter | 5144 | 907 | 677 |

Splits are performed at sentence level to prevent multi-aspect sentence
leakage between training and validation.

## Baseline

The reference baseline is RoBERTa-base using sequence-pair encoding:

    sentence + aspect

The aspect therefore conditions Transformer representations throughout the
encoder.

Current Laptop reference result:

- Test Accuracy: 82.75%
- Test Macro-F1: 79.89%

This baseline is frozen and serves as the control condition for future
experiments.

## Research Principles

Future proposed models must:

- use the fixed benchmark splits;
- compare against the same baseline protocol;
- avoid test-set-driven architecture tuning;
- isolate one research hypothesis per experiment;
- include appropriate ablations;
- distinguish benchmark performance from mechanistic claims.

Historical Target-Evidence V1/V2 experiments are preserved in Git history
and the archive/target-evidence-v1-v2 branch.

## Repository Structure

    data/               Processed benchmark metadata
    scripts/            Dataset and baseline experiment commands
    src/data/           Dataset parsing and graph utilities
    src/models/         Future experimental models
    src/evaluation/     Evaluation methods
    src/training/       Training infrastructure
    tests/              Integrity and model tests
    outputs/            Local experiment artifacts
