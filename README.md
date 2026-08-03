# Aspect-Based Sentiment Analysis with Large Language Models

A research-oriented framework for studying reliable target-conditioned
reasoning in large language models using aspect-based sentiment analysis.

The repository currently supports three domains:

- Laptop reviews
- Restaurant reviews
- Twitter posts

## Research Goal

The broader objective is to investigate whether large language models bind
sentiment evidence to the correct target rather than relying on globally
salient or domain-specific sentiment cues.

The project will study:

- target-specific evidence binding;
- cross-domain generalization;
- counterfactual intervention consistency;
- syntax-guided LLM adaptation;
- explanation faithfulness;
- uncertainty and selective prediction;
- robustness under linguistic and structural perturbations.

## Dataset Representation

Each processed example may contain:

- tokenized text;
- part-of-speech tags;
- dependency heads;
- dependency relation labels;
- one or more target aspects;
- aspect token boundaries;
- sentiment polarity;
- dependency shortest-path distances.

The principal sentiment labels are:

- `positive`
- `negative`
- `neutral`

## Planned Repository Structure

```text
configs/                         Experiment configurations
docs/                            Research and dataset documentation
notebooks/                       Exploratory analysis notebooks
scripts/                         Command-line experiment scripts
src/aspect_sentiment/data/       Dataset loading and validation
src/aspect_sentiment/models/     Baselines and proposed models
src/aspect_sentiment/evaluation/ Metrics and evaluation protocols
src/aspect_sentiment/utils/      Shared utilities
tests/                           Unit and integrity tests
artifacts/                       Generated experiment outputs
