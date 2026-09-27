# Sprint 2 — Diversity-Aware Retrieval and Generation

This directory is the compact reproducibility package for Sprint 2 of
**CONTEXT MATTERS — Evaluating Diversity-Aware Retrieval for RAG**.

Sprint 2 takes the frozen relevance baselines from Sprint 1 and changes the
composition of the retrieved context before generation.

## Experimental design

For every query:

1. retrieve a frozen Top-20 candidate pool;
2. apply one of the predefined diversification conditions;
3. select the final Top-5 passages;
4. provide those Top-5 passages to the generator;
5. preserve the resulting generation for later evaluation.

The final eight retrieval conditions are:

- `none`
- `mmr_0`
- `mmr_0.25`
- `mmr_0.5`
- `mmr_0.75`
- `kmeans_k2`
- `agglo_k3`
- `dpp_map`

The three logical generator conditions are:

- `gemma4-26b`
- `llama-3.3-70b`
- `ministral-3-14b`

The frozen logical-to-physical mapping preserves:

`ministral-3-14b -> qwen3.6-36b`

The historical logical identifier is retained for reproducibility; final
reporting may display the physical model name Qwen 3.6 36B.

## Datasets and retrievers

### PubMedQA

Retrievers:

- BM25
- DPR
- Contriever
- ColBERTv2

### HotpotQA

Retrievers:

- BM25
- DPR
- Contriever
- ColBERTv2

### ASQA

Retrievers:

- BM25
- DPR
- Contriever

Full-corpus ColBERTv2 was not completed for ASQA under the
supervisor-approved full-corpus constraint and is intentionally not
represented as a completed ASQA condition.

## Diversification implementation

The exact scientific implementation is preserved under:

`src/diversification/`

It contains:

- MMR
- KMeans-based selection
- agglomerative-clustering selection
- DPP MAP approximation
- dispatch/common utilities

No diversification implementation was rewritten during repository cleanup.

## MMR

The frozen implementation balances normalized native retrieval relevance
against semantic redundancy:

\[
MMR_\lambda(i|S)
=
\lambda \tilde r_i
-
(1-\lambda)
\max_{j \in S}
\tilde e_i^\top \tilde e_j
\]

The final experiment uses lambda values:

`0, 0.25, 0.5, 0.75`

## Final generation scale

Frozen completed generation counts:

- PubMedQA: 96,000
- HotpotQA: 710,880
- ASQA: 68,256

Total diversified-generation collection:

**875,136 generations**

These raw outputs are intentionally not stored in Git.

## What this directory contains

- exact diversification source code;
- final dataset-specific diversification producers;
- final generation producers required for reproducibility;
- frozen logical/physical model binding;
- compact generation-package metadata;
- compact final retrieval/diversification summaries;
- final generation and retrieval freeze records;
- file-level SHA-256 inventory.

## What this directory intentionally excludes

Large or operational artifacts are not committed:

- raw generation outputs;
- materialized generation-input JSONL files;
- full candidate-pool JSONL files;
- full retrieval-metric JSONL files;
- large correctness/faithfulness per-generation outputs;
- model caches;
- indexes and embedding stores;
- cluster logs;
- PID/JOBID files;
- checkpoint tarballs;
- temporary or `_staging_*` directories;
- decomposer experiments not part of the final design.

Those resources remain outside Git and are represented through frozen
manifests, hashes, counts, and provenance.

## Scientific boundary

This directory represents the **Sprint 2 intervention layer**:

`baseline retrieval -> Top-20 -> diversification -> Top-5 -> generation`

Downstream correctness, faithfulness, coverage, retrieval-diversity,
output-diversity, trade-off, correlation, and paraphrase-robustness analyses
belong to the final evaluation package rather than this Sprint 2 directory.

No scientific experiment was rerun to create this repository package.
