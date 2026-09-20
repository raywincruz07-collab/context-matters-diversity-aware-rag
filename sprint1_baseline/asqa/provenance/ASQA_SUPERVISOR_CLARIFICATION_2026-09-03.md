# ASQA Supervisor Clarification — 2026-09-03

## Status

This note records explicit methodological guidance received from supervisor
Andreea on 2026-09-03 for the Sprint 3 ASQA experiment.

This clarification is authoritative over earlier project-internal planning where
there is a conflict.

## Canonical ASQA corpus

Use the full standard DPR English Wikipedia passage collection.

Canonical corpus:

- snapshot lineage: DPR Wikipedia 2018-12-20
- passage count: 21,015,324
- canonical passage surface: exact DPR passage BODY
- no corpus subsampling is permitted for the canonical ASQA experiment

The full ASQA official dev split of 948 questions remains
PROJECT_PROTECTED_FINAL.

## Retriever execution

The same full DPR Wikipedia corpus must underlie the ASQA retrieval experiment.

Published/prebuilt artifacts should be used where compatible rather than
re-encoding the full corpus unnecessarily:

- BM25: use a compatible published/prebuilt Lucene index over the DPR Wikipedia
  collection.
- DPR: use a compatible published/prebuilt FAISS/indexed retrieval artifact over
  the DPR Wikipedia collection.
- Contriever: use the published passage embeddings over the DPR Wikipedia
  collection.
- ColBERTv2: build the required full-corpus index over the same DPR Wikipedia
  collection.

Exact artifact identity, model/checkpoint identity, corpus compatibility, and
passage-ID mapping must be verified and frozen before canonical execution.

## ColBERTv2 contingency

If the full-corpus ColBERTv2 index cannot be completed in time, do not shrink
the ASQA corpus.

Instead:

- omit ColBERTv2 from the ASQA experiment;
- report ColBERTv2 on HotpotQA only;
- retain BM25, DPR, and Contriever on the full ASQA DPR Wikipedia corpus;
- document the ColBERTv2 omission transparently as a resource/time limitation.

This contingency changes retriever availability for ASQA only. It does not
authorize any reduced ASQA corpus.

## Unchanged methodology

This clarification does not reopen:

- ASQA train/dev partitioning;
- DEVELOPMENT / SELECTION / PROJECT_PROTECTED_FINAL roles;
- aspect matcher or alias rules;
- SRecall@5;
- alpha-nDCG@5;
- coverability or c*;
- candidate_pool=20;
- top_k=5;
- diversification treatments;
- generation protocol;
- statistical protocol.

No ASQA SELECTION or PROJECT_PROTECTED_FINAL outcomes were inspected to make
this clarification.
