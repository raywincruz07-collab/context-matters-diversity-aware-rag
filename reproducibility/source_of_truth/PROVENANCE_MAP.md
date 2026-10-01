# CONTEXT MATTERS — Canonical Scientific Provenance Map

Status: **FROZEN SCIENTIFIC EXECUTION / FINAL-DELIVERY RECONSTRUCTION**

This document is a navigation and provenance layer over already completed scientific artifacts. It does not represent a new experiment, metric recomputation, retrieval run, generation run, embedding run, or NLI run.

## 1. Authority

- Final professor-facing Git authority: `/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3/context-matters-diversity-aware-rag-final-handoff`
- Final authority commit: `3769bc9e202de8852f2a68532b1da53cfe5b5aa2`
- Frozen historical scientific source repository: `/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3/context-matters-rag-sprint3`
- Frozen historical scientific source commit: `ea4d79bb4a742a23c79ede324c3550a1884805a9`
- The final-handoff repository is the current delivery authority.
- The historical research repository remains preserved as scientific lineage evidence and must not be cleaned or rewritten.

## 2. Canonical source-of-truth registries

- `EXPERIMENT_REGISTRY.csv` — 264 final experiment cells; SHA256 `36bea942e41b11fca9232b5fd51def07ff266cab62b9ab4fcc5c9746e196218d`
- `RESULTS_REGISTRY.csv` — 76 frozen result artifacts; SHA256 `3fd20110c19246a30276fcb02c4b5171937d2b3efd628d9ab12d9d7699a6dca8`
- `FIGURE_REGISTRY.csv` — 262 frozen figure files (131 PNG + 131 PDF); SHA256 `170d8030c9a05a922d33e5e8162aec888344f391815dd9e9c5688abe6c5c1e0e`

## 3. Final experimental populations

| Dataset | Population | Corpus | Final retrievers | Final cells | Generations |
|---|---:|---:|---|---:|---:|
| PubMedQA | 1,000 questions | 3,358 context sections | BM25, DPR, Contriever, ColBERTv2 | 96 | 96,000 |
| HotpotQA | 7,405 official test queries | 5,233,329 BEIR documents | BM25, DPR, Contriever, ColBERTv2 | 96 | 710,880 |
| ASQA | 948 dev questions | 21,015,324 DPR Wikipedia passages | BM25, DPR, Contriever | 72 | 68,256 |
| **Total** | — | — | — | **264** | **875,136** |

## 4. Canonical experiment flow

```text
Dataset / corpus
      ↓
Frozen question population
      ↓
Retriever
      ↓
Top-20 candidate pool
      ↓
Diversification condition
      ↓
Top-5 selected context
      ↓
Fixed dataset prompt
      ↓
Generator, temperature = 0
      ↓
Generated answer
      ↓
Phase 4 metrics
      ↓
Phase 5 paired / association analyses
      ↓
Frozen figures → notebooks → report → presentation
```

## 5. Final diversification conditions

1. `none` — relevance-ranked baseline; not a diversification treatment
2. `mmr_0` — MMR λ = 0
3. `mmr_0.25` — MMR λ = 0.25
4. `mmr_0.5` — MMR λ = 0.5
5. `mmr_0.75` — MMR λ = 0.75
6. `kmeans_k2` — K-Means k = 2
7. `agglo_k3` — Agglomerative clustering k = 3
8. `dpp_map` — deterministic DPP MAP selection

## 6. Generator authority

- Physical models:
  - `gemma4-26b`
  - `llama-3.3-70b`
  - `qwen3.6-36b`
- Historical logical alias: `ministral-3-14b` → physical `qwen3.6-36b`.
- Temperature: `0`.
- PubMedQA max tokens: `256`.
- HotpotQA max tokens: `256`.
- ASQA max tokens: `512`.
- Generator comparisons are interpreted within the same generator; the three physical models provide replication across generators.

## 7. Evaluation lineage

| Phase | Scientific purpose | Frozen authority present |
|---|---|---:|
| 4.1 | Answer correctness | 12 registered artifacts |
| 4.2 | Faithfulness / hallucination | 4 registered artifacts |
| 4.3 | Evidence / answer coverage | 3 registered artifacts |
| 4.4 | Top-5 retrieval diversity | 9 registered artifacts |
| 4.5 | ASQA answer-output diversity | 3 registered artifacts |
| 5.1 | Diversity–accuracy relationship | 4 registered artifacts |
| 5.2 | Diversity–faithfulness / hallucination relationship | 3 registered artifacts |
| 5.3 | Context→output diversity association | 5 registered artifacts |
| 5.4 | Paraphrase robustness | 23 registered artifacts |

## 8. Important scientific distinctions

- `none` is the relevance-ranked Top-5 baseline and is not equivalent to `mmr_0`.
- Retrieval diversity is a manipulation/diagnostic signal, not an independent answer-quality metric.
- ASQA final experiments use BM25, DPR, and Contriever only. ColBERTv2 was not completed for the full ASQA corpus and no subsampled replacement is treated as final.
- Historical DEVELOPMENT and SELECTION artifacts are not interchangeable with protected-final results.
- Historical Sprint-1 observations and the later controlled `none` baseline must remain separately labelled.

## 9. Final presentation figure authority

- Frozen figure collection contains 131 PNG and 131 PDF files.
- Presentation storyboard core is F01–F22; every core figure is available in both PNG and PDF form.
- Figures are consumed from frozen artifacts; final-delivery work does not regenerate scientific plots unless explicitly justified later.

## 10. Final-delivery rule

From this point forward, notebooks, report text, GitHub documentation, and presentation claims must trace back to this source-of-truth layer and the frozen authoritative artifacts it indexes.

**No scientific reruns are authorized by this provenance map.**
