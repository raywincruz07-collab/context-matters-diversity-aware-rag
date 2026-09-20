# Sprint 1 Baseline — PubMedQA

This directory contains the reproducible PubMedQA baseline experiments for Sprint 1 of the Context Matters diversity-aware RAG project.

## Dataset

- Source: `qiaojin/PubMedQA`
- Configuration: `pqa_labeled`
- Split: `train`
- Revision: `9001f2853fb87cab8d220904e0de81ac6973b318`
- Evaluation samples: 1,000
- Sampling: `full_split_source_order.v1`
- Constructed retrieval corpus: 3,358 documents
- Corpus construction: `pubmedqa_context_sections_source_order_strip.v1`

Frozen identities:
- `manifests/pubmedqa_sample_manifest_v2.json`
- `manifests/pubmedqa_corpus_manifest_v1.json`

## Baseline Conditions

Five context conditions are evaluated:
- without retrieved context
- BM25
- DPR
- Contriever
- ColBERTv2

Each condition uses three logical generator configurations:
- `llama-3.3-70b`
- `gemma4-26b`
- `ministral-3-14b`

Generator bindings are frozen in `configs/maki_model_bindings_v7.json`.

This yields 15 experiment blocks × 1,000 samples = 15,000 generation records.

## Canonical Raw Generations

Archive:
`raw_generation/pubmedqa_sprint1_canonical_raw_v1.tar.gz`

SHA256:
`cab27f82ed9153e3a7e8e65400ff993fff5c00e0a18bf6a91aa16afc7ef99d33`

Verified contents:
- 15 experiment inventories
- 15,000 generation samples
- 14,966 OK
- 33 PARSE_FAILURE
- 1 TRUNCATED
- 0 ERROR
- 0 REFUSAL

Integrity check:

```bash
cd raw_generation
sha256sum -c pubmedqa_sprint1_canonical_raw_v1.tar.gz.sha256
```

## Governed BM25 + Llama Replacement

Original failed run:
`run-sprint1-pubmedqa-s1-with-context-bm25-none-llama-3-3-70b-ffb0e1228071b46cb867d649`

Canonical replacement:
`run-sprint1-pubmedqa-s1-with-context-bm25-none-llama-3-3-70b-06239dda58f6febfd0be97cd`

The replacement is the BM25 + Llama block contained in the canonical archive. Run provenance is retained in `provenance/pubmedqa_run_registry_v1.jsonl`.

## Repository Contents

- `manifests/` — frozen sample and corpus manifests
- `retrieval/` — baseline retrieval metrics and summaries
- `selected_contexts/` — selected top contexts used for generation
- `raw_generation/` — canonical raw outputs and checksum
- `analysis/` — final evaluation tables
- `notebooks/` — final PubMedQA baseline notebook
- `scripts/` — dataset, retrieval, generation, and evaluation entry points
- `src/` — implementation dependencies
- `configs/` — model bindings
- `provenance/` — repeatability gate and run registry
- `environment/` — dependency specifications and lock files

## Final Analysis Outputs

- `analysis/decision_accuracy_summary.csv`
- `analysis/decision_class_recall.csv`
- `analysis/generation_correctness_per_query.parquet`
- `analysis/generation_status_summary.csv`
- `analysis/paired_context_effects.csv`
- `analysis/retrieval_answer_summary.csv`
- `notebooks/01_pubmedqa_baseline.ipynb`

Historical PubMedQA Sprint 1 outputs from the earlier project version are intentionally not included here.
