# Reproducibility Guide

This document explains how to inspect, verify, and reproduce the completed
University of Mannheim team project:

CONTEXT MATTERS — Evaluating Diversity-Aware Retrieval for RAG

The scientific experiments are complete and frozen.

The repository preserves the original scientific code, configuration,
manifests, compact results, provenance, and selected original final execution
scripts needed to inspect how the reported results were produced.

No scientific Python source was rewritten, refactored, cleaned, formatted, or
parameterized for this handoff.

## 1. Repository structure

    sprint1_baseline/
        pubmedqa/
        hotpotqa/
        asqa/

    sprint2_diversification/

    sprint3_evaluation/

    reproducibility/
        original_final_scripts/
        original_final_protocols/

Sprint 1, Sprint 2, and Sprint 3 are the frozen scientific handoff packages.

The directory reproducibility/original_final_scripts contains additional final
execution scripts copied byte-for-byte from the original BWUNICLUSTER
scientific workspace.

The directory reproducibility/original_final_protocols contains selected final
scientific protocol documents copied byte-for-byte from the same workspace.

## 2. Integrity verification

Verify Sprint 2:

    cd sprint2_diversification
    sha256sum -c SHA256SUMS

Verify Sprint 3:

    cd ../sprint3_evaluation
    sha256sum -c SHA256SUMS

Verify the additional original execution scripts:

    cd ../reproducibility/original_final_scripts
    sha256sum -c ORIGINAL_FILES_SHA256SUMS

Verify the copied original protocol documents:

    cd ../original_final_protocols
    sha256sum -c ORIGINAL_PROTOCOLS_SHA256SUMS

## 3. Experimental pipeline

The frozen pipeline is:

    question
      -> retriever
      -> Top-20 candidate passages
      -> diversification
      -> Top-5 selected passages
      -> LLM generation
      -> Phase 4 evaluation
      -> Phase 5 cross-metric analysis

Datasets:

- PubMedQA
- HotpotQA
- ASQA

Retrievers:

- BM25
- DPR
- Contriever
- ColBERTv2 where feasible

Final diversification conditions:

- none
- mmr_0
- mmr_0.25
- mmr_0.5
- mmr_0.75
- kmeans_k2
- agglo_k3
- dpp_map

Candidate pool:

    Top-20

Final context:

    Top-5

## 4. Generator conditions

Three logical generator conditions were used:

- gemma4-26b
- llama-3.3-70b
- ministral-3-14b

The historical logical ministral-3-14b slot physically used:

    qwen3.6-36b

Generation used temperature 0.

Frozen completed Sprint 2 generation counts:

- PubMedQA: 96,000
- HotpotQA: 710,880
- ASQA: 68,256
- Total: 875,136

## 5. Environment

Dataset-specific environment specifications are preserved under Sprint 1.

Examples:

    sprint1_baseline/pubmedqa/environment/
    sprint1_baseline/hotpotqa/environment/
    sprint1_baseline/asqa/environment/

These directories preserve requirements files and, where available, exact
environment locks from the original execution environments.

Different retrieval components required different environments and hardware.
The project therefore does not claim that one universal Python environment can
reproduce every CPU, GPU, FAISS, Pyserini, and ColBERT component.

## 6. External datasets and large scientific resources

The Git repository intentionally does not contain every large research
artifact.

Examples include:

- full Wikipedia / DPR passage collections
- the full HotpotQA corpus
- retrieval indexes
- FAISS indexes
- ColBERT indexes
- Contriever embedding stores
- model caches
- complete candidate JSONL collections
- materialized generation input collections
- the complete Sprint 2 raw generation collection
- large per-generation NLI outputs
- large intermediate parquet files
- temporary cluster checkpoints

Dataset revisions, corpus identities, sample identities, record counts,
configuration values, manifests, hashes, and provenance records are preserved
in the repository.

Large resources must be reconstructed or downloaded according to the
dataset-specific README and provenance documents.

## 7. Mannheim maKI generation access

Generation used the University of Mannheim maKI service.

Generation code expects the environment variable:

    MAKI_API_KEY

An authorized reviewer must supply their own valid credential.

No API key, password, access token, SSH private key, or other secret is stored
in this repository.

The hosted model provider did not expose immutable model revisions for every
generator condition. Therefore future API calls reproduce the frozen protocol
and configuration, but byte-identical future generated text is not guaranteed.

## 8. Original BWUNICLUSTER execution paths

Some original execution scripts contain paths such as:

    /pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3

These paths are intentionally preserved.

The files under reproducibility/original_final_scripts are byte-for-byte copies
of the scientific execution code used or preserved in the original
BWUNICLUSTER workspace.

They were not rewritten for portability.

When running these scripts on a different machine, a reviewer may need to
recreate the corresponding directory structure or adapt paths in a separate
local working copy.

Such adaptation is not part of the frozen source preserved here.

## 9. Sprint 1 — relevance baselines

Sprint 1 contains baseline retrieval and generation implementation for:

- PubMedQA
- HotpotQA
- ASQA

Each dataset directory includes the relevant combination of:

- README
- scripts
- source modules
- configuration
- environment information
- sample and corpus manifests
- provenance
- baseline artifacts
- generation metadata

The dataset-specific README should be read before executing a Sprint 1 stage.

## 10. Sprint 2 — diversification and generation

The frozen diversification implementation is preserved under:

    sprint2_diversification/src/diversification/

The original diversification implementation includes:

- MMR
- KMeans-based selection
- agglomerative-clustering selection
- DPP greedy MAP selection
- common dispatch utilities

Final dataset-specific producers and generation scripts are preserved under:

    sprint2_diversification/scripts/

The raw generation collection remains outside Git because of its size.

## 11. Sprint 3 — final evaluation

Sprint 3 evaluates the frozen generation collection.

Phase 4 includes:

- correctness
- faithfulness and hallucination
- evidence coverage
- retrieval diversity
- output diversity

Phase 5 includes:

- diversity-accuracy trade-off
- diversity-faithfulness / hallucination analysis
- context-diversity to output-diversity association
- paraphrase robustness

Compact final results and provenance are preserved under:

    sprint3_evaluation/

Additional original execution scripts used for final evaluation are preserved
under:

    reproducibility/original_final_scripts/

These include:

- Phase 4 faithfulness workload construction
- Phase 4 faithfulness workload materialization
- Phase 4 NLI execution
- Phase 4 output-diversity execution
- ASQA governed retrieval metrics
- HotpotQA governed retrieval metrics

## 12. Reproduction levels

There are three practical levels of reproduction.

### A. Integrity verification

This requires only the Git repository.

The reviewer can:

- inspect all committed source code
- inspect frozen manifests and provenance
- inspect compact result summaries
- verify SHA-256 inventories
- inspect the exact additional original final execution scripts

### B. Pipeline reproduction with external scientific resources

The reviewer reconstructs the corresponding:

- datasets
- corpora
- indexes
- embedding stores
- model resources
- generation input collections

using the preserved identities and provenance.

This allows retrieval, diversification, and evaluation stages to be rerun,
subject to sufficient compute and storage.

### C. Full LLM-generation reproduction

This additionally requires:

- authorized Mannheim maKI access
- sufficient CPU/GPU resources
- corresponding materialized retrieval and generation inputs

The same generation protocol can be executed.

Future hosted-model output is not claimed to be byte-identical.

## 13. Scientific provenance versus portability

The handoff prioritizes preservation of the original scientific implementation.

Therefore:

- original scientific Python code is preserved unchanged
- original cluster-specific paths may remain
- large scientific resources remain external
- secrets are excluded
- exact manifests and cryptographic hashes are retained
- completed experiment outputs are represented by frozen summaries and
  provenance where the raw data are too large for Git
- no scientific experiment was rerun during repository packaging

## 14. Recommended review order

A reviewer can inspect the repository in this order:

1. README.md
2. REPRODUCIBILITY.md
3. sprint1_baseline/pubmedqa/README.md
4. sprint1_baseline/hotpotqa/README.md
5. sprint1_baseline/asqa/README.md
6. sprint2_diversification/README.md
7. sprint3_evaluation/README.md
8. reproducibility/original_final_protocols/
9. reproducibility/original_final_scripts/
10. package manifests and SHA-256 inventories

## 15. Scientific status

Scientific experimentation:

    COMPLETE / FROZEN

Repository packaging does not alter the reported scientific results.
