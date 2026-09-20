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

---

## Professor Quick Start

This package is designed so that the completed Sprint 1 PubMedQA experiment can be inspected and verified without rerunning the LLMs.

### Experimental Matrix

Five context conditions were evaluated:

- WITHOUT_CONTEXT
- BM25
- DPR
- Contriever
- ColBERTv2

Each condition was evaluated with three logical generator configurations:

- `llama-3.3-70b`
- `gemma4-26b`
- `ministral-3-14b`

The historical logical identifier `ministral-3-14b` was bound to the physical Mannheim Maki model `qwen3.6-36b`.

Therefore the experiment contains:

```text
5 conditions × 3 generators × 1,000 questions = 15,000 generations
```

The complete experiment matrix can be inspected without making API calls:

```bash
PYTHONPATH=src python scripts/run_pubmedqa_generation.py matrix
```

Expected structure:

```text
15 experiment blocks
1,000 questions per block
15,000 total generation records
```

---

## Final Answer Accuracy

Accuracy is calculated over successfully measurable generations.

| Generator | No context | BM25 | DPR | Contriever | ColBERTv2 |
|---|---:|---:|---:|---:|---:|
| Llama 3.3 70B | 0.571 | 0.656 | 0.579 | 0.691 | 0.708 |
| Gemma 4 26B | 0.460 | 0.404 | 0.319 | 0.461 | 0.496 |
| Qwen3.6 36B | 0.559 | 0.596 | 0.497 | 0.653 | 0.673 |

Detailed values and measured sample counts are stored in:

```text
analysis/decision_accuracy_summary.csv
```

Measured sample counts differ slightly from 1,000 for a few Gemma and Qwen conditions because parsing failures or truncation are retained rather than silently removed.

---

## Paired Effect of Retrieved Context

Each retrieval condition was paired against the same generator answering the same question without retrieved context.

| Generator | Retriever | Accuracy difference | 95% CI | Interpretation |
|---|---|---:|---|---|
| Llama 3.3 70B | BM25 | +0.085 | [0.058, 0.112] | Positive effect |
| Llama 3.3 70B | DPR | +0.008 | [-0.024, 0.040] | CI includes zero |
| Llama 3.3 70B | Contriever | +0.120 | [0.092, 0.149] | Positive effect |
| Llama 3.3 70B | ColBERTv2 | +0.137 | [0.108, 0.166] | Positive effect |
| Gemma 4 26B | BM25 | -0.054 | [-0.093, -0.016] | Negative effect |
| Gemma 4 26B | DPR | -0.141 | [-0.183, -0.100] | Negative effect |
| Gemma 4 26B | Contriever | +0.001 | [-0.037, 0.041] | CI includes zero |
| Gemma 4 26B | ColBERTv2 | +0.036 | [-0.003, 0.076] | CI includes zero |
| Qwen3.6 36B | BM25 | +0.037 | [0.003, 0.071] | Positive effect |
| Qwen3.6 36B | DPR | -0.061 | [-0.099, -0.022] | Negative effect |
| Qwen3.6 36B | Contriever | +0.094 | [0.062, 0.127] | Positive effect |
| Qwen3.6 36B | ColBERTv2 | +0.114 | [0.082, 0.146] | Positive effect |

Full paired results are stored in:

```text
analysis/paired_context_effects.csv
```

These results demonstrate that retrieved context can help or hurt depending on the retriever-generator combination.

---

## Retrieval Quality

The baseline retrieval results are:

| Retriever | Recall@5 | MRR@5 | Candidate Recall@20 |
|---|---:|---:|---:|
| BM25 | 0.599 | 0.912 | 0.702 |
| DPR | 0.325 | 0.579 | 0.465 |
| Contriever | 0.740 | 0.972 | 0.837 |
| ColBERTv2 | 0.738 | 0.976 | 0.825 |

Detailed retrieval and downstream-answer results are stored in:

```text
analysis/retrieval_answer_summary.csv
```

The baseline retrieval metric contract uses:

- positive-gold-section Recall@5
- positive-gold-section MRR@5
- positive-gold-section candidate Recall@20

---

## Generation Completion Status

The canonical experiment contains exactly 15,000 generation records:

| Status | Count |
|---|---:|
| OK | 14,966 |
| PARSE_FAILURE | 33 |
| TRUNCATED | 1 |
| REFUSAL | 0 |
| ERROR | 0 |
| Total | 15,000 |

Detailed status information is stored in:

```text
analysis/generation_status_summary.csv
```

The parse failures and truncation are preserved as part of the scientific record.

---

## Generator Bindings

Frozen model bindings are stored in:

```text
configs/maki_model_bindings_v7.json
```

The three experiment identities are:

| Logical experiment ID | Physical Mannheim Maki model | Seed |
|---|---|---:|
| `llama-3.3-70b` | `llama-3.3-70b` | 20260823 |
| `gemma4-26b` | `gemma4-26b` | 20260823 |
| `ministral-3-14b` | `qwen3.6-36b` | 20260823 |

The logical identifier is preserved because it is part of the frozen historical experiment configuration.

The provider did not expose immutable model revision identifiers. Therefore the original outputs can be verified exactly, but future API calls are not claimed to produce byte-identical text.

---

## Repeatability Gate

The frozen repeatability artifact is:

```text
provenance/repeatability_gate_v6.json
```

Before the main experiment, each primary generator configuration was tested using 20 prompts with three repeated calls per prompt.

| Logical model | Identical prompts | Tested prompts | Result |
|---|---:|---:|---|
| `llama-3.3-70b` | 20 | 20 | PASS |
| `gemma4-26b` | 20 | 20 | PASS |
| `ministral-3-14b` → `qwen3.6-36b` | 20 | 20 | PASS |

The required threshold was 19/20 identical prompts.

This establishes repeatability at the time of the experiment but does not replace an immutable provider-side model revision.

---

## Canonical Retrieval Artifacts

Each retriever has a frozen Top-20 candidate artifact for every PubMedQA question.

```text
retrieval/candidates/
├── bm25/          1,000 files
├── dpr/           1,000 files
├── contriever/    1,000 files
└── colbertv2/     1,000 files
```

Canonical candidate-set inventories are stored in:

```text
retrieval/candidate_sets/
├── bm25_candidate_set_v1.json
├── dpr_candidate_set_v1.json
├── contriever_candidate_set_v1.json
└── colbertv2_candidate_set_v1.json
```

All 4,000 candidate artifacts were verified against the artifact IDs recorded in their canonical inventories.

Only the canonical Sprint 1 Top-20 candidate artifacts are included here.

Later Top-50 sensitivity artifacts are intentionally excluded.

---

## Exact Contexts Used for Generation

For every retriever and every question, the exact selected Top-5 contexts supplied to the generator are preserved.

```text
selected_contexts/
├── bm25/          1,000 files
├── dpr/           1,000 files
├── contriever/    1,000 files
└── colbertv2/     1,000 files
```

Canonical selected-context inventories are:

```text
selected_contexts/bm25_selected_context_set_v1.json
selected_contexts/dpr_selected_context_set_v1.json
selected_contexts/contriever_selected_context_set_v1.json
selected_contexts/colbertv2_selected_context_set_v1.json
```

All 4,000 selected-context artifacts were verified against their canonical inventories.

This allows a reviewer to inspect exactly what evidence was supplied to each retrieval-augmented generation condition.

---

## Canonical Raw Generation Archive

The complete frozen generation package is:

```text
raw_generation/pubmedqa_sprint1_canonical_raw_v1.tar.gz
```

Canonical SHA256:

```text
cab27f82ed9153e3a7e8e65400ff993fff5c00e0a18bf6a91aa16afc7ef99d33
```

Verify it with:

```bash
cd raw_generation
sha256sum -c pubmedqa_sprint1_canonical_raw_v1.tar.gz.sha256
```

The archive contains:

- 15 experiment inventories
- 15,000 generation samples
- the exact raw outputs used for the final analysis

---

## Governed BM25 + Llama Replacement

The original governed BM25 + Llama run did not complete because of infrastructure failure.

Original run:

```text
run-sprint1-pubmedqa-s1-with-context-bm25-none-llama-3-3-70b-ffb0e1228071b46cb867d649
```

Canonical replacement:

```text
run-sprint1-pubmedqa-s1-with-context-bm25-none-llama-3-3-70b-06239dda58f6febfd0be97cd
```

The replacement completed all 1,000 samples and is the BM25 + Llama block contained in the canonical raw archive.

Historical provenance remains available in:

```text
provenance/pubmedqa_run_registry_v1.jsonl
```

The unchanged runtime uses its operational registry under:

```text
artifacts/run_registry/
```

The operational registry included in this organized package begins with a clean canonical registry header for future executions.

---

## Package Layout

```text
pubmedqa/
├── README.md
├── SHA256SUMS
├── analysis/
├── artifacts/
│   ├── corpus_manifests/
│   ├── run_registry/
│   └── sample_manifests/
├── configs/
├── environment/
├── manifests/
├── notebooks/
├── provenance/
├── raw_generation/
├── retrieval/
│   ├── candidate_sets/
│   ├── candidates/
│   │   ├── bm25/
│   │   ├── dpr/
│   │   ├── contriever/
│   │   └── colbertv2/
│   └── retrieval metric outputs
├── scripts/
├── selected_contexts/
│   ├── bm25/
│   ├── dpr/
│   ├── contriever/
│   └── colbertv2/
└── src/
```

---

## Verification Without LLM Access

A reviewer does not need Mannheim Maki access to inspect or verify the completed experiment.

### Verify package integrity

From:

```text
sprint1_baseline/pubmedqa/
```

run:

```bash
sha256sum -c SHA256SUMS
```

### Verify the canonical generation archive

```bash
cd raw_generation
sha256sum -c pubmedqa_sprint1_canonical_raw_v1.tar.gz.sha256
cd ..
```

### Inspect the experiment matrix

```bash
PYTHONPATH=src python scripts/run_pubmedqa_generation.py matrix
```

This command does not make model requests.

### Inspect final analysis tables

```bash
cat analysis/decision_accuracy_summary.csv
cat analysis/paired_context_effects.csv
cat analysis/retrieval_answer_summary.csv
cat analysis/generation_status_summary.csv
```

---

## Environment Setup

### Standard CPU Environment

Create a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements-local-cpu.txt
export PYTHONPATH=src
```

The fully pinned CPU environment is:

```text
environment/requirements-lock-local-cpu.txt
```

### ColBERT CPU Environment

The isolated ColBERT CPU lock is:

```text
environment/requirements-lock-colbert-local-cpu.txt
```

This environment records validated CPU compatibility for the pinned ColBERT checkpoint.

### CUDA / RunPod Environment Records

CUDA 13.0 environment files are retained in:

```text
environment/requirements-lock-runpod-cu130.txt
environment/requirements-lock-colbert-runpod-cu130.txt
```

The ColBERT CUDA lock explicitly records that CPU compatibility was validated while complete RTX 5090 runtime validation was still pending.

It should therefore not be interpreted as a claim that every ColBERT CUDA environment has been independently revalidated.

---

## Retrieval Reproduction

Retrieval entry points are:

```text
scripts/run_pubmedqa_bm25_candidates.py
scripts/run_pubmedqa_dpr_candidates.py
scripts/run_pubmedqa_contriever_candidates.py
scripts/run_pubmedqa_colbert_candidates.py
scripts/materialize_pubmedqa_selected_contexts.py
scripts/run_pubmedqa_governed_retrieval_metrics.py
```

After installing the appropriate environment, CLI options can be inspected with:

```bash
PYTHONPATH=src python scripts/run_pubmedqa_bm25_candidates.py --help
PYTHONPATH=src python scripts/run_pubmedqa_dpr_candidates.py --help
PYTHONPATH=src python scripts/run_pubmedqa_contriever_candidates.py --help
PYTHONPATH=src python scripts/run_pubmedqa_colbert_candidates.py --help
PYTHONPATH=src python scripts/materialize_pubmedqa_selected_contexts.py --help
```

BM25 can be reproduced on CPU.

Dense retrieval and ColBERT require their corresponding model dependencies and may require substantially more compute.

Because the canonical candidate artifacts are included, a reviewer does not need to regenerate retrieval merely to inspect the scientific results.

The governed retrieval metrics entry point is:

```bash
PYTHONPATH=src python scripts/run_pubmedqa_governed_retrieval_metrics.py
```

---

## Generation Reproduction

Generation uses Mannheim University's Maki service.

Authorized users must provide the API credential through:

```text
MAKI_API_KEY
```

For example:

```bash
export MAKI_API_KEY="YOUR_AUTHORIZED_KEY"
```

The credential must never be committed to the repository.

Frozen generator configuration:

```text
configs/maki_model_bindings_v7.json
```

Frozen repeatability gate:

```text
provenance/repeatability_gate_v6.json
```

The generation CLI requires an explicit acknowledgement before any model requests are made:

```text
I_UNDERSTAND_THIS_MAKES_MODEL_REQUESTS
```

The safest first command is:

```bash
PYTHONPATH=src python scripts/run_pubmedqa_generation.py matrix
```

This performs no external model calls.

Example no-context Llama generation command:

```bash
PYTHONPATH=src python scripts/run_pubmedqa_generation.py run \
  --llm llama-3.3-70b \
  --context-mode without_context \
  --model-bindings configs/maki_model_bindings_v7.json \
  --repeatability-gate provenance/repeatability_gate_v6.json \
  --confirm-api-calls I_UNDERSTAND_THIS_MAKES_MODEL_REQUESTS
```

Full generation reproduction requires authorized Mannheim Maki access as well as the required dataset/model resources.

Because Mannheim Maki did not expose immutable provider-side model revisions, a future API rerun reproduces the frozen protocol but is not claimed to produce byte-identical future text.

The exact experiment outputs are therefore preserved in the canonical raw-generation archive.

---

## Final Analysis Files

Primary final outputs are:

```text
analysis/decision_accuracy_summary.csv
analysis/decision_class_recall.csv
analysis/generation_correctness_per_query.parquet
analysis/generation_status_summary.csv
analysis/paired_context_effects.csv
analysis/retrieval_answer_summary.csv
```

The analysis notebook is:

```text
notebooks/01_pubmedqa_baseline.ipynb
```

---

## Reproducibility Levels

This package supports three distinct levels of reproducibility.

### Level 1 — Exact Verification

No external LLM API is required.

A reviewer can verify:

- frozen dataset manifests
- corpus identity
- candidate-set inventories
- 4,000 canonical candidate artifacts
- selected-context inventories
- 4,000 exact selected-context artifacts
- canonical raw generation archive
- generation counts and statuses
- final result tables
- file checksums

### Level 2 — Retrieval and Evaluation Reproduction

The supplied scientific scripts and environment files allow retrieval and evaluation reproduction subject to:

- upstream dataset availability
- model/checkpoint availability
- required hardware
- required Python dependencies

### Level 3 — Full Generation Reproduction

Full generation additionally requires:

- authorized Mannheim Maki access
- `MAKI_API_KEY`
- frozen model bindings
- frozen repeatability gate
- required dataset/model caches

The protocol is reproducible, while exact future model-text equality is not guaranteed because immutable provider-side model revisions were unavailable.

---

## Scope of This Directory

This directory contains only the fresh **Sprint 1 PubMedQA baseline**.

It intentionally excludes:

- later diversification experiments
- Top-50 sensitivity experiments
- temporary staging artifacts
- temporary caches
- downloaded model indexes
- debugging logs
- infrastructure-operation logs
- duplicate generation attempts
- unrelated historical PubMedQA submissions

These exclusions keep the Sprint 1 package focused on the artifacts required to understand, verify, and reproduce the baseline experiment.
