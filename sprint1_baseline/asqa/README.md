# Sprint 1 — ASQA Baseline

This directory contains the reproducibility package for the **ASQA baseline RAG
experiment** in Sprint 1 of the Context Matters project.

The package preserves the frozen ASQA evaluation population, full-corpus
retrieval provenance, exact generation protocol, model bindings, generation
input/output provenance, methodological authority documents, notebooks, and the
scientific implementation used for the completed baseline retrieval and
generation stages.

Large corpora, retrieval indexes, published embedding archives, materialized
generation-input JSONL files, and the complete raw generation-output collection
are intentionally not committed to Git.

## Experiment scope

The completed ASQA Sprint 1 baseline contains four generation conditions:

- WITHOUT_CONTEXT
- BM25 Top-5 context
- DPR Top-5 context
- Contriever Top-5 context

Each condition was evaluated with three logical generator conditions:

- `gemma4-26b`
- `llama-3.3-70b`
- `ministral-3-14b`

The historical logical generator identity is preserved separately from the
physical Mannheim maKI model identity.

The frozen mapping is:

`ministral-3-14b -> qwen3.6-36b`

This mapping must not be renamed because the logical condition is part of the
frozen experimental design.

The completed matrix therefore contains:

`948 questions x 4 conditions x 3 generators = 11,376 generations`

## Dataset and protected evaluation population

Dataset source:

`din0s/asqa`

Frozen dataset revision:

`084060f16b46f3165318f760b2339208b19a0bde`

Evaluation split:

`dev`

Evidence role:

`PROJECT_PROTECTED_FINAL`

Evaluation population:

- questions: **948**
- unique sample IDs: **948**
- no internal repartition of the protected dev split

Physical protected-dev source:

`data/dev-00000-of-00001-58a9a40c6e69f07b.parquet`

Physical SHA256:

`9e017bc3a4409902ec994e87c1d0407c06d4b09342bae982031b6c3ea4daaf5b`

Frozen ordered sample-ID sequence SHA256:

`ca694f4e29ffd8b2c3330d51368a26ac9103b767609b4fd4a4ed587bad4fea30`

The generation inputs and the published ALCE DPR artifact were independently
verified to contain these same 948 IDs in exactly this order.

## Canonical retrieval corpus

Canonical ASQA retrieval uses the standard DPR English Wikipedia passage
collection.

Frozen corpus identity:

- logical corpus: `dpr-wikipedia-2018-12-20`
- snapshot lineage: 2018-12-20
- passages: **21,015,324**
- passage ID column: `id`
- canonical retrieval surface: exact DPR passage body
- corpus policy: **full collection; no subsampling**

Physical DPR corpus archive:

`psgs_w100.tsv.gz`

Physical corpus SHA256:

`c39b020c855a2b5c25ffef3abe4a3b6f9b829ad7dbc14ec3d163d34d7c53ea8d`

Scientific corpus SHA256:

`cd7c9e5bcb8ab748584001d47449ea63b62ed994f938c97e9024916c253cc652`

The detailed frozen corpus provenance is preserved at:

`artifacts/corpus_manifests/asqa_dpr_corpus_source_provenance_v1.json`

## Retrieval baseline

Three ASQA retrievers were completed over the full canonical corpus:

| Retriever | Queries | Stored depth | Retrieved entries | Artifact SHA256 |
|---|---:|---:|---:|---|
| BM25 | 948 | 100 | 94,800 | `ec71626bfcffe6c921f6696a894cefea2729a17a4a5b6c082a5a7b21399bdbee` |
| DPR | 948 | 100 | 94,800 | `221b4a7fc074346096cf6298319feb635256ec12c5cdf10aae528402ee39c252` |
| Contriever | 948 | 100 | 94,800 | `5a9b90fbd5761d75837a9f0c96c13747d139edf253602d8ad805e1a1f97f4e57` |

### BM25

BM25 uses the Pyserini prebuilt `wikipedia-dpr` Lucene index over the complete
DPR Wikipedia collection.

Frozen parameters recorded in the completed artifact:

- index: `wikipedia-dpr`
- `k1 = 0.9`
- `b = 0.4`
- stored candidate depth: Top-100

### DPR

DPR uses the published ALCE ASQA DPR Top-100 artifact.

Published artifact:

`asqa_eval_dpr_top100.json`

Its compatibility with the protected ASQA dev population and canonical DPR
Wikipedia passage universe was independently validated.

Detailed DPR provenance is preserved at:

`artifacts/retrieval_manifests/asqa/dpr_alce_top100_provenance_v1.json`

### Contriever

Contriever uses Meta-published DPR-Wikipedia Contriever passage embeddings.

Frozen query model:

`facebook/contriever`

Frozen model revision:

`2bd46a25019aeea091fd42d1f0fd4801675cf699`

Recorded score semantics:

`exact inner product`

## ColBERTv2 scope

ColBERTv2 was **not completed for ASQA**.

This is not treated as an undocumented missing experiment.

Explicit supervisor guidance received on 2026-09-03 required:

- use the full 21,015,324-passage DPR Wikipedia corpus;
- do not shrink or subsample the corpus to make ColBERTv2 feasible;
- if the full-corpus ColBERTv2 build could not be completed in time, omit
  ColBERTv2 from ASQA;
- report ColBERTv2 on HotpotQA instead;
- retain BM25, DPR, and Contriever on the full ASQA corpus.

The governing clarification is preserved at:

`provenance/ASQA_SUPERVISOR_CLARIFICATION_2026-09-03.md`

Therefore the completed ASQA Sprint 1 baseline matrix contains
WITHOUT_CONTEXT, BM25, DPR, and Contriever only.

## Generation inputs

The materialized generation package contains four files, each covering the same
948 protected ASQA questions.

| Condition | Rows | Context passages per row | File SHA256 |
|---|---:|---:|---|
| WITHOUT_CONTEXT | 948 | 0 | `c58819096a3196a94e01a530763f65dca3b668c1a7c36cbeabed912866175513` |
| BM25 | 948 | 5 | `187e2e5347626441488df81618720b015ac04993d27420f47fab4bec9972d723` |
| DPR | 948 | 5 | `b09cc3507aa95bb0f2cd889d05c95a940e024cb80d87541979ce7fc7ac7f8927` |
| Contriever | 948 | 5 | `266c32bef94f00535506be4345dbc575ee5b6cec2db5b0c16c667a068d0f70dc` |

The three context-bearing files contain exactly five canonical passage bodies
for every query.

The materialized JSONL files are not committed to Git.

Their complete cryptographic provenance is preserved at:

`generation/input_provenance/asqa_generation_inputs_manifest.json`

Manifest SHA256:

`532735e79235e27e804d8dc0ef84bfee2b489f561135c6b18383f8a34d7cb524`

Each context input also records the SHA256 of the exact upstream retrieval
artifact, providing a direct retrieval-to-generation provenance link.

## Generation protocol

Canonical generation is implemented in:

`scripts/run_asqa_sprint1_generation.py`

Frozen generation settings include:

- 948 rows per condition
- 4 conditions
- 3 logical generator configurations
- 11,376 expected requests
- selected context `top_k = 5`
- temperature: 0
- `max_tokens = 512`
- `n = 1`
- canonical concurrency: 16

Generation model bindings are stored at:

`configs/sprint3/maki_model_bindings_v7.json`

Bindings SHA256:

`3827ceaf993eaa061e0b87645cf5b36bc15c0e093b1712332b1fcbb8a06264b3`

The bindings preserve the three logical-to-physical mappings:

| Logical experiment ID | Physical Mannheim maKI model |
|---|---|
| `gemma4-26b` | `gemma4-26b` |
| `llama-3.3-70b` | `llama-3.3-70b` |
| `ministral-3-14b` | `qwen3.6-36b` |

The provider did not expose immutable model revision identifiers, so future API
calls reproduce the frozen protocol but are not claimed to produce byte-identical
future model text.

## Generation outputs

The completed ASQA Sprint 1 generation collection contains:

- **12 generation cells**
- **948 JSON records per cell**
- **11,376 JSON records total**
- **0 invalid JSON files**

Global recorded statuses:

| Status | Count |
|---|---:|
| `OK` | 11,334 |
| `PARSE_FAILURE` | 3 |
| `TRUNCATED` | 39 |

`PARSE_FAILURE` and `TRUNCATED` are preserved experimental outcomes and must not
be silently repaired, discarded, or converted into successful generations.

Every logical generator condition contains exactly 3,792 records.

Recorded mappings:

- `gemma4-26b -> gemma4-26b`: 3,792
- `llama-3.3-70b -> llama-3.3-70b`: 3,792
- `ministral-3-14b -> qwen3.6-36b`: 3,792

The complete raw generation-output directory is intentionally external to Git.

Its cryptographic provenance is preserved at:

`generation/output_provenance/asqa_generation_outputs_manifest.json`

Manifest SHA256:

`4f7c78e4fbd8da04e9085159e9077528d38ea25be085789c374cb0799c01ee28`

The manifest records per-cell counts, status distributions, logical-to-physical
model mappings, and aggregate hashes derived from the completed raw output
files.

## Retrieval provenance manifest

The compact cross-retriever provenance manifest is:

`retrieval/provenance/asqa_retrieval_provenance_manifest.json`

Manifest SHA256:

`d694c9c99a3f6889808337b46a5bb7f5f051d12ae7b41097894df5ec678e911a`

It records:

- protected evaluation population
- canonical corpus identity
- BM25 artifact identity
- DPR artifact identity
- Contriever artifact identity
- query and retrieval counts
- full-corpus policy
- ColBERTv2 supervisor-approved omission

## Analysis notebooks

Baseline notebook:

`notebooks/03_asqa_baseline.ipynb`

Dataset profile notebook:

`notebooks/datasets/03_asqa_dataset.ipynb`

Both notebooks are preserved byte-for-byte from the completed scientific
workspace.

The baseline notebook contains:

- 25 cells
- 9 code cells
- 0 executed code cells
- 0 preserved outputs

It is therefore a frozen baseline protocol/reporting notebook rather than an
executed final-results notebook.

It has not been re-executed during repository cleanup.

## Metric status

The Sprint 1 retrieval and generation stages documented in this package are
complete for WITHOUT_CONTEXT, BM25, DPR, and Contriever.

The following ASQA evaluation layers remain pending and are **not fabricated or
approximated in this package**:

- official ASQA QA-F1 / Disambig-F1 execution
- answer-side alias coverage
- S-Recall@5
- alpha-nDCG@5

The official ASQA correctness metric is not replaced here by an ad-hoc token
metric, ROUGE score, alias heuristic, or LLM judge.

Likewise, later diversification-specific retrieval evaluation is outside this
Sprint 1 package.

## Scientific provenance

The `provenance/` directory contains the frozen documents relevant to the
baseline corpus and protected evaluation design:

- `ASQA_CORPUS_AUTHORITY_NOTE.md`
- `ASQA_INTERNAL_PARTITION_PROTOCOL.md`
- `ASQA_SAMPLE_ID_SOURCE_AMENDMENT_01.md`
- `ASQA_SUPERVISOR_CLARIFICATION_2026-09-03.md`

These documents originated in the historical project under a later branch name,
but their scientific role includes the baseline ASQA experiment represented in
this Sprint 1 package.

Later DEVELOPMENT/SELECTION Top-50 execution documents and the later
retrieval-diversity metric protocol are intentionally excluded from this
baseline package.

## Repository contents

ASQA package structure:

    asqa/
    ├── README.md
    ├── artifacts/
    ├── configs/
    ├── environment/
    ├── generation/
    ├── notebooks/
    ├── provenance/
    ├── retrieval/
    ├── scripts/
    └── src/

The implementation under `scripts/` and `src/` was copied from the completed
scientific workspace without refactoring or modifying the experimental logic.

## Large artifacts intentionally excluded from Git

The following scientific resources remain external because of their size:

- full 21,015,324-passage DPR Wikipedia corpus
- Pyserini Lucene index
- DPR FAISS/index resources
- published Contriever passage embeddings
- downloaded Hugging Face caches
- materialized ASQA generation-input JSONL files
- published ALCE DPR Top-100 JSON artifact
- full BM25 and Contriever Top-100 JSONL artifacts
- 11,376 raw generation-output JSON files
- unfinished/full-corpus ColBERT indexing resources

Their identities are preserved through frozen source revisions, record counts,
SHA256 hashes, corpus provenance, retrieval provenance, and generation
provenance.

## Reproducibility chain

The intended Sprint 1 traceability chain is:

    frozen ASQA source revision
    -> protected official dev population
    -> full DPR Wikipedia corpus
    -> BM25 / DPR / Contriever Top-100 retrieval artifacts
    -> exact Top-5 materialized contexts
    -> frozen maKI generator bindings
    -> 11,376 generation outputs
    -> baseline notebook / later official evaluation

This package is designed so that a reviewer can inspect that chain without
rerunning expensive retrieval or LLM generation.

## Verification without LLM access

A reviewer does not need Mannheim maKI access to inspect the completed Sprint 1
artifacts preserved here.

After `SHA256SUMS` is generated, package integrity can be checked from the ASQA
directory with:

    sha256sum -c SHA256SUMS

The compact provenance manifests can also be inspected directly:

    cat generation/input_provenance/asqa_generation_inputs_manifest.json
    cat generation/output_provenance/asqa_generation_outputs_manifest.json
    cat retrieval/provenance/asqa_retrieval_provenance_manifest.json

## Environment

The historical project dependency specification is preserved at:

`environment/requirements.txt`

Generation requires authorized Mannheim maKI access through the environment
variable:

`MAKI_API_KEY`

Credentials must never be committed to the repository.

Retrieval reproduction additionally requires the corresponding published
indexes, embedding artifacts, upstream datasets, and sufficient compute.

ColBERTv2 reproduction for ASQA is not claimed because the full-corpus ASQA
ColBERT experiment was intentionally not completed under the supervisor-approved
contingency.

## Scope of this directory

This directory contains only the fresh **Sprint 1 ASQA relevance baseline**.

It intentionally excludes:

- diversification experiments
- DEVELOPMENT/SELECTION Top-50 sensitivity artifacts
- candidate-pool sensitivity experiments
- later retrieval-diversity metric outputs
- temporary staging files
- caches
- downloaded indexes and embeddings
- debugging logs
- infrastructure-operation logs

These exclusions keep the package focused on the artifacts required to
understand, verify, and reproduce the completed ASQA baseline retrieval and
generation stages without overstating evaluation work that remains pending.
