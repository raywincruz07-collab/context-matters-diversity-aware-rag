# Sprint 1 — HotpotQA Baseline

This directory contains the reproducibility package for the **HotpotQA baseline
RAG experiments** in Sprint 1 of the Context Matters project.

The package preserves the scientific contracts, frozen manifests, retrieval
summaries, generation provenance, executed analysis notebook, configuration,
environment references, and the exact implementation used for the completed
baseline experiments.

Large corpora, indexes, caches, materialized context files, and the complete
generation-output collection are intentionally not committed to Git.

## Experiment scope

The baseline evaluates four retrievers:

- BM25
- DPR
- Contriever
- ColBERTv2

Generation was evaluated under five context conditions:

- no retrieved context
- BM25 context
- DPR context
- Contriever context
- ColBERTv2 context

Each condition was evaluated with three logical generator conditions:

- `gemma4-26b`
- `llama-3.3-70b`
- `ministral-3-14b`

The recorded physical model mapping is preserved separately from the logical
experimental condition. In particular:

`ministral-3-14b -> qwen3.6-36b`

This mapping must not be renamed because the logical condition is part of the
frozen experimental design.

## Dataset and evaluation population

Dataset:

`BeIR/hotpotqa`

Canonical corpus:

- documents: **5,233,329**
- dataset revision:
  `a7e8bab212f5a89f9be1bc9b654aa6dfa317f32b`
- scientific corpus SHA256:
  `ac98c20e24668bb886df75382baff8edf3944f294d1fd0dc8609e58f53f739c9`
- document-ID-map SHA256:
  `845e95ee21594d3a0730ad7d12fff12d8a429fbb4de1dffec0fcca38835f1912`
- logical-entry-stream SHA256:
  `efbeeb87500cc3e5f9b69502d4f146161103b6769ff064c262fed84c6bd989f0`
- retrieval serialization:
  `hotpot_retrieval_content_v1`

Official test population:

- queries: **7,405**
- evidence role: `OFFICIAL_TEST_FULL`
- query-manifest SHA256:
  `efedb99b0bf844896611f37d19bebcf710b171a375c8c9fb932c055b7aa16e9c`

No query subsampling is used in this baseline.

## Retrieval protocol

All four retrievers produced a candidate pool of **Top 20** documents for each
of the 7,405 official test queries.

Canonical candidate SHA256 values:

| Retriever | Candidate SHA256 |
|---|---|
| BM25 | `bba4df860997fb09f71faaaaa8dbb0aee7fc2c957279adef14053d632e5494cb` |
| DPR | `4dd232738237a042a1078eb25f6661a32e6e963af89db794c44012feaec6dfe7` |
| Contriever | `7e45be5ddaa9281089870f4166d66346f4bdb9eb6f4f9fa07b13778835ce5dbd` |
| ColBERTv2 | `88c84b6fcc4aef6166151195f50e289a510acce818df10079db123e255518b60` |

The compact candidate-set manifests needed by the generation materialization
stage are stored under:

`retrieval/candidate_sets/`

The canonical retrieval summaries are stored under:

`retrieval/summaries/`

## Generation inputs

The frozen generation package used:

- 7,405 no-context query rows
- 7,405 BM25 context rows
- 7,405 DPR context rows
- 7,405 Contriever context rows
- 7,405 ColBERTv2 context rows
- candidate pool: 20
- selected context `top_k`: 5

The package manifest is stored at:

`generation/input_manifest/package_manifest.json`

Its SHA256 is:

`0b5fb1d3fce913a0641c682e2395d764efbddaf350bb7f7f6b27050b34abb04b`

The large materialized JSONL context files are not committed to Git.

## Generation outputs

The completed baseline contains:

- **15 generation cells**
- **7,405 records per cell**
- **111,075 JSON records total**
- **0 invalid JSON files**

Global recorded statuses:

| Status | Count |
|---|---:|
| `OK` | 109,087 |
| `PARSE_FAILURE` | 435 |
| `TRUNCATED` | 1,535 |
| `UPSTREAM_CONTEXT_MISSING` | 18 |

`PARSE_FAILURE`, `TRUNCATED`, and `UPSTREAM_CONTEXT_MISSING` are preserved
experimental outcomes and must not be silently repaired or discarded.

The 18 `UPSTREAM_CONTEXT_MISSING` records correspond exactly to six governed
Contriever context-missing queries across the three generator conditions.

The complete raw generation-output directory contains approximately 919 MB and
is not committed to Git.

Instead, the package contains a compact cryptographic provenance manifest:

`generation/output_provenance/hotpotqa_generation_outputs_manifest.json`

Manifest SHA256:

`8d386da667de94a2b0fd48051a66b2551aa5249fdf0ba8feffa75bf48e3bed30`

This manifest records per-cell counts, status distributions, logical-to-physical
model mappings, and aggregate hashes derived from every generation-output file.

## Analysis notebook

The completed baseline analysis notebook is:

`notebooks/02_hotpotqa_baseline.ipynb`

The preserved notebook contains:

- 32 cells total
- 16 code cells
- all 16 code cells executed
- preserved outputs

The notebook is treated as the authoritative completed baseline analysis
artifact. It has not been rewritten or re-executed during repository cleanup.

## Environment and model configuration

Generation model bindings:

`configs/maki_model_bindings_v7.json`

Detailed execution-environment provenance:

`environment/EXECUTION_ENVIRONMENT.md`

Included environment references:

- `environment/requirements.txt`
- `environment/requirements-lock-colbert-runpod-cu130.txt`
- `environment/requirements-lock-colbert-local-cpu.txt`

The environment documentation distinguishes the frozen reproduction locks from
the exact preserved runtime evidence and should be read before reproducing GPU
retrieval.

## Scientific provenance

The `provenance/` directory contains the frozen methodological and governance
documents used to establish:

- corpus source and revision
- full-corpus authority
- retrieval text contract
- official test evaluation population
- resource-gate procedure
- final HotpotQA evaluation protocol

These documents originated in the historical project under a later branch name,
but they are included here because their scientific role is the baseline
HotpotQA experiment represented by this Sprint 1 package.

## Repository contents

hotpotqa/
├── README.md
├── artifacts/
├── configs/
├── environment/
├── generation/
├── manifests/
├── notebooks/
├── provenance/
├── retrieval/
├── scripts/
└── src/

The implementation under `scripts/` and `src/` was copied from the completed
scientific workspace without refactoring the experimental logic.

## Large artifacts intentionally excluded from Git

The following completed artifacts remain external because of their size:

- canonical 5.23M-document HotpotQA corpus shards
- retriever indexes
- dense embedding caches
- ColBERT PLAID index
- full Top-20 candidate JSONL files
- large materialized with-context JSONL generation inputs
- 111,075 raw generation-output JSON files

Their scientific identities are preserved through source revisions, manifests,
record counts, SHA256 hashes, retrieval summaries, and generation provenance.

The preserved full RunPod workspace backup is also external to Git.

## Reproducibility chain

The intended traceability chain is:

dataset source
→ canonical corpus manifest
→ official test manifest
→ retriever configuration
→ Top-20 candidates
→ Top-5 materialized context
→ generator/model binding
→ generation outputs
→ baseline analysis notebook

The purpose of this package is to make that chain inspectable without altering
the completed experiment or committing large regenerable artifacts.
