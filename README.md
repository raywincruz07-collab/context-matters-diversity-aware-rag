# CONTEXT MATTERS — Evaluating Diversity-Aware Retrieval for RAG

This repository contains the reproducibility package for the University of
Mannheim team project:

**CONTEXT MATTERS — Evaluating Diversity-Aware Retrieval for RAG**

The project studies a central question:

> Does increasing the diversity of retrieved context help an LLM answer
> questions better, or can additional diversity introduce irrelevant evidence
> and reduce answer quality?

The repository is organized by experimental stage and preserves the completed
scientific pipeline without committing very large corpora, retrieval indexes,
model caches, raw generation collections, or large per-example evaluation
artifacts.

---

## Research pipeline

The final pipeline is:

```text
question
   ↓
retriever
   ↓
Top-20 candidate passages
   ↓
diversification
   ↓
Top-5 selected passages
   ↓
LLM generation
   ↓
answer
   ↓
Phase 4 evaluation
   ↓
Phase 5 cross-metric analysis
```

The central experimental intervention occurs between retrieval and generation:
the same relevance-oriented Top-20 candidate pool is transformed into different
Top-5 context sets.

---

## Datasets

Three QA datasets are used:

| Dataset | Role |
|---|---|
| PubMedQA | Biomedical question answering |
| HotpotQA | Multi-hop question answering |
| ASQA | Ambiguous / long-form question answering |

---

## Retrievers

The project evaluates four retrieval families where feasible:

- **BM25** — lexical sparse retrieval
- **DPR** — supervised dense retrieval
- **Contriever** — dense semantic retrieval
- **ColBERTv2** — late-interaction retrieval

Final retriever coverage:

| Dataset | BM25 | DPR | Contriever | ColBERTv2 |
|---|:---:|:---:|:---:|:---:|
| PubMedQA | ✓ | ✓ | ✓ | ✓ |
| HotpotQA | ✓ | ✓ | ✓ | ✓ |
| ASQA | ✓ | ✓ | ✓ | — |

ASQA ColBERTv2 is intentionally absent from the final matrix.

Under supervisor guidance, ASQA retrieval had to use the full canonical
Wikipedia corpus rather than a reduced corpus. Full-corpus ColBERTv2 could not
be completed within the available resource/time constraints, so it was omitted
rather than replaced with a scientifically weaker subsampled experiment.

---

## Diversification methods

The frozen Sprint 2 experiment contains eight context-selection conditions:

- `none`
- `mmr_0`
- `mmr_0.25`
- `mmr_0.5`
- `mmr_0.75`
- `kmeans_k2`
- `agglo_k3`
- `dpp_map`

The baseline `none` condition preserves the relevance-ranked Top-5 passages.

The remaining methods alter the semantic composition of the final Top-5
context.

### MMR

Maximal Marginal Relevance balances relevance against redundancy:

```text
MMR(i | S)
=
λ × relevance(i)
-
(1 - λ) × max similarity(i, selected passages)
```

Lower lambda values emphasize diversity more strongly.

### KMeans

Candidate passages are clustered in embedding space and selected to increase
semantic coverage across groups.

### Agglomerative clustering

Candidate passages are hierarchically grouped before representative passages
are selected.

### DPP

A Determinantal Point Process objective rewards sets that jointly combine
quality and diversity.

The project uses a greedy MAP approximation.

---

## Context size

The final design is fixed at:

- candidate pool: **Top-20**
- final context: **Top-5**

The number of passages supplied to the generator therefore remains constant
across diversification conditions.

---

## Generators

Three frozen logical generator conditions were used:

- `gemma4-26b`
- `llama-3.3-70b`
- `ministral-3-14b`

The physical model used for the historical `ministral-3-14b` logical slot was:

```text
qwen3.6-36b
```

The historical logical identifier is retained in reproducibility artifacts,
while final reporting may display the physical model name **Qwen 3.6 36B**.

Generation used deterministic settings with temperature 0.

Frozen completed generation counts:

| Dataset | Generations |
|---|---:|
| PubMedQA | 96,000 |
| HotpotQA | 710,880 |
| ASQA | 68,256 |
| **Total** | **875,136** |

The complete raw generation collection is intentionally external to Git.

---

# Experimental stages

## Sprint 1 — Relevance baselines

Directory:

```text
sprint1_baseline/
```

Sprint 1 establishes the baseline retrieval and generation pipeline for:

- PubMedQA
- HotpotQA
- ASQA

The package preserves dataset and corpus provenance, retrieval configuration,
generation protocol, scientific implementation, manifests, hashes, and
baseline artifacts.

Large corpora, indexes, caches, and complete generation collections remain
outside Git.

---

## Sprint 2 — Diversity-aware context selection

Directory:

```text
sprint2_diversification/
```

Sprint 2 applies the predefined diversification methods to the frozen candidate
pools and produces the final Top-5 context used for generation.

It preserves:

- exact diversification implementation;
- final diversification producers;
- generation producers;
- model bindings;
- compact retrieval summaries;
- generation-package metadata;
- frozen generation/retrieval provenance;
- SHA-256 package inventory.

No scientific experiment was rerun during repository packaging.

---

## Sprint 3 — Final evaluation

Directory:

```text
sprint3_evaluation/
```

Sprint 3 evaluates the completed generation collection.

### Phase 4

The final metric layer contains:

1. **4.1 Correctness**
2. **4.2 Faithfulness and hallucination**
3. **4.3 Evidence coverage**
4. **4.4 Retrieval diversity**
5. **4.5 Output diversity**

### Phase 5

The cross-metric analysis contains:

1. **5.1 Diversity–accuracy trade-off**
2. **5.2 Diversity–faithfulness / hallucination**
3. **5.3 Context-diversity → output-diversity association**
4. **5.4 Robustness under paraphrase**

---

# Evaluation metrics

## Correctness

Correctness remains task-specific:

- **PubMedQA:** decision accuracy
- **HotpotQA:** token F1, with EM as a secondary measure
- **ASQA:** QA and STR metric family

This avoids forcing one metric onto datasets with different answer structures.

---

## Faithfulness and hallucination

Answer sentences or claims are evaluated against the five retrieved passages
using the frozen NLI protocol.

A claim is supported when at least one retrieved passage entails it.

```text
faithfulness
=
supported claims / total claims
```

An answer is flagged for hallucination when at least one claim is unsupported.

These metrics are evaluated only for context-bearing generations.

---

## Coverage

Coverage is dataset-specific:

- **PubMedQA:** gold-document Recall@5 and MRR@5
- **HotpotQA:** supporting-document Recall@5 and both-supporting-documents rate
- **ASQA:** STR-based coverage

These metrics measure whether the selected Top-5 context contains evidence
required by the task.

---

## Retrieval diversity

For the final five passages, semantic diversity is measured as mean pairwise
cosine distance:

```text
diversity(i,j) = 1 - cosine(e_i, e_j)
```

The final retrieval-diversity score averages the ten unique passage pairs among
five passages.

---

## Output diversity

ASQA output diversity applies the same semantic-distance idea to answer
sentences.

Answers with fewer than two sentences are ineligible and retain a null value
rather than being assigned artificial zero diversity.

---

# Final cross-metric findings

## Diversity versus correctness

The tested diversification operating points generally expose a
**relevance–diversity trade-off**.

Increasing semantic context diversity does not produce a consistent
correctness improvement across datasets, retrievers, generators, and
conditions.

---

## Diversity versus faithfulness

Faithfulness and hallucination effects vary by experimental configuration.

There is no consistent cross-dataset evidence that greater retrieval diversity
systematically increases faithfulness or systematically reduces hallucination.

---

## Context diversity versus output diversity

For ASQA, the frozen overall cross-sectional association is weakly positive:

- Pearson ≈ **+0.129**
- Spearman ≈ **+0.122**

The matched change analysis is essentially flat:

- paired-delta Pearson ≈ **+0.0186**
- paired-delta Spearman ≈ **+0.0220**

Increasing context diversity therefore did not meaningfully translate into
greater answer diversity in the paired intervention comparison.

These associations are descriptive and are not interpreted causally.

---

## Robustness under paraphrase

Phase 5.4 uses:

- **30 human-validated ASQA paraphrase pairs**
- BM25
- DPR
- Contriever
- baseline `none`
- MMR-0.5

Robustness is measured using Jaccard overlap between document sets retrieved
for the original and paraphrased question.

Higher Jaccard means greater retrieval invariance to wording changes.

It does **not** measure retrieval quality.

Observed effects are retriever-dependent.

MMR-0.5 does not consistently improve paraphrase invariance.

---

# Main conclusion

The project supports the following conclusion:

> **Diversification changes the semantic evidence presented to the LLM, but
> greater diversity is not inherently beneficial. The tested operating points
> generally introduce a relevance–diversity trade-off rather than a consistent
> correctness gain.**

The experiments also show that:

- faithfulness and hallucination effects are configuration-dependent;
- context diversity does not automatically propagate to output diversity;
- paraphrase robustness is retriever-dependent.

The practical lesson is:

> **Balance relevance and diversity rather than maximizing diversity alone.**

The results do **not** justify stronger claims such as:

- diversity is always harmful;
- diversity always causes hallucination;
- MMR is universally poor;
- one diversification method is universally best.

---

# Repository structure

```text
.
├── LICENSE
├── README.md
├── sprint1_baseline/
│   ├── pubmedqa/
│   ├── hotpotqa/
│   └── asqa/
├── sprint2_diversification/
└── sprint3_evaluation/
```

Sprint 2 and Sprint 3 contain SHA-256 integrity inventories.

Sprint 1 contains dataset-specific integrity and provenance records.

---

# Reproducibility and provenance

The repository favors scientific provenance over uploading large binary
research outputs.

Preserved information includes:

- source code used in the completed scientific workspace;
- dataset and corpus identities;
- frozen sample populations;
- retrieval parameters;
- model bindings;
- experimental conditions;
- compact result summaries;
- status distributions;
- manifests;
- cryptographic hashes;
- final freeze records.

Large resources intentionally kept outside Git include:

- full corpora;
- retrieval indexes;
- model and embedding caches;
- full candidate JSONL collections;
- 875,136 raw generation records;
- large per-generation metric outputs;
- NLI workloads and large result files;
- joined analysis parquet files;
- full figure archive;
- temporary cluster checkpoints.

Their scientific identity is preserved through manifests and hashes where
applicable.

---

# Integrity verification

Sprint 2:

```bash
cd sprint2_diversification
sha256sum -c SHA256SUMS
```

Sprint 3:

```bash
cd sprint3_evaluation
sha256sum -c SHA256SUMS
```

---

# Project status

Scientific experimentation:

**COMPLETE / FROZEN**

Remaining project deliverables outside this repository handoff:

- final written report;
- final presentation;
- individual reflection, where required.

No additional retrieval, generation, diversification, or evaluation
experiment is required for the frozen project.
