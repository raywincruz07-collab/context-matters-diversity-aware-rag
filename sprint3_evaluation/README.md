# Sprint 3 — Final Evaluation and Cross-Metric Analysis

This directory is the compact final evaluation package for
**CONTEXT MATTERS — Evaluating Diversity-Aware Retrieval for RAG**.

It consumes the frozen Sprint 2 generation collection and evaluates whether
increasing the diversity of retrieved context improves downstream RAG behavior.

No retrieval, generation, embedding, NLI inference, or other scientific
experiment was rerun to create this repository package.

## Research question

> Is diversifying retrieved context actually helpful for LLM reasoning in a
> RAG pipeline, or does it introduce noise and increase hallucination?

## Phase 4 — Evaluation metrics

### 4.1 Correctness

Dataset-specific primary metrics are preserved rather than forcing one metric
across incompatible tasks:

- PubMedQA: decision accuracy
- HotpotQA: token F1, with EM retained as a secondary metric
- ASQA: official QA/STR metric family

Compact per-condition/per-generator summaries and frozen provenance are under:

`phase4/phase41_correctness/`

### 4.2 Faithfulness and hallucination

Generated answer claims/sentences were evaluated against the five retrieved
passages using the frozen DeBERTa-v3-large NLI protocol.

A claim is supported when at least one of the five passages entails it.

Faithfulness:

`# supported claims / # total claims`

Hallucination indicator:

an answer contains at least one unsupported claim.

The large per-generation NLI outputs remain external to Git. Their frozen run
manifests are preserved under:

`phase4/phase42_faithfulness/`

The compact operating-point faithfulness and hallucination results used in the
final analysis are preserved in Phase 5.2.

### 4.3 Coverage

Coverage summaries are preserved under:

`phase4/phase43_coverage/`

Dataset-specific definitions are retained:

- PubMedQA: gold-document Recall@5 and MRR@5
- HotpotQA: supporting-document Recall@5 and both-supporting-documents rate
- ASQA: STR-based coverage metrics

These quantities should not be interpreted as directly interchangeable across
datasets.

### 4.4 Retrieval diversity

For the final Top-5 passages, semantic context diversity is the mean pairwise
cosine distance:

`1 - cosine(e_i, e_j)`

across the ten unique passage pairs.

Compact summaries and the global freeze record are under:

`phase4/phase44_retrieval_diversity/`

### 4.5 Output diversity

ASQA answer output diversity is measured analogously over sentence embeddings.

Answers with fewer than two sentences are ineligible and retain a null output
diversity rather than being assigned zero.

Compact results are under:

`phase4/phase45_output_diversity/`

## Phase 5 — Cross-metric analyses

### 5.1 Diversity–accuracy trade-off

Files:

`phase5/phase51_diversity_accuracy/`

The paired analysis compares each diversified condition with the `none`
baseline for matched examples.

Final interpretation:

greater semantic context diversity was not inherently beneficial to
correctness at the tested operating points. The results generally expose a
relevance–diversity trade-off rather than a universal diversity gain.

### 5.2 Diversity–faithfulness / hallucination

Files:

`phase5/phase52_diversity_faithfulness/`

Final interpretation:

faithfulness and hallucination effects are dataset-, retriever-, generator-,
and condition-dependent. No consistent cross-dataset improvement from
diversification was observed.

### 5.3 Context → output diversity

Files:

`phase5/phase53_context_output_diversity/`

Frozen overall associations:

- cross-sectional Pearson: approximately +0.129
- cross-sectional Spearman: approximately +0.122
- paired-delta Pearson: approximately +0.0186
- paired-delta Spearman: approximately +0.0220

The cross-sectional relationship is weakly positive, but the matched
intervention-style comparison is essentially flat.

These results are descriptive and do not establish causality.

### 5.4 Robustness under paraphrase

Files:

`phase5/phase54_paraphrase_robustness/`

Scope:

- ASQA development subset
- 30 human-validated original/paraphrase pairs
- BM25, DPR, Contriever
- baseline `none`
- predefined MMR-0.5 comparison

Metric:

Jaccard overlap between original-query and paraphrased-query retrieved document
sets.

Higher Jaccard means greater wording invariance. It does **not** mean higher
retrieval quality.

Frozen mean Jaccard@20:

- BM25: approximately 0.307
- DPR: approximately 0.545
- Contriever: approximately 0.502

MMR-0.5 did not consistently improve robustness:

- BM25 Top-5 overlap decreased
- DPR Top-5 overlap increased
- Contriever Top-5 overlap decreased

Therefore the robustness effect is retriever-dependent.

The final Phase 5.4 directory preserves the human authoring/review/adjudication
trail, final validated pair set, compact retrieval artifacts, query-level
metrics, summary, and freeze manifest.

## Final scientific conclusion

Diversification clearly changes the semantic composition of the evidence shown
to the LLM.

However, **more diversity is not automatically better**.

Across the frozen experiments:

- context diversity can be increased;
- correctness does not improve consistently;
- faithfulness does not improve consistently;
- hallucination does not consistently decrease;
- ASQA paired context-diversity increases produced almost no corresponding
  output-diversity increase;
- paraphrase robustness is retriever-dependent.

The practical conclusion is therefore:

> **balance relevance and diversity rather than maximizing diversity alone.**

This does not support claims that diversity is universally harmful, that MMR is
universally poor, or that diversification universally causes hallucination.

## ASQA ColBERTv2 scope

Final ASQA experiments include:

- BM25
- DPR
- Contriever

Full-corpus ASQA ColBERTv2 is intentionally absent under the
supervisor-approved full-corpus feasibility constraint.

It must not be represented as a failed or silently missing final condition.

## Generator naming

Frozen logical generators:

- `gemma4-26b`
- `llama-3.3-70b`
- `ministral-3-14b`

The physical model used for the historical `ministral-3-14b` logical slot was:

`qwen3.6-36b`

The logical alias is preserved for reproducibility.

## Large artifacts intentionally excluded

The following remain outside Git:

- raw generation outputs;
- large per-generation correctness outputs;
- large NLI workloads and NLI result JSONL files;
- joined Phase 5 parquet tables;
- full retrieval-metric JSONL files;
- embedding stores and model caches;
- full figure archive;
- checkpoint and transfer tarballs;
- temporary failed/intermediate analysis directories.

Their identity is represented through frozen manifests, counts, hashes, and
compact summaries.

## Figure provenance

The complete final figure collection was independently frozen.

This package stores:

- repository provenance pointer;
- complete file-level figure-freeze manifest.

The binary figure archive itself remains outside Git.

## Directory overview

    sprint3_evaluation/
    ├── README.md
    ├── SHA256SUMS
    ├── sprint3_package_manifest_v1.json
    ├── src/evaluation/
    ├── phase4/
    │   ├── phase41_correctness/
    │   ├── phase42_faithfulness/
    │   ├── phase43_coverage/
    │   ├── phase44_retrieval_diversity/
    │   └── phase45_output_diversity/
    ├── phase5/
    │   ├── phase51_diversity_accuracy/
    │   ├── phase52_diversity_faithfulness/
    │   ├── phase53_context_output_diversity/
    │   └── phase54_paraphrase_robustness/
    ├── artifacts/final_figures/
    └── provenance/final_figures/

## Scientific status

**COMPLETE / FROZEN**

The next project activities are reporting, presentation, and final repository
handoff—not additional scientific experimentation.
