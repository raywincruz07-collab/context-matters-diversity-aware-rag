# CONTEXT MATTERS — Known Limitations and Reporting Guardrails

Status: **CANONICAL FINAL-DELIVERY LIMITATIONS**

This document records limitations already established by the frozen project evidence. It does not introduce new experiments, metrics, results, or post-hoc scientific changes.

## 1. ASQA ColBERTv2

- Final ASQA experiments use **BM25, DPR, and Contriever only**.
- ColBERTv2 was not completed for the full 21,015,324-passage ASQA corpus.
- No reduced-corpus or subsampled ColBERTv2 replacement is treated as a final ASQA result.
- PubMedQA and HotpotQA retain all four retrievers.

## 2. ASQA protected-final aspect retrieval metrics

- No frozen protected-final dev948 numeric artifact was found for:
  - `S-Recall@5`,
  - `alpha-nDCG@5`,
  - corpus coverability,
  - or `c*`.
- Historical DEVELOPMENT / SELECTION artifacts must not be substituted for dev948.
- These values must therefore be reported as **not available / not computed in the frozen protected-final package**.
- They must not be reconstructed, inferred, or invented for the final report or presentation.

## 3. ASQA Phase 4.3 coverage semantics

- The frozen final Phase 4.3 ASQA coverage outputs use **STR-EM / STR-Hit**.
- These are answer-side ASQA coverage signals.
- They are not interchangeable with retrieval-side `S-Recall@5` or `alpha-nDCG@5`.

## 4. Retrieval diversity is a manipulation check

- Retrieval diversity is the mean pairwise cosine distance among the final Top-5 passage embeddings.
- The project uses the frozen Contriever embedding representation for this metric.
- Because embedding-space diversity is also involved in parts of diversification, the metric is not a fully independent quality measure.
- It should be described as a **retrieval-diversity / manipulation diagnostic**, not proof that retrieval quality improved.

## 5. Diversity does not imply better answers

- Increased retrieval diversity must not be interpreted automatically as increased correctness, faithfulness, coverage, or answer quality.
- Final interpretation must remain configuration-specific across dataset × retriever × generator × diversification condition.
- The frozen evidence does not support a universal claim that diversification is always beneficial or always harmful.

## 6. Output-diversity eligibility

- Phase 4.5 output diversity is evaluated on ASQA answers with at least two frozen sentences.
- Answers with fewer than two sentences are excluded from the metric.
- They must not be assigned `output_diversity = 0`.
- Phase 5.3 inherits this eligibility restriction.

## 7. Phase 5.3 interpretation

- Phase 5.3 reports descriptive associations between context diversity and output diversity.
- Pearson / Spearman association does not by itself establish causality.
- The paired analysis is secondary and restricted to pairs where both outputs satisfy Phase 4.5 eligibility.

## 8. Phase 5.4 scope

- Paraphrase robustness uses the frozen 30-pair ASQA DEVELOPMENT set.
- It evaluates BM25, DPR, and Contriever retrieval under original versus paraphrased queries.
- Jaccard overlap measures retrieval invariance, not answer quality.
- This analysis must remain clearly labelled as a robustness / development analysis rather than the full protected-final ASQA population.

## 9. Generator naming

- Historical artifact paths use the logical identifier `ministral-3-14b`.
- The physical model executed for that logical slot was `qwen3.6-36b`.
- Final professor-facing reporting should use **Qwen 3.6 36B** while retaining the historical alias only where needed for provenance.

## 10. Historical Sprint 1 versus controlled baseline

- Historical Sprint-1 observations are preserved as historical evidence.
- The later controlled `none` condition provides the consistent baseline used for direct Sprint-2 comparisons.
- Historical and later controlled outputs must not be merged as though they were generated under one identical protocol.

## 11. Frozen versus developmental artifacts

- DEVELOPMENT, SELECTION, historical, and protected-final evidence roles must remain distinct.
- Development or selection results must not silently replace missing final results.
- Failed or superseded analysis artifacts must not be promoted to final authority.

## 12. No retrospective scientific repair

- Final research execution is frozen.
- Do not rerun generation, retrieval, NLI, diversification, embeddings, or final metrics solely to make documentation look more complete.
- Missing frozen evidence must be documented as a limitation rather than recreated post hoc.

## 13. Final reporting rule

Every quantitative claim in the final notebooks, report, GitHub documentation, and presentation must trace to the canonical registries and frozen source artifacts.

Canonical inputs:
- `EXPERIMENT_REGISTRY.csv` SHA256 `36bea942e41b11fca9232b5fd51def07ff266cab62b9ab4fcc5c9746e196218d`
- `RESULTS_REGISTRY.csv` SHA256 `3fd20110c19246a30276fcb02c4b5171937d2b3efd628d9ab12d9d7699a6dca8`
- `FIGURE_REGISTRY.csv` SHA256 `170d8030c9a05a922d33e5e8162aec888344f391815dd9e9c5688abe6c5c1e0e`
- `PROVENANCE_MAP.md` SHA256 `829979e33d73990fc3ead6d939832eebd6c8b3222cef86ebc111845e6648048f`

**Scientific rerun authorized by this document: NO.**
