# Sprint 3 Diversification Configuration Protocol

## 1. Purpose

This document freezes the diversification candidate universe and
protected-final treatment structure before development or selection outcomes are
observed.

This document does not yet select the final KMeans or Agglomerative `k`. Those
values remain dependent on the later frozen selection objective and statistical
protocol.

The following values are already frozen globally:

```text
candidate_pool = 20
final_top_k    = 5
```

For a given query and retriever, every diversification method consumes the exact
same ranked first-stage 20-document candidate artifact.

## 2. Baseline

One explicit relevance-only condition is frozen:

```text
method = none
```

It selects candidate ranks 1 through 5 from the exact shared top-20 candidate
artifact.

The explicit baseline must not be replaced with MMR `lambda = 1`.

## 3. MMR Development Grid

The development MMR candidate values are frozen as:

```text
lambda = {
  0.00,
  0.25,
  0.50,
  0.75,
  1.00
}
```

Their interpretations are:

### `lambda = 0.00`

- extreme-diversity diagnostic;
- still relevance-seeded by the current implementation.

### `lambda = 0.25`

- diversity-heavy intermediate.

### `lambda = 0.50`

- balanced relevance/diversity.

### `lambda = 0.75`

- relevance-heavy intermediate.

### `lambda = 1.00`

- development equivalence check against the explicit relevance-only baseline's
  document IDs and order.

If `lambda = 1` fails selected-document or order equivalence with the baseline
under the shared candidate contract, treat that as an implementation or
contract defect, not a scientific treatment difference.

## 4. MMR Protected-Final Policy

The protected-final MMR intensity curve is frozen as:

```text
lambda = {
  0.00,
  0.25,
  0.50,
  0.75
}
```

These four levels are used alongside the separate relevance-only baseline.

Do not include `lambda = 1` as another generated protected-final context,
because its selected document IDs and order should duplicate the explicit
baseline.

Do not choose or drop these four MMR points according to selection performance.
They are predeclared treatment levels for the scientific question of how
increasing diversification pressure changes:

- retrieval coverage;
- relevance;
- evidence diversity;
- answer correctness;
- faithfulness and hallucination;
- Bidirectional Atomic-Content Coverage.

Label `lambda = 0` as an extreme-diversity diagnostic rather than the presumed
optimal treatment.

## 5. KMeans Development Grid

The KMeans candidate values are frozen as:

```text
k = {
  2,
  3,
  5
}
```

The current scientific implementation semantics are frozen unless separately
amended prospectively:

- normalized Contriever-space embeddings;
- k-means++ initialization;
- `n_init = 1`;
- Lloyd algorithm;
- `max_iter = 300`;
- `tol = 1e-4`;
- `seed = 42`;
- relevance-ranked members within clusters;
- clusters ordered by their strongest relevance-ranked member;
- round-robin selection across clusters;
- exactly five final documents;
- collapsed or fewer-than-requested effective-cluster conditions fail loudly.

With `final_top_k = 5`:

### `k = 2`

- two first-round cluster representatives followed by three relevance-ranked
  round-robin fills.

### `k = 3`

- three first-round representatives followed by two fills.

### `k = 5`

- one selected representative from each of five effective clusters.

The representative is the highest-relevance document within the cluster. This
must not be described as centroid-nearest representative selection unless the
implementation is prospectively changed and separately validated.

## 6. KMeans Protected-Final Policy

KMeans `k` is a method hyperparameter, not a protected-final
diversity-intensity curve.

Exactly one globally selected KMeans `k` from `{2, 3, 5}` will later reach the
canonical protected-final matrix. The winning `k` must be selected using the
later predeclared selection objective and statistical rule.

It must not be selected separately by:

- dataset;
- retriever;
- LLM.

The exact winning `k` is:

```text
UNRESOLVED UNTIL METHOD-SELECTION RULE IS FROZEN AND APPLIED
```

## 7. Agglomerative Development Grid

The Agglomerative candidate values are frozen as:

```text
k = {
  3,
  5
}
```

Do not add `k = 2` merely for symmetry with KMeans.

The current scientific implementation semantics are frozen unless separately
amended prospectively:

- normalized Contriever embedding space;
- Euclidean distance;
- Ward linkage;
- deterministic clustering subject to pinned library behavior;
- relevance-ranked members within clusters;
- clusters ordered by their strongest relevance member;
- round-robin selection;
- exactly five final documents.

Equal-distance and tie behavior depends on the pinned scikit-learn and
environment implementation and must be provenance-recorded.

## 8. Agglomerative Protected-Final Policy

Agglomerative `k` is a method hyperparameter, not a protected-final intensity
curve.

Exactly one globally selected Agglomerative `k` from `{3, 5}` will later reach
the canonical protected-final matrix.

It must not be selected separately by:

- dataset;
- retriever;
- LLM.

The exact winning `k` is:

```text
UNRESOLVED UNTIL METHOD-SELECTION RULE IS FROZEN AND APPLIED
```

## 9. Deterministic DPP

The canonical DPP treatment is frozen as:

```text
method = dpp_map
```

Its current meaning is:

- deterministic forward-greedy determinant-maximization procedure;
- DPP MAP-style approximation;
- fixed cardinality of five;
- `theta = 1.0`;
- no random seed.

This method must not be called an exact global MAP solver.

It is the primary and canonical DPP treatment for selection and protected-final
evaluation.

## 10. Stochastic DPP

The corrected exact fixed-k stochastic k-DPP is retained only as a
development-stage retrieval sensitivity.

The seeds are frozen as:

```text
1
2
3
```

These are three stochastic realizations of one k-DPP configuration, not three
independent methods.

For this sensitivity:

- preserve every seed-level retrieval result;
- never cherry-pick the best seed;
- compute seed-specific retrieval and diversity metrics;
- report between-seed variability;
- for a summary, average seed-specific metric values within the same question
  and configuration before question-level aggregation.

No downstream generation is required for all three stochastic seeds under the
canonical design.

The stochastic DPP sensitivity does not enter:

- routine selection;
- protected-final generation;
- protected-final canonical statistics.

Promotion would require a prospective amendment before observing
protected-final outcomes.

Historical Sprint 2 stochastic DPP remains `HISTORICAL_OBSERVED` because it was
not exact fixed-k k-DPP.

## 11. DPP Authority Note

The kickoff experimental suggestions mention using three or more random seeds
for DPP sampling.

This is guidance for stochastic DPP sampling, not a requirement that a
deterministic DPP-MAP treatment itself have multiple seeds.

The current protocol satisfies that guidance through the exact fixed-k
stochastic development sensitivity with seeds `{1, 2, 3}`, while using
deterministic `dpp_map` as the canonical DPP treatment.

This protocol does not claim that the kickoff provided no DPP-seed guidance.

## 12. Spectral Clustering and xQuAD

The following are excluded from canonical Sprint 3 scope:

- Spectral clustering;
- xQuAD.

They appeared only as optional or suggested additional techniques, not as later
hard requirements.

The frozen project already contains:

- one relevance-only baseline;
- MMR;
- KMeans;
- Agglomerative;
- DPP;
- four retrievers;
- three datasets;
- three LLMs;
- `WITH_CONTEXT` and `WITHOUT_CONTEXT`;
- multiple evaluation constructs.

Adding optional method families would increase implementation and selection
degrees of freedom without being necessary to answer the research questions
within a 12-ECTS project.

## 13. Cross-Dataset Policy

Use the same diversification candidate universe across:

- PubMedQA;
- HotpotQA;
- ASQA.

Do not define dataset-specific MMR grids, clustering grids, or DPP parameters.

PubMedQA remains control and replication evidence. Its expected weaker benefit
from diversification is not a reason to retune it.

## 14. Cross-Retriever Policy

Do not tune diversification parameters separately for:

- BM25;
- DPR;
- Contriever;
- ColBERTv2.

The protocol freezes:

- the same full MMR intensity curve everywhere;
- one globally selected KMeans `k`;
- one globally selected Agglomerative `k`;
- one deterministic DPP configuration.

This limits degrees of freedom and improves interpretability.

## 15. Development Matrix

Per base retriever, development physical retrieval conditions are:

| Family | Physical conditions |
| --- | ---: |
| Baseline | 1 |
| MMR | 5 |
| KMeans | 3 |
| Agglomerative | 2 |
| Deterministic DPP | 1 |
| Stochastic k-DPP | 3 seed realizations |
| **Total** | **15** |

Stochastic seeds are conceptually repeated realizations, not three separate
methods.

This is not the official generation matrix.

## 16. Selection Matrix

Before protected-final evaluation, the planned selection comparison per base
retriever is:

| Family | Retrieval conditions |
| --- | ---: |
| Baseline | 1 |
| MMR protected scientific curve (`0, 0.25, 0.50, 0.75`) | 4 |
| KMeans candidates | 3 |
| Agglomerative candidates | 2 |
| Deterministic DPP | 1 |
| **Total** | **11** |

The following are excluded from selection:

- the MMR `lambda = 1` equivalence check;
- the stochastic DPP seed sensitivity.

The exact use of downstream generation and evaluation at selection remains
dependent on the later method-selection objective and statistical protocol. Do
not silently assume that all 11 conditions require full downstream generation
until that protocol is frozen.

## 17. Protected-Final Structure

The eventual canonical protected-final structure per base retriever is:

| Family | Retrieval/context treatment conditions |
| --- | ---: |
| Baseline | 1 |
| MMR intensity curve | 4 |
| Selected KMeans | 1 |
| Selected Agglomerative | 1 |
| Deterministic DPP | 1 |
| **Total** | **8** |

Across four retrievers, this gives 32 treatment conditions per dataset.

For the HotpotQA and ASQA protected-final datasets, this gives 64 retrieval and
context treatment conditions before LLM multiplication.

PubMedQA may use the same structural matrix as control or replication evidence
but must not be called protected final.

These counts are structural treatment counts only, not an official generation
count.

## 18. Still Unresolved

The following remain unresolved:

- exact selected KMeans `k`;
- exact selected Agglomerative `k`;
- HotpotQA lead selection and relevance-complementarity metric;
- exact acceptable relevance-loss or non-inferiority margin;
- cross-dataset aggregation or weighting used to select the global clustering
  settings;
- exact selection objective;
- paired statistical and uncertainty protocol;
- tie and acceptable-loss rule;
- whether full downstream generation is required for every selection candidate
  or only candidates passing a retrieval-screen gate.

This document does not resolve those items.

`candidate_pool = 20` and `final_top_k = 5` are already frozen elsewhere and
must not be reopened.

## 19. Frozen Before Observation

This protocol freezes the diversification candidate universe, the full MMR
protected intensity curve, the KMeans and Agglomerative candidate grids, the
deterministic canonical DPP treatment, the stochastic-DPP development
sensitivity, the cross-dataset and cross-retriever parameter-sharing policy, and
the protected-final treatment structure before selection or protected-final
outcomes are observed.

It does not select final KMeans or Agglomerative `k` values.

No selection or protected-final results are observed while writing this
document.
