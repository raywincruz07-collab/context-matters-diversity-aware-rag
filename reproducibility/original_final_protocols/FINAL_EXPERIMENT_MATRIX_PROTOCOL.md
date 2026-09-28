# Sprint 3 Final Experiment Matrix Protocol

## 1. Purpose

This protocol defines exactly which canonical Sprint-3 experimental
conditions exist after bounded clustering-k selection is completed.

The matrix answers:

- which datasets;
- which evidence roles;
- which retrievers;
- which diversification conditions;
- which generator LLMs;
- which context modes;
- which evaluations;

are part of the canonical final Sprint-3 research story.

The matrix is frozen before canonical `SELECTION` or protected-final outcomes
are observed. It does not authorize execution.

The forthcoming
`docs/sprint3/CONFIRMATORY_CONTRAST_REGISTRY.md` will define exactly which
matrix comparisons are confirmatory.

This document defines conditions, not hypothesis-test multiplicity families.

## 2. Datasets and Evidence Roles

Freeze exactly three canonical Sprint-3 datasets.

### PubMedQA

Dataset population:

```text
complete expert-labeled PQA-L1000
```

Canonical Sprint-3 role:

```text
HISTORICAL_OBSERVED_CONTROL_REPLICATION
N = 1,000
```

The complete PQA-L1000 has already been historically exposed during earlier
project work.

Therefore:

- it is not protected-final evidence;
- it must never be called untouched, unseen, held-out final, or independent
  confirmation;
- it remains scientifically useful as a single-document or control
  replication and cross-dataset comparison.

Freeze Sprint-3 breadth as running the same canonical final retrieval,
diversification, and generation structure used for the other datasets. This
closes the previous PubMedQA breadth ambiguity.

PubMedQA contributes to:

- three-dataset descriptive comparisons;
- no-context versus context analysis;
- relevance-only versus diversification analysis;
- correctness;
- faithfulness;
- ACC;
- robustness and control interpretation.

PubMedQA contributes no numeric vote to clustering-k `SELECTION`.

The canonical project corpus is the existing PubMedQA-derived section-level
corpus constructed from those 1,000 examples, with an expected corpus size of:

```text
3,358 passage/section documents
```

Physical corpus identity, count, and hash must still be verified and bound to
provenance before canonical execution.

For each PubMedQA question `q`, define:

```text
Rel_q = the set of canonical corpus document IDs corresponding to the
        dataset-provided context sections for q
```

Use the same logical 3,358-document corpus across:

- BM25;
- DPR;
- Contriever;
- ColBERTv2;
- all canonical diversification conditions.

Do not rebuild a query-specific corpus. Do not add outside PubMed documents.
Do not inject gold sections into a candidate list after retrieval.

Running the full 1,000-question, full eight-condition, three-LLM canonical
Sprint-3 PubMedQA replication is a project structural design choice. It is
chosen prospectively to provide:

- the same retriever factor across all three datasets;
- the same diversification-family coverage across all three datasets;
- the same three-LLM comparison;
- `WITH_CONTEXT` versus `WITHOUT_CONTEXT` evidence;
- canonical correctness, faithfulness, and ACC under the new frozen
  instruments;
- a focused-evidence control against which HotpotQA and ASQA behavior can be
  interpreted.

It is not claimed to be:

- a professor-mandated full factorial requirement;
- protected confirmation;
- untouched evidence.

It is not chosen because historical PubMedQA results favor any method.
Historical Sprint-1 results remain separate artifacts and are not overwritten
or relabeled as canonical Sprint-3 outputs.

### HotpotQA

Canonical final role:

```text
PROJECT_PROTECTED_FINAL
N = 500
```

The canonical BEIR test population contains 7,405 queries. The historically
observed subset contains 500 queries, and the eligible protected population is
the remaining 6,905 BEIR-test queries. Therefore:

```text
historical_500 union eligible_6905 = canonical BEIR test7405
historical_500 intersection eligible_6905 = empty set
```

The canonical protected-final `N=500` is sampled only from those eligible
6,905 queries according to
`docs/sprint3/HOTPOTQA_FINAL_EVALUATION_PROTOCOL.md`.

BEIR dev `N=5,447` remains the separate `SELECTION` split governed by the
selection protocol.

The exact protected 500 IDs are not materialized by this document. They are
produced later by the already-frozen deterministic protected-sample procedure.
Do not expose them before the applicable stage gate.

### ASQA

Canonical final role:

```text
PROJECT_PROTECTED_FINAL
```

Population:

```text
official ASQA dev split
N = 948
```

The entire dev948 is used. Do not subsample it for canonical final evaluation.

The internal train-derived partitions:

```text
DEVELOPMENT = 3,482
SELECTION   = 871
```

do not enter the protected-final matrix. They remain governed by
`docs/sprint3/ASQA_INTERNAL_PARTITION_PROTOCOL.md`.

## 3. Four Canonical Retrievers

For every dataset, use exactly:

1. BM25;
2. DPR;
3. Contriever;
4. ColBERTv2.

Do not select a winning retriever and discard the others. Retriever identity
is an experimental factor.

Every retriever must operate according to the applicable frozen corpus and
retriever contracts. Physical indexes may differ. The logical corpus and
passages within a dataset must remain shared according to the corpus protocol.

## 4. Canonical First-Stage Retrieval

For every:

```text
dataset x sample x retriever
```

freeze:

```text
candidate_pool = 20
```

The exact first-stage top-20 artifact is shared by:

- relevance-only baseline;
- MMR conditions;
- KMeans;
- Agglomerative;
- `dpp_map`.

Do not reretrieve separately for diversification methods.

Final selected context size is:

```text
top_k = 5
```

The canonical generator always receives exactly five successfully selected
passage bodies for `WITH_CONTEXT`.

These rules remain governed by
`docs/sprint3/CANDIDATE_POOL_TOPK_PROTOCOL.md`.

## 5. Eight Canonical WITH_CONTEXT Conditions

For each retriever, the final canonical Sprint-3 matrix contains exactly eight
`WITH_CONTEXT` retrieval and context conditions.

Use these stable logical condition IDs:

1. `relevance_baseline`;
2. `mmr_lambda_0`;
3. `mmr_lambda_0_25`;
4. `mmr_lambda_0_50`;
5. `mmr_lambda_0_75`;
6. `kmeans_selected`;
7. `agglo_selected`;
8. `dpp_map`.

### Relevance Baseline

Select the first five passages from the canonical relevance-ranked top-20.
This is the same-retriever comparator for every diversification treatment.

### MMR

Canonical final MMR levels are exactly:

```text
lambda = 0
lambda = 0.25
lambda = 0.50
lambda = 0.75
```

Do not include `lambda=1` as a canonical final treatment. `lambda=1` is
`DEVELOPMENT` equivalence or defect validation only.

Do not select the best lambda using `SELECTION` or protected outcomes. The
four canonical MMR values are predeclared treatments.

### KMeans

Use exactly one globally selected KMeans `k`.

Logical matrix identity:

```text
kmeans_selected
```

The physical selected `k` is filled only after the bounded selection procedure
in `docs/sprint3/SELECTION_STATISTICS_PROTOCOL.md` validly completes.

The winner must be one of `{2,3,5}` and must be the same selected `k` across:

- all three datasets;
- all four retrievers;
- all three LLMs.

Rejected KMeans `k` values do not enter protected-final evaluation.

### Agglomerative

Use exactly one globally selected Agglomerative `k`.

Logical matrix identity:

```text
agglo_selected
```

The physical selected `k` is filled only after valid bounded selection.

The winner must be one of `{3,5}` and must be shared across:

- all three datasets;
- all four retrievers;
- all three LLMs.

Rejected Agglomerative `k` values do not enter protected-final evaluation.

### DPP

Use `dpp_map` as the canonical deterministic DPP treatment.

Do not include stochastic fixed-k DPP seeds 1, 2, or 3 in the canonical final
matrix. Those seeds are `DEVELOPMENT` sensitivity only.

## 6. Methods Excluded from the Canonical Final Matrix

Do not include as canonical final conditions:

- MMR `lambda=1`;
- KMeans non-selected `k` values;
- Agglomerative non-selected `k` values;
- stochastic DPP seed 1;
- stochastic DPP seed 2;
- stochastic DPP seed 3;
- Spectral clustering;
- xQuAD;
- `candidate_pool=10`;
- `candidate_pool=50`;
- any post-hoc method;
- any newly discovered diversification method.

Development sensitivity evidence may still be reported as development
evidence. Do not relabel it protected-final evidence.

## 7. Three Primary Generator LLMs

For every canonical generated-answer condition, use exactly:

1. `llama-3.3-70b`;
2. `gemma4-26b`;
3. `ministral-3-14b`.

The reserve `qwen3.5-122b` is not a fourth matrix LLM. It enters only if the
prospective reserve-substitution procedure is validly invoked before the
frozen Stage-2 cutoff.

If one primary LLM is validly replaced before that cutoff, update the physical
LLM mapping prospectively while preserving a three-generator design.

Do not:

- choose the best LLM;
- average away LLM identity;
- use different diversification configurations by LLM.

## 8. WITH_CONTEXT Generation Matrix

For every:

```text
dataset
x sample
x retriever
x one of the 8 canonical context conditions
x one of the 3 primary LLMs
```

generate exactly one canonical `WITH_CONTEXT` answer.

Canonical generation replicas:

```text
1
```

Use `docs/sprint3/GENERATION_PROTOCOL.md` for:

- system prompt;
- exact user template;
- context rendering;
- temperature;
- token limits;
- retry policy;
- status semantics.

## 9. WITHOUT_CONTEXT Matrix

For every:

```text
dataset x sample x LLM
```

generate exactly one canonical `WITHOUT_CONTEXT` answer.

Do not multiply `WITHOUT_CONTEXT` by:

- retriever;
- diversification method;
- lambda;
- clustering family;
- clustering `k`;
- DPP identity.

The canonical key remains:

```text
dataset
x sample_id
x LLM
x prompt_version
x decoding_version
```

The same `WITHOUT_CONTEXT` generation is reused for every valid RQ1 comparison
that needs it.

## 10. Exact Generation Counts

Given the frozen structural design:

```text
WITH_CONTEXT rows per dataset:
N x 4 retrievers x 8 context conditions x 3 LLMs
= 96N

WITHOUT_CONTEXT rows per dataset:
N x 3 LLMs
= 3N

Total canonical generation rows per dataset:
99N
```

Freeze the expected counts.

### PubMedQA

```text
N = 1,000

WITH_CONTEXT:    96,000
WITHOUT_CONTEXT:  3,000
TOTAL:           99,000

Evidence role:
HISTORICAL_OBSERVED_CONTROL_REPLICATION
```

### HotpotQA

```text
N = 500

WITH_CONTEXT:    48,000
WITHOUT_CONTEXT:  1,500
TOTAL:           49,500

Evidence role:
PROJECT_PROTECTED_FINAL
```

### ASQA

```text
N = 948

WITH_CONTEXT:    91,008
WITHOUT_CONTEXT:  2,844
TOTAL:           93,852

Evidence role:
PROJECT_PROTECTED_FINAL
```

### Grand Total

```text
WITH_CONTEXT:    235,008
WITHOUT_CONTEXT:   7,344
TOTAL:           242,352 canonical generation rows
```

These are structural expected counts. They are not permission to launch
generation. Actual measured or `OK` counts may be smaller because governed
failures remain visible.

Do not reduce the matrix merely because the expected generation count is large
without a prospective methodological amendment.

## 11. Retrieval Condition Counts

Retrieval and context selection do not depend on LLM.

For each dataset:

```text
N x 4 retrievers x 8 canonical context conditions
= 32N
```

Expected canonical retrieval and context-selection rows are:

```text
PubMedQA: 32,000
HotpotQA: 16,000
ASQA:     30,336

Grand total: 78,336
```

Do not duplicate retrieval solely because three LLMs consume the same selected
context artifact.

## 12. Correctness Evaluation

Evaluate canonical answer correctness for:

- every valid `WITH_CONTEXT` generation;
- every valid `WITHOUT_CONTEXT` generation.

Use only `docs/sprint3/ANSWER_CORRECTNESS_PROTOCOL.md`.

Dataset-specific primary correctness remains:

```text
PubMedQA:
categorical decision accuracy

HotpotQA:
official answer F1 primary, EM secondary

ASQA:
official QA-F1 / Disambig-F1 primary according to the frozen correctness
protocol, with answer-side alias coverage remaining diagnostic
```

Do not create method-specific correctness definitions.

## 13. Faithfulness Evaluation

Evaluate canonical `Faithfulness@5` for every eligible `WITH_CONTEXT` answer
only.

Do not evaluate numeric faithfulness for `WITHOUT_CONTEXT`.

Use `docs/sprint3/FAITHFULNESS_PROTOCOL.md`.

Therefore, every canonical `WITH_CONTEXT` condition may contribute:

- correctness;
- faithfulness.

## 14. ACC Evaluation

ACC is defined only for:

```text
same dataset
x same sample
x same retriever
x same LLM
x relevance_baseline WITH_CONTEXT
versus
one diversified WITH_CONTEXT treatment
```

Diversified ACC treatments are exactly:

- `mmr_lambda_0`;
- `mmr_lambda_0_25`;
- `mmr_lambda_0_50`;
- `mmr_lambda_0_75`;
- `kmeans_selected`;
- `agglo_selected`;
- `dpp_map`.

Thus there are seven canonical ACC comparisons per:

```text
dataset x sample x retriever x LLM
```

Expected ACC pair count:

```text
N x 4 x 7 x 3
= 84N
```

Expected counts:

```text
PubMedQA: 84,000
HotpotQA: 42,000
ASQA:     79,632

Grand total: 205,632 canonical ACC pairs
```

These are expected pair identities, not guaranteed measured numeric ACC rows.

Generation, decomposer, and verifier failures remain governed by
`docs/sprint3/ACC_PROTOCOL.md`.

`WITHOUT_CONTEXT` does not enter ACC.

## 15. Retrieval Metric Evaluation

Evaluate retrieval metrics on all eight canonical `WITH_CONTEXT` retrieval
conditions.

No retrieval metric exists for `WITHOUT_CONTEXT`.

For HotpotQA, include the frozen retrieval metrics, with positive-qrel document
`Recall@5` carrying the already-frozen lead role where applicable.

For ASQA, use:

- `SRecall@5` lead;
- `alpha-nDCG@5` with `alpha=0.5` as mandatory secondary;
- `alpha=0.3` and `alpha=0.7` sensitivity;
- corpus coverability;
- `c*` diagnostic;

according to `docs/sprint3/ASQA_RETRIEVAL_METRIC_PROTOCOL.md`.

For PubMedQA, freeze the primary descriptive retrieval-effectiveness metric as
positive-gold-section `Recall@5`. For each question `q` with `|Rel_q| > 0`:

```text
Recall@5(q) = |Retrieved_top5(q) intersection Rel_q| / |Rel_q|
```

The secondary PubMedQA retrieval metric is `MRR@5`:

```text
MRR@5(q) = 1 / rank of the first retrieved document in Rel_q
           if one occurs in the top five;
           otherwise 0
```

`Recall@5` measures recovery of the dataset-provided PubMedQA context sections
inside the constructed project corpus. `MRR@5` measures how early the first
such relevant section appears.

Do not claim these metrics measure:

- independent-study coverage;
- aspect diversity;
- multi-source evidence diversity;
- external biomedical retrieval quality.

PubMedQA is structurally a focused or single-source control, and its retrieval
metrics remain descriptive or control evidence. Do not introduce PubMedQA
nDCG or a new primary retrieval metric in this protocol.

Do not invent a new cross-dataset pooled retrieval metric here.

## 16. Retrieval Diversity Diagnostics

For every canonical `WITH_CONTEXT` retrieval condition, report the applicable
frozen retrieval-diversity diagnostics.

Their role is:

- manipulation check;
- mechanism description;
- diversity-versus-quality analysis.

Do not treat diversity itself as a quality objective. Do not change a selected
configuration because another condition achieves greater raw geometric
diversity.

## 17. RQ Coverage of the Matrix

The matrix must support the already-frozen research questions.

### RQ0

When and under what conditions does diversity-aware retrieval help RAG, and
when does it introduce noise or hallucination?

### RQ1

Does retrieval context improve generated answers compared with no context?

This is supported structurally by:

```text
WITHOUT_CONTEXT
versus
relevance_baseline WITH_CONTEXT
```

within the same:

```text
dataset x sample x LLM
```

with relevance-baseline comparisons stratified by retriever.

### RQ2

Does diversification improve outcomes relative to relevance-only retrieval?

This is supported by:

```text
relevance_baseline
versus each of the seven diversified treatments
```

within:

```text
dataset x retriever x LLM
```

### RQ3

What relevance and diversity trade-offs correspond to coverage, correctness,
and faithfulness changes?

This is supported by joint retrieval-diversity, retrieval-effectiveness,
correctness, faithfulness, and ACC measurements.

### RQ4

How do effects depend on dataset and generator LLM?

This is supported by preserving all three datasets and all three LLMs rather
than selecting one.

### RQ5

How does diversification affect context-groundedness or hallucination risk?

This is supported by canonical `Faithfulness@5` and its component claim
states.

The forthcoming contrast registry determines which exact comparisons are
confirmatory versus descriptive or exploratory.

## 18. Evidence-Role Discipline

Do not pool evidence roles as if they were equivalent.

Protected-final reporting must clearly separate:

```text
HotpotQA PROJECT_PROTECTED_FINAL
ASQA PROJECT_PROTECTED_FINAL
```

from:

```text
PubMedQA HISTORICAL_OBSERVED_CONTROL_REPLICATION
```

A three-dataset figure or table may display all three, but captions and
analysis must retain evidence-role labels.

Do not write:

> all three datasets are independent held-out final tests.

## 19. Selection Results Enter Only as Configuration IDs

The final matrix is structurally frozen now but contains placeholders:

```text
KMEANS_SELECTED_K
AGGLO_SELECTED_K
```

Selection may later replace only those two placeholders with the validly
selected values.

It may not change:

- number of clustering conditions;
- MMR levels;
- DPP condition;
- retriever set;
- dataset set;
- LLM set;
- context modes;
- metrics.

The selected-value substitution must be provenance-recorded. No other matrix
redesign is permitted from `SELECTION` outcomes.

## 20. Final Configuration Count

For one dataset and one sample:

```text
Canonical WITH_CONTEXT retrieval configurations:
4 retrievers x 8 conditions
= 32

Generated WITH_CONTEXT cells:
32 x 3 LLMs
= 96

Canonical WITHOUT_CONTEXT generated cells:
3

Total generated cells:
99 per sample
```

Do not describe `WITHOUT_CONTEXT` as a ninth per-retriever retrieval condition.
It is retrieval-independent.

## 21. Protected-Final Entry Conditions

Protected-final execution may not begin until all applicable stage-gate
requirements pass.

At minimum, this includes:

- professor or supervisor margins and rules resolved;
- clustering selection completed validly;
- KMeans winner frozen;
- Agglomerative winner frozen;
- final matrix frozen;
- confirmatory contrast registry frozen;
- generator bundle validated;
- decomposer winner frozen;
- NLI verifier snapshot and thresholds frozen;
- ASQA matcher validated;
- corpus identities frozen;
- run-registry and provenance infrastructure ready;
- applicable resource gates passed.

Do not interpret this list as overriding
`docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL.md`. That protocol remains
authoritative for stage transition.

## 22. Corpus and Resource Blockers

This matrix does not authorize a reduced or query-specific corpus.

### HotpotQA

Corpus and resource behavior remains governed by:

- `docs/sprint3/HOTPOTQA_FINAL_EVALUATION_PROTOCOL.md`;
- `docs/sprint3/HOTPOTQA_RESOURCE_GATE_PROTOCOL.md`.

The substantive and resource fallback item, including a defensible `M_min`,
must be closed before protected execution if full-corpus feasibility fails.

### ASQA

The canonical preference remains the full DPR-2018 passage collection.

No fallback corpus is authorized by this matrix.

If full-corpus ASQA proves infeasible, stop. A prospective amendment is
required before protected execution. Do not silently create a smaller ASQA
corpus.

## 23. Failed Selection Family

If KMeans or Agglomerative has no valid selection winner under
`docs/sprint3/SELECTION_STATISTICS_PROTOCOL.md`, do not:

- choose the least-bad `k`;
- substitute a `DEVELOPMENT` winner;
- omit the family silently;
- proceed with protected evaluation and decide later.

Protected-final matrix execution is blocked. A prospective amendment with full
exposure accounting is required.

## 24. Experimental Independence

Retrieval artifacts are generator-independent.

Use the exact same selected-context artifact across all three LLMs.

Do not allow:

- an LLM-specific retrieved context;
- LLM-specific diversification parameters;
- LLM-specific ranking changes.

The LLM observes the fixed retrieval intervention. This preserves
interpretation of generator heterogeneity.

## 25. No Post-Hoc Matrix Reduction

After `SELECTION` or protected-final outcomes are observed, do not remove a
canonical condition because it:

- performs poorly;
- appears redundant;
- creates failures;
- contradicts the hypothesis;
- is expensive;
- gives an inconvenient result.

Failure and missingness are results. Any unavoidable matrix change requires a
prospective amendment and explicit exposure accounting.

## 26. Confirmatory and Exploratory Boundary

Presence in this matrix does not automatically make every possible pairwise
comparison confirmatory.

The forthcoming `docs/sprint3/CONFIRMATORY_CONTRAST_REGISTRY.md` must
prospectively define:

- contrast IDs;
- reference and treatment conditions;
- metric;
- direction or hypothesis, where applicable;
- evidence role;
- dataset, retriever, and LLM strata;
- multiplicity family;
- p-value status;
- confirmatory versus descriptive role.

Unregistered combinations remain descriptive or exploratory even though their
raw results are retained.

## 27. Expected Matrix Artifact

Before protected-final execution, materialize a machine-readable matrix
registry from this protocol.

At minimum, each registered cell or condition must bind:

- matrix protocol version and hash;
- dataset;
- evidence role;
- sample-manifest identity;
- retriever or `NOT_APPLICABLE`;
- candidate-pool version;
- context-condition ID;
- diversification method;
- diversification hyperparameter or selected-winner reference;
- selected top-k;
- context mode;
- LLM or `NOT_APPLICABLE` for retrieval-only rows;
- generation protocol and hash;
- expected evaluation metrics;
- protected, selection, and development prohibition flags where applicable;
- run-registry identity.

The machine-readable registry is implementation and provenance work. Do not
materialize it in this task.

## 28. Expected Counts Are Assertions

Before canonical execution, implementation must validate that registry counts
match this protocol.

Expected totals are:

```text
datasets = 3
retrievers = 4
WITH_CONTEXT conditions per retriever = 8
LLMs = 3

canonical generation rows = 242,352
canonical retrieval/context-selection rows = 78,336
canonical ACC pair identities = 205,632
```

If implementation produces a different expected count, stop. Do not begin
expensive canonical generation until the discrepancy is explained and either:

- implementation is corrected; or
- a prospective protocol amendment is approved.

## 29. Relation to Other Protocols

This protocol must be read with:

- `docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL.md`;
- `docs/sprint3/CANDIDATE_POOL_TOPK_PROTOCOL.md`;
- `docs/sprint3/DIVERSIFICATION_CONFIGURATION_PROTOCOL.md`;
- `docs/sprint3/SELECTION_STATISTICS_PROTOCOL.md`;
- `docs/sprint3/GENERATION_PROTOCOL.md`;
- `docs/sprint3/ANSWER_CORRECTNESS_PROTOCOL.md`;
- `docs/sprint3/FAITHFULNESS_PROTOCOL.md`;
- `docs/sprint3/ACC_PROTOCOL.md`;
- `docs/sprint3/ASQA_INTERNAL_PARTITION_PROTOCOL.md`;
- `docs/sprint3/ASQA_RETRIEVAL_METRIC_PROTOCOL.md`;
- `docs/sprint3/HOTPOTQA_FINAL_EVALUATION_PROTOCOL.md`.

The stage-gate protocol governs when stages may run. The selection protocol
resolves clustering winners. This protocol governs which canonical final cells
exist. The generation protocol governs how answers are generated. Metric
protocols govern how outputs are measured. The forthcoming contrast registry
governs which statistical comparisons are confirmatory.

## 30. Frozen Before Outcome Observation

This protocol freezes the structural canonical Sprint-3 experiment matrix
before canonical HotpotQA or ASQA `SELECTION` outcomes or protected-final
outcomes are inspected.

No such outcome was used to choose:

- dataset breadth;
- PubMedQA full-control replication;
- retriever set;
- MMR treatment set;
- clustering-family count;
- deterministic DPP condition;
- three-LLM design;
- no-context design;
- evaluation coverage;
- expected generation counts.

The only future performance-derived substitutions permitted are:

```text
KMEANS_SELECTED_K
AGGLO_SELECTED_K
```

through the already-frozen bounded selection protocol.
