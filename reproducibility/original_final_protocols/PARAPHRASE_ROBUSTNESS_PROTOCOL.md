# Sprint 3 Paraphrase-Robustness Protocol

## 1. Authority Status

The project kickoff explicitly contains:

> Sub-task 5.4: Robustness Under Paraphrase

with the hypothesis that diversification may improve robustness by reducing
sensitivity to specific query phrasing.

This item appears under the kickoff section "Suggestions for Each Task."
Its frozen authority status is therefore:

- `KICKOFF_SUGGESTION`;
- not a hard professor or supervisor requirement;
- not part of later explicit supervisor instructions; and
- not a core configuration-selection objective.

This bounded pilot addresses the kickoff suggestion without expanding the
canonical factorial experiment.

## 2. Scientific Purpose

The construct is:

```text
QUERY_SURFACE_ROBUSTNESS
```

The question is:

> Holding the underlying information need fixed, how much does a semantically
> equivalent rewording alter (1) first-stage top-20 candidates and (2) the
> final selected top-5 retrieval context?

This pilot does not establish:

- answer robustness;
- correctness robustness;
- faithfulness robustness;
- ACC robustness; or
- generator robustness.

Those claims must not be made from this pilot.

## 3. Statistical Role

This analysis is classified as:

```text
SECONDARY_DESCRIPTIVE_ROBUSTNESS_ANALYSIS
```

It is:

- not configuration selection;
- not part of the protected-final confirmatory family;
- not a reason to change retrievers or diversification;
- not a reason to change candidate-pool size or final top-k; and
- not a new primary research question.

Analysis is paired between each original question and its accepted
paraphrase.

## 4. Evidence Roles

Only the following evidence roles may be used:

### PubMedQA

`HISTORICAL_OBSERVED` control evidence.

### HotpotQA

BEIR-train `DEVELOPMENT` evidence.

### ASQA

Internal `DEVELOPMENT` evidence.

The pilot must never use:

- HotpotQA `SELECTION`;
- ASQA `SELECTION`;
- the HotpotQA protected-final 500; or
- ASQA dev948 protected final.

No robustness result may influence Stage-3 configuration selection.

## 5. Sample Size

The target is frozen as:

```text
30 valid original-paraphrase pairs per dataset
90 valid pairs in total
```

There is exactly one accepted paraphrase per original question.

Thirty pairs per dataset is a project-scale convention for a bounded
descriptive robustness pilot. It is not presented as a literature-derived
power threshold.

## 6. Original Question Sampling

Candidate originals must be selected using deterministic, ID-only SHA-256
ordering.

Selection must not depend on:

- query-text content;
- answers;
- qrels;
- retrieval output;
- metric values; or
- model performance.

A reserve ordering must be frozen before paraphrase creation. Accepted
questions are taken from the ordered list until 30 valid pairs per dataset
have been obtained.

Rejected or failed candidates remain recorded and must not be repeatedly
rewritten.

The exact sampling namespace string and reserve-list size are implementation
and provenance closures that must be frozen before materialization. They are
not materialized by this protocol.

## 7. Paraphrase Creation

Use one human-authored paraphrase per candidate original.

The paraphrase author may see only the original question. The author must not
see:

- the gold answer;
- aliases;
- qrels;
- gold or supporting documents;
- retrieved passages;
- candidate rankings;
- retriever identity;
- diversification method; or
- experimental results.

The paraphrase must preserve:

- the underlying information need;
- entities;
- relations;
- polarity and negation;
- temporal constraints;
- numerical constraints;
- answer granularity; and
- presuppositions.

For ASQA, it must additionally preserve the complete ambiguity and
interpretation structure.

A substantive wording change is required. An unchanged question or a trivial
punctuation or formatting modification is invalid.

## 8. No Iterative Convenience Rewriting

If a paraphrase is rejected as semantically invalid:

- record the rejection and its reason;
- do not repeatedly rewrite that question until it passes; and
- advance to the next candidate in the frozen reserve ordering.

This prevents convenience-selected or cherry-picked paraphrases.

## 9. Semantic-Equivalence Validation

Two human reviewers independently classify each candidate pair as:

- `EQUIVALENT`;
- `NOT_EQUIVALENT`; or
- `UNCERTAIN`.

Reviewers are blind to:

- retrieval results;
- gold answers and qrels;
- retriever identity;
- method identity; and
- experimental performance.

Disagreements must be adjudicated before inclusion. Only adjudicated
`EQUIVALENT` pairs enter retrieval.

For ASQA, reviewers must explicitly confirm that the paraphrase:

- does not collapse an interpretation;
- does not add a new interpretation; and
- preserves the ambiguity structure.

Embedding similarity, lexical similarity, and LLM judging must not replace
the human equivalence decision.

The exact accepted UTF-8 original and paraphrase texts and their hashes must
be frozen.

## 10. Method Scope

Use all four canonical retrievers:

- BM25;
- DPR;
- Contriever; and
- ColBERTv2.

Use only two top-5 context-selection conditions:

1. relevance-only baseline; and
2. MMR with `lambda = 0.50`.

Both conditions use:

```text
candidate_pool = 20
final_top_k = 5
```

MMR `lambda=0.50` is selected prospectively as a representative midpoint
diversification intervention from the already-defined MMR grid. It is not
selected based on performance.

Do not include:

- the full MMR lambda curve;
- KMeans;
- Agglomerative clustering;
- DPP;
- Stage-3 selected configurations; or
- generators.

## 11. Surface-Specific Candidates

The original query and paraphrased query are separate retrieval inputs.

For every:

```text
(dataset, question_pair, query_surface, retriever)
```

produce exactly one first-stage top-20 candidate artifact, where
`query_surface` is one of:

- `ORIGINAL`; or
- `PARAPHRASE`.

For a given query surface, baseline and MMR must consume the same top-20
artifact. Baseline and MMR must not independently rerun first-stage retrieval.

## 12. Primary Robustness Metrics

### Primary Candidate Robustness

Use Jaccard overlap between the original and paraphrase top-20 document-ID
sets. For sets `A` and `B`:

```text
J20 = |A intersection B| / |A union B|
```

Also report the raw intersection count.

### Primary Selected-Context Robustness

Use Jaccard overlap between the original and paraphrase top-5 document-ID
sets. Report this separately for:

- the relevance-only baseline; and
- MMR `lambda=0.50`.

Also report the raw intersection count.

Higher values mean greater invariance to wording. Higher robustness must not
be interpreted as necessarily implying higher retrieval quality.

## 13. Secondary Diagnostics

Report:

### A. Exact Ordered-Top-5 Identity

Report an indicator for whether the original and paraphrase selected top-5
document lists are exactly identical in document IDs and order.

### B. Dataset-Native Retrieval-Effectiveness Change

Where the corresponding development or control judgments are valid, report:

- PubMedQA: paired change in the already-canonical retrieval `Recall@5`
  measure available for its control analysis;
- HotpotQA: paired change in the frozen HotpotQA lead retrieval `Recall@5`;
  and
- ASQA: paired change in the frozen `S-recall@5`.

Do not create new gold-matching rules for this pilot.

If PubMedQA's canonical retrieval metric is unavailable or inapplicable at
implementation time, mark that effectiveness diagnostic unavailable rather
than inventing a new metric.

## 14. No Extra Metric Proliferation

Do not add the following as canonical robustness metrics:

- embedding cosine similarity;
- BERTScore;
- rank-biased overlap;
- multiple rank correlations;
- lexical query similarity; or
- an LLM robustness judge.

They may not be introduced after pilot results have been observed without a
prospective amendment.

## 15. Paired Analysis

The analysis unit is the original-paraphrase question pair.

Report, per dataset, retriever, and baseline or MMR condition where
applicable:

- valid paired N;
- mean and median overlap where useful;
- raw overlap distributions;
- paired retrieval-effectiveness changes; and
- confidence intervals using the final frozen common paired-bootstrap
  implementation.

Do not freeze a separate bootstrap algorithm here while the main statistical
protocol remains open. This pilot inherits the final common paired-bootstrap
resample count and seed once that protocol is frozen.

Do not use robustness p-values as primary evidence. Do not create a new
configuration-selection multiple-testing family.

## 16. Invalid and Failure Policy

Use the explicit statuses:

```text
PARAPHRASE_CREATION_FAILURE
INVALID_SEMANTIC_EQUIVALENCE
RETRIEVAL_FAILURE
```

Semantic invalidity includes:

- meaning change;
- a lost or added constraint;
- polarity change;
- entity or relation change;
- a trivial non-paraphrase;
- ASQA interpretation collapse; or
- ASQA interpretation expansion.

Semantically invalid paraphrases:

- are excluded before retrieval;
- are not counted as robustness failures; and
- trigger movement to the next frozen reserve candidate.

A valid paraphrase that makes retrieval much easier or harder remains
included. That is the phenomenon being measured.

Existing retrieval-failure semantics remain unchanged. Retrieval failures
must never be converted to zero overlap.

## 17. Reporting Counts

For each dataset, report:

- requested candidates;
- paraphrases created;
- creation failures;
- equivalence accepted;
- equivalence rejected;
- uncertain;
- adjudicated;
- valid target pairs;
- retrieval-analyzable pairs; and
- retrieval failures.

There must be no silent replacement or denominator changes.

## 18. Development-Only Policy

This pilot is intentionally restricted to `DEVELOPMENT` and
`HISTORICAL_OBSERVED` evidence. It does not touch protected-final examples.

This restriction is sufficient because the claim is limited to secondary
query-surface robustness and the pilot is not used for final configuration
selection.

Its findings may be discussed as robustness evidence or as a limitation, but
they must not alter canonical methods after Stage 3 opens.

## 19. Experimental Burden

For 90 valid pairs, the planned burden is:

- 90 accepted paraphrases;
- 180 independent semantic-equivalence judgments plus adjudication;
- at most 720 original/paraphrase first-stage retrievals if neither surface
  has existing candidates: `90 * 2 * 4`;
- 1,440 top-5 outputs across baseline and MMR: `90 * 2 * 4 * 2`;
- zero generator calls;
- zero decomposer calls; and
- zero NLI verifier calls.

If original candidate artifacts already validly exist, reuse them by exact
artifact identity and run only the missing paraphrase-side retrieval. Do not
rerun valid originals merely for symmetry.

## 20. Provenance

Preserve:

- protocol version;
- sampling namespace and ordering;
- original sample ID;
- original UTF-8 text and hash;
- paraphrase UTF-8 text and hash;
- authoring status;
- independent equivalence labels;
- adjudicated equivalence;
- invalid or rejection reason;
- retriever and configuration identity;
- original candidate-artifact hash;
- paraphrase candidate-artifact hash;
- baseline and MMR context-artifact hashes;
- metric-implementation version;
- analysis-artifact hash; and
- Git commit and run-registry identity.

Private reviewer identities must not be exposed publicly.

## 21. Implementation Gaps

The following remain to be implemented or frozen later and are not solved by
this document:

- exact sampling namespace;
- reserve-list size;
- paired manifest schema;
- human paraphrase guide;
- equivalence-review guide;
- adjudication schema;
- paired retrieval producer;
- Jaccard metrics;
- ordered-top-5 identity;
- status-aware paired analysis;
- provenance and run-registry integration; and
- tests.

## 22. Frozen Before Observation

This protocol is frozen before paraphrase creation, paraphrase retrieval
results, Stage-3 `SELECTION` outcomes, or protected-final outcomes are
observed.

MMR `lambda=0.50` is a prospective representative robustness treatment and
is not selected using retrieval performance.

The pilot cannot be used to retune canonical methods.

