# Sprint 3 Faithfulness Protocol

## 1. Purpose and Construct

Canonical Sprint-3 faithfulness is **Atomic Claim Context-Groundedness**.

The scientific question is:

> To what extent are the factual claims asserted by a generated
> `WITH_CONTEXT` answer supported by the exact retrieved passage bodies
> supplied to that generation?

Faithfulness is not:

- gold-answer correctness;
- world factuality;
- answer relevance;
- retrieval relevance;
- ASQA aspect coverage;
- lexical overlap;
- citation-format compliance;
- answer-content change or ACC.

A claim can be faithful to context but factually wrong in the world if the
context itself is wrong. A claim can also be correct according to gold or
world knowledge but unfaithful if the supplied context does not support it.

Correctness and faithfulness remain scientifically separate.

The recommended registry label is:

```text
faithfulness_atomic_context_v1
```

The primary answer-level metric is:

```text
Faithfulness@5
```

## 2. Applicability

Canonical faithfulness is defined only for:

```text
WITH_CONTEXT
```

because the construct measures grounding in supplied retrieval evidence.

For `WITHOUT_CONTEXT`, freeze:

```text
Faithfulness@5 = NA
status = NOT_APPLICABLE_NO_CONTEXT
```

Do not assign:

- zero;
- one;
- unsupported;
- hallucinated;

merely because no retrieval context exists. Correctness may still be
evaluated for `WITHOUT_CONTEXT`.

## 3. Evidence Set

For every successful `WITH_CONTEXT` generation, the canonical evidence set is
the exact five passage bodies actually supplied to the generator:

```text
E = {e1, e2, e3, e4, e5}
```

in selected rank order.

Use the exact canonical passage body text.

Exclude:

- the `[Document N]` prompt scaffold;
- title unless title is literally part of the frozen canonical passage body
  for that dataset;
- retriever score;
- document ID;
- qrel;
- gold or supporting-fact annotation;
- ASQA aliases or aspects;
- relevance label;
- method identity;
- diversification identity;
- any passage not supplied to the generator.

Faithfulness must evaluate the evidence the model actually received. Do not
substitute gold or supporting documents that were not in its context.

## 4. Shared Claim Decomposer

Use the same frozen atomic-claim decomposer used for ACC.

The decomposer winner and provenance are governed by:

- `docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL.md`;
- `docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL_AMENDMENT_01.md`;
- any later prospective decomposer amendment.

Do not use a separate faithfulness-specific decomposer. The physical
decomposer winner is a Stage-1 execution result and is not reopened here.

## 5. Exact Decomposer Input

For every eligible answer, decomposer input is exactly:

1. the exact canonical dataset question text; and
2. the dataset-specific faithfulness answer-content surface defined below.

The question is non-evidential context used only for self-contained reference
resolution according to the frozen decomposer contract.

The decomposer must never receive:

- retrieved passages;
- context bodies;
- gold answers;
- qrels;
- ASQA aliases or aspects;
- correctness results;
- ACC results;
- verifier results;
- retriever identity;
- diversification identity.

The question itself must not become a factual claim unless that factual
content is actually asserted by the answer surface.

## 6. Dataset-Specific Faithfulness Content Surface

### PubMedQA

Use the parsed Explanation only.

Do not decompose:

```text
Decision: yes|no|maybe
```

for faithfulness. The categorical verdict is evaluated by correctness. A bare
yes, no, or maybe verdict is not a self-contained factual proposition for
context-groundedness.

### HotpotQA

Use the parsed Explanation only.

Do not turn the bare short:

```text
Answer: <short answer>
```

field into an invented self-contained proposition using information from the
question. The short Answer field is evaluated by official correctness. The
requested Explanation exists specifically to provide factual rationale claims
that can be grounded against context.

### ASQA

Use the complete canonical long-form Answer content. No gold aliases or
aspects are supplied to the decomposer.

## 7. Zero-Claim Answers

If successful decomposition legitimately yields zero factual claims after the
frozen exact-deduplication rule:

```text
Faithfulness@5 = NA
status = NO_FACTUAL_CLAIMS
```

Do not assign faithfulness zero. This includes legitimate cases where there is
simply no measurable factual-claim surface.

Zero claims are not equivalent to:

- evaluator failure;
- decomposer failure;
- generation failure.

Report zero-claim frequency by dataset and experimental condition.

## 8. Exact-Duplicate Claim Handling

Use the same deterministic exact-deduplication rule as ACC.

After successful decomposition:

- deduplicate exact duplicate claim strings within the answer before
  verification;
- exact means equality of parsed canonical `claim_text`;
- preserve the first occurrence;
- discard later exact duplicates.

Record:

- raw claim count;
- deduplicated claim count;
- exact duplicates removed.

Do not:

- casefold for deduplication;
- semantically deduplicate;
- normalize punctuation specially for deduplication;
- use embeddings;
- use NLI to merge claims;
- use another LLM to merge claims.

Do not apply an additional subjective runtime trivial-claim filter.
Valid-schema factual claims are retained literally.

## 9. Verifier

Use the same verifier model and revision used for ACC, governed by
`docs/sprint3/NLI_VERIFIER_CALIBRATION_PROTOCOL.md`.

The logical verifier family is:

```text
MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
```

Faithfulness uses its own calibrated:

```text
faithfulness entailment threshold:     tau_faith_ent
faithfulness contradiction threshold:  tau_faith_con
```

These thresholds are separate from ACC thresholds.

Do not create:

- dataset-specific faithfulness thresholds;
- LLM-specific thresholds;
- retriever-specific thresholds;
- diversification-specific thresholds;
- pair-specific thresholds.

The physical snapshot and numeric thresholds are later calibration outputs,
not open methodology here.

## 10. NLI Orientation

For every evidence check:

```text
premise    = exact canonical evidence passage body
hypothesis = original natural-language atomic claim
```

For two-passage support checks:

```text
premise    = exact deterministic serialization of the two passage bodies
hypothesis = original atomic claim
```

Do not reverse premise and hypothesis.

Do not normalize the claim using:

- ASQA matcher normalization;
- stemming;
- lemmatization;
- fuzzy matching;
- alias replacement;
- LLM rewriting.

Use the decomposer's original natural claim text.

## 11. Single-Passage Verification

Stage 1 checks the claim against each of the five individual passage bodies:

```text
e1
e2
e3
e4
e5
```

All five single-passage checks are required.

For each single passage, compute:

```text
p_entail
p_contradiction
```

using the frozen faithfulness thresholds.

For each single check, define signals:

```text
support_signal =
p_entail >= tau_faith_ent

contradiction_signal =
p_contradiction >= tau_faith_con
```

The two signals are independent. A single passage may therefore trigger:

- support only;
- contradiction only;
- both;
- neither.

Do not infer contradiction merely because entailment fails.

## 12. Single-Passage Failure

If any of the five required single-passage checks fails because of:

- inference exception;
- non-finite probabilities;
- tokenizer failure;
- malformed verifier call;
- provenance mismatch;
- `FAILED_OVERLENGTH`;
- evaluator-capacity failure;

the claim verification is `FAILED`.

Do not classify that claim using only the remaining passages because a failed
single passage could contain either missing support or missing contradiction.

A failed claim is excluded from the numeric faithfulness denominator and
counted explicitly.

Do not map failure to:

- unsupported;
- neutral;
- contradicted;
- zero.

## 13. Single-Stage Aggregation

After all five single-passage checks succeed, define:

```text
single_support =
TRUE if at least one passage has support_signal

single_contradiction =
TRUE if at least one passage has contradiction_signal
```

If:

```text
single_support = TRUE
and
single_contradiction = TRUE
```

then:

```text
claim_status = INCONSISTENT
```

and stop.

If:

```text
single_support = TRUE
and
single_contradiction = FALSE
```

then:

```text
claim_status = SUPPORTED
```

and stop.

Only when:

```text
single_support = FALSE
```

proceed to the two-passage support stage.

## 14. Why Pairs Exist

The pair stage handles factual claims whose support requires information
distributed across two retrieved passages.

It is not:

- another contradiction detector;
- a search over arbitrary subsets;
- a query-dependent evidence-selection method;
- a way to concatenate the whole context.

No triples or larger evidence groups are permitted.

There are exactly:

```text
C(5,2) = 10
```

unordered passage pairs.

## 15. Pair Order

Evaluate unordered pairs in deterministic lexicographic rank order:

```text
(1,2)
(1,3)
(1,4)
(1,5)
(2,3)
(2,4)
(2,5)
(3,4)
(3,5)
(4,5)
```

Within each pair, the lower-ranked passage number is serialized first.

Do not reorder pairs based on:

- retriever scores;
- lexical similarity;
- NLI probabilities;
- dataset;
- method.

## 16. Exact Pair Serialization

For pair `(i,j)`, where `i < j`, the verifier premise is exactly:

```text
<body_i>

<body_j>
```

Operationally, this means:

```text
body_i
+
exactly two newline characters
+
body_j
```

There are:

- no `[Document N]` labels;
- no `[Passage N]` labels;
- no `[SEP]` textual marker;
- no title;
- no document IDs;
- no scores;
- no added metadata.

Only the two actual evidence bodies are supplied, in rank order.

The exact serialized bytes and hash must be recorded for validation and
provenance. The separate pair-support validation in
`docs/sprint3/NLI_VERIFIER_CALIBRATION_PROTOCOL.md` must use this exact
serialization.

## 17. Pair Stage Is Support-Only

For pair checks, only the calibrated faithfulness entailment threshold is used
for the scientific decision:

```text
pair_support =
p_entail >= tau_faith_ent
```

Do not use pair contradiction outputs to label a claim contradicted.
Contradiction evidence is defined from individual passage bodies only.

Reason: a contradiction should be attributable to an actual supplied evidence
unit, while the pair stage exists solely to detect distributed support.

Do not create a separate pair threshold.

## 18. Pair Overlength and Failure Policy

Faithfulness does not use ACC-style windows.

Do not truncate evidence bodies. Do not sentence-window a single passage. Do
not window passage pairs.

If a single passage plus claim exceeds verifier capacity:

```text
FAILED_OVERLENGTH
```

for that required single check; therefore, the claim is `FAILED`.

If a pair plus claim exceeds verifier capacity:

```text
FAILED_OVERLENGTH
```

for that pair check.

Continue through later deterministic pairs because a later valid pair may
still establish support.

If a later valid pair establishes support, an earlier pair failure does not
prevent classification because pair-stage contradiction is not used.

If no valid pair establishes support and one or more required pair checks
failed:

```text
claim_status = FAILED
```

because missing pair evidence could have contained support.

If all required pairs complete successfully and none supports, continue to
the final claim classification.

No pair failure may be silently treated as no support.

## 19. Lazy Pair Stopping

Pairs are checked in the frozen order above.

As soon as one valid pair satisfies:

```text
p_entail >= tau_faith_ent
```

pair support is established. Stop evaluating later pairs for that claim.

No later pair is needed because:

- the pair stage is support-only;
- contradiction has already been fully assessed from all five singles.

Record:

- the pair that first established support;
- number of pair checks attempted;
- any preceding pair failures.

## 20. Final Claim Status

After singles and, where required, pairs:

### SUPPORTED

Support exists from at least one single passage or at least one valid passage
pair, and there is no single-passage contradiction signal.

### CONTRADICTED

No single or pair support exists, all required verification needed to
establish absence of support succeeded, and at least one single passage has a
contradiction signal.

### INCONSISTENT

Support exists from a single passage or a passage pair, and at least one
individual passage has a contradiction signal.

### UNSUPPORTED

There is no single support, no pair support, no individual contradiction, and
all required checks completed successfully.

### FAILED

The claim cannot be validly assigned one of the four scientific states because
the required verification path contains unresolved evaluator failure.

The four scientific measured states are exactly:

```text
SUPPORTED
CONTRADICTED
INCONSISTENT
UNSUPPORTED
```

`FAILED` is instrumentation missingness, not a fifth semantic state.

## 21. Answer-Level Metric

Let:

```text
N_supported
N_contradicted
N_inconsistent
N_unsupported
```

be successfully verified claim counts.

Let:

```text
N_success =
N_supported
+ N_contradicted
+ N_inconsistent
+ N_unsupported
```

Let:

```text
N_failed
```

be failed claim verifications.

For `N_success > 0`, define:

```text
Faithfulness@5 =
N_supported / N_success
```

Also report:

```text
Supported Rate     = N_supported    / N_success
Contradiction Rate = N_contradicted / N_success
Inconsistency Rate = N_inconsistent / N_success
Unsupported Rate   = N_unsupported  / N_success
```

Therefore:

```text
Supported Rate
+ Contradiction Rate
+ Inconsistency Rate
+ Unsupported Rate
= 1
```

Do not fold `INCONSISTENT` into contradiction or support.

Do not define:

```text
Faithfulness@5 = 1 - Unsupported Rate
```

because contradiction and inconsistency are separate measured states.

## 22. Failed-Claim Denominator

Failed claim verifications are excluded from `N_success`. This is intentional
and must be visible.

For every answer, report:

```text
N_raw_claims
N_deduplicated_claims
N_success
N_failed
```

and:

```text
verification_coverage =
N_success / N_deduplicated_claims
```

when `N_deduplicated_claims > 0`.

If:

```text
N_success = 0
```

then:

```text
Faithfulness@5 = NA
```

even if one or more claims existed but every verification failed.

Never silently report a numeric zero. The later paired statistical protocol
governs measured-pair intersections.

## 23. Generation Status Policy

Faithfulness requires a complete `WITH_CONTEXT` answer.

For `OK`, run the decomposer and faithfulness pipeline.

For `REFUSAL`, freeze:

```text
Faithfulness@5 = NA
status = NOT_APPLICABLE_REFUSAL
```

Do not treat refusal as a perfectly faithful answer merely because it contains
no factual claims. Do not assign zero. Do not decompose refusal wording merely
to manufacture claims.

For:

```text
PARSE_FAILURE
TRUNCATED
ERROR
```

freeze:

```text
Faithfulness@5 = NA
```

with the corresponding generation-failure status.

Do not evaluate a truncated partial output as if it were a complete canonical
answer. Do not convert generation failures to unsupported claims or numeric
zero.

## 24. Dataset-Specific Interpretation

### PubMedQA

- Faithfulness applies to Explanation claims only.
- Decision correctness is separate.
- Biomedical qualifiers, negation, quantities, association-versus-causation,
  and uncertainty wording must remain literal in claims.
- Do not weaken scientific hedging during preprocessing.

### HotpotQA

- Faithfulness applies to Explanation claims only.
- Short Answer correctness remains external.
- Pair support is particularly relevant because explanatory claims may
  require evidence distributed across multiple retrieved passages.

### ASQA

- Faithfulness applies to atomic factual claims from the full long-form
  Answer.
- ASQA alias or aspect matching is not part of faithfulness.
- A claim may be relevant to an ASQA aspect yet unsupported by retrieved
  context; coverage and faithfulness remain separate.

## 25. No Gold Information

Faithfulness verification must never use:

- reference answers;
- PubMedQA decision label;
- HotpotQA answer;
- HotpotQA supporting-fact annotation;
- ASQA short-answer aliases;
- ASQA aspect judgments;
- qrels;
- correctness results.

Only:

```text
generated factual claim
versus
actual supplied context bodies
```

determines faithfulness.

## 26. Relation to ACC

ACC and faithfulness share:

- atomic decomposer;
- verifier model and revision;
- evaluator-bundle provenance.

They use different premises:

```text
ACC:          opposing generated answer
Faithfulness: retrieved passage body or allowed two-passage support pair
```

They use separately calibrated NLI thresholds.

ACC measures content change. Faithfulness measures context grounding. Do not
infer one from the other.

Because the instruments are shared, report this shared-instrument dependency
as a limitation when analyzing relationships between ACC and faithfulness.

## 27. NLI Calibration and Validation

Do not redefine calibration here. Inherit exactly from
`docs/sprint3/NLI_VERIFIER_CALIBRATION_PROTOCOL.md`, including:

- faithfulness 150 fitting units;
- faithfulness 60 disjoint held-out units;
- separate entailment and contradiction thresholds;
- point precision target 0.90;
- fitting predicted-positive `N >= 20`;
- held-out predicted-positive `N >= 15`;
- Wilson intervals for uncertainty reporting;
- no hidden Wilson lower-bound gate;
- one permitted fresh `DEVELOPMENT` expansion;
- two blinded annotators and adjudication;
- kappa below 0.70 as a review trigger, not sole hard failure.

Pair support additionally inherits the separate 30-case `DEVELOPMENT`
pair-support validation.

Use the same faithfulness entailment threshold. There is no pair-specific
threshold.

The pass rule is at least 27 of 30 according to the already-frozen NLI
calibration protocol.

Do not combine this set with the ACC window-validation set.

## 28. No Faithfulness Windows

This distinction from ACC is explicit.

ACC may use its frozen deterministic answer-window fallback. Faithfulness does
not.

Faithfulness evidence units are:

- an exact passage body;
- an exact two-passage serialized pair.

If they exceed capacity, record `FAILED_OVERLENGTH` according to Section 18.

Do not truncate, summarize, window, or retrieve alternative evidence.

## 29. Role in Method Selection

Faithfulness is a downstream quality and non-harm guardrail, not a diversity
objective.

Do not maximize faithfulness alone to select clustering `k`.

Do not select a method merely because it has:

- more unsupported claims;
- fewer claims;
- more refusals;
- higher faithfulness caused by answering less.

The final `SELECTION_STATISTICS_PROTOCOL.md` will define the exact prospective
non-inferiority margin and Stage-B decision rule.

Do not freeze a numeric faithfulness margin in this document. That numeric
substantive margin remains professor or supervisor dependent.

## 30. Paired Analysis

Canonical scientific comparisons include, as later registered:

```text
relevance-only WITH_CONTEXT
versus
diversified WITH_CONTEXT
```

for the same:

- dataset;
- question;
- retriever;
- LLM;
- generation protocol.

Use question-level pairing.

The later statistical protocol defines:

- complete-case family policy;
- bootstrap method;
- confidence level;
- margin;
- multiplicity handling.

Do not independently choose those here. `WITHOUT_CONTEXT` has no faithfulness
value and cannot enter a numeric faithfulness difference.

## 31. Reporting

At minimum, report per experimental cell:

- expected `WITH_CONTEXT` answers;
- eligible `OK` answers;
- refusal count;
- generation-failure counts by status;
- zero-claim count;
- total raw claims;
- total deduplicated claims;
- successfully verified claims;
- failed claims;
- verification coverage;
- supported claims;
- contradicted claims;
- inconsistent claims;
- unsupported claims;
- mean and median `Faithfulness@5` where defined;
- number of answers with defined `Faithfulness@5`.

Do not let dataframe or aggregation defaults silently drop `NA` rows without
reporting denominators.

Historical Sprint-2 faithfulness values are legacy proxies and must not be
relabeled as this canonical metric.

## 32. Provenance

Every canonical faithfulness artifact must bind or reference:

- dataset;
- evidence role;
- `sample_id`;
- retriever;
- diversification condition;
- LLM;
- generation artifact and hash;
- generation status;
- exact faithfulness content-surface hash;
- exact five evidence passage-body hashes and order;
- decomposer logical ID;
- decomposer physical snapshot or revision;
- decomposer protocol and version;
- raw claims and hash;
- deduplicated claims and hash;
- verifier logical ID;
- verifier immutable snapshot;
- tokenizer identity;
- faithfulness entailment threshold;
- faithfulness contradiction threshold;
- every single-passage premise hash;
- every pair-premise hash actually attempted;
- single and pair probabilities;
- per-check statuses;
- per-claim scientific status;
- per-claim failure status, if applicable;
- first supporting evidence rank or pair, where applicable;
- answer-level counts and rates;
- `Faithfulness@5`;
- verification coverage;
- evaluator-bundle ID;
- Git commit;
- run-registry identity.

## 33. What Not to Use

Do not use as canonical faithfulness:

- BLEU;
- ROUGE;
- BERTScore;
- embedding similarity;
- lexical overlap;
- answer-reference similarity;
- generic LLM judge;
- retrieval relevance;
- gold correctness;
- ASQA alias coverage;
- historical BART max-document proxy.

Do not silently fall back to:

```text
facebook/bart-large-mnli
```

or any Sprint-1 or Sprint-2 faithfulness implementation.

Canonical implementation must fail fast if the frozen evaluator bundle is not
available.

## 34. Relation to Other Protocols

This protocol must be read with:

- `docs/sprint3/GENERATION_PROTOCOL.md`;
- `docs/sprint3/ACC_PROTOCOL.md`;
- `docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL.md`;
- `docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL_AMENDMENT_01.md`;
- `docs/sprint3/NLI_VERIFIER_CALIBRATION_PROTOCOL.md`;
- `docs/sprint3/ANSWER_CORRECTNESS_PROTOCOL.md`;
- `docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL.md`.

`GENERATION_PROTOCOL.md` defines output fields, statuses, and actual context.
`ACC_PROTOCOL.md` defines output-content change separately. The decomposer
protocol defines the shared claim-extraction instrument. The NLI calibration
protocol defines verifier, calibration, and integration gates. Correctness
remains independent.

## 35. Frozen Before Observation

This document records and consolidates the canonical Sprint-3 faithfulness
methodology before Stage-3 `SELECTION` or protected-final outcomes are
observed.

No `SELECTION` or protected-final result was used to choose:

- the claim-grounding construct;
- content surfaces;
- singles-to-pairs architecture;
- pair order;
- pair serialization;
- support and contradiction semantics;
- answer-level formula;
- zero-claim handling;
- failure denominator;
- generation-status treatment.

## 36. Implementation and Execution Items Not Solved Here

The following are downstream work, not open faithfulness methodology:

- decomposer bake-off and winner;
- immutable verifier snapshot acquisition;
- human NLI calibration;
- numeric threshold fitting;
- held-out threshold validation;
- pair-support 30-case validation;
- evaluator implementation;
- artifact schemas;
- tests;
- metric-registry synchronization;
- evaluator-bundle materialization;
- statistical-margin adjudication.

These items do not reopen the faithfulness construct unless a frozen
validation gate formally fails.
