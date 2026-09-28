# Sprint 3 Answer-Correctness Protocol

## 1. Purpose

This document defines canonical reference-answer correctness separately from:

- retrieval quality;
- context faithfulness;
- hallucination and groundedness;
- Bidirectional Atomic-Content Coverage (`ACC_bi`) and answer-content change.

Correctness evaluates generated answer content against dataset gold or reference
answers.

This document freezes the scientific constructs. Some exact physical scorer,
checkpoint, and environment provenance remains a Stage-2 implementation gate and
does not reopen the metric choice.

## 2. Cross-Dataset Principle

Use benchmark-native correctness metrics wherever possible.

Do not introduce:

- a generic LLM-as-judge correctness metric;
- generic embedding similarity;
- BERTScore;
- reference NLI;
- one generic lexical scorer across all three datasets.

Use exactly the same dataset-specific correctness definition across all three
generator LLMs.

## 3. PubMedQA Gold

The canonical gold field is `final_decision`.

The allowed labels are:

- `yes`;
- `no`;
- `maybe`.

The explanatory `long_answer` is not the canonical answer-correctness target.
Historical Sprint-1 lexical comparison against `long_answer` is historical only
and must not govern Sprint-3 canonical correctness.

## 4. PubMedQA Primary Correctness

The primary metric is frozen as:

`pubmedqa_decision_accuracy`

For each measurable question, the score is:

- `1` if the parsed `Decision` exactly equals `final_decision`;
- `0` otherwise.

Aggregate the metric as mean categorical accuracy.

Optional descriptive diagnostics may include:

- a three-class confusion matrix;
- per-class recall;
- macro-F1.

These diagnostics are not lead correctness metrics or configuration-selection
objectives. Do not score the `Explanation` with another opaque correctness judge.

## 5. PubMedQA Parser

For an upstream `OK` generation, the expected format is:

```text
Decision: <yes|no|maybe>
Explanation: <1-3 concise factual sentences>
```

Verdict parsing must:

1. normalize line endings;
2. require exactly one canonical `Decision:` field;
3. strip surrounding whitespace from the value;
4. lowercase the ASCII value;
5. accept only the complete value `yes`, `no`, or `maybe`;
6. ignore `Explanation` for categorical scoring;
7. never infer the verdict from explanation prose.

A missing, duplicate, empty, or non-enumerated `Decision` is a
`PARSE_FAILURE`. Do not perform fuzzy or semantic repair.

## 6. PubMedQA Status Policy

- `OK`: parse and score `0` or `1`.
- `REFUSAL`: correctness is `0`; also report refusal separately.
- `PARSE_FAILURE`: correctness is `NA`; report separately.
- `TRUNCATED`: correctness is `NA`; do not score a partial response.
- `ERROR`: correctness is `NA` after governed infrastructure retries.

## 7. HotpotQA Gold

The canonical gold is the exact official HotpotQA `answer` joined to the BEIR
query using the already-frozen exact source-ID mapping. No fuzzy answer mapping
is allowed.

Supporting facts and qrels are retrieval evidence and must not enter answer
correctness. The official dataset supplies one canonical answer string per
example rather than a project-invented alias set.

## 8. HotpotQA Primary Correctness

Official HotpotQA token-level answer F1 is frozen as the primary
answer-correctness metric. Use exact official HotpotQA semantics.

For each measurable question:

1. normalize prediction and gold;
2. tokenize the normalized strings;
3. compute multiset token overlap;
4. compute precision and recall;
5. compute their harmonic-mean F1.

Use the official categorical guard for:

- `yes`;
- `no`;
- `noanswer`.

If either normalized answer is one of these categorical values and prediction
and gold differ, precision, recall, and F1 are zero.

Do not substitute the generic existing repository scorer unless exact parity
with the frozen official scorer is demonstrated.

## 9. HotpotQA Secondary Correctness

Official HotpotQA answer Exact Match is frozen as the sole secondary
answer-correctness metric.

`EM = 1` only when the official-normalized prediction equals the
official-normalized gold.

Do not use supporting-fact or joint answer/supporting-fact metrics as answer
correctness in this study.

## 10. HotpotQA Normalization

Freeze the exact official normalization order:

1. lowercase;
2. remove Python ASCII `string.punctuation`;
3. remove the whole-word English articles `a`, `an`, and `the`;
4. collapse whitespace.

The repository's generic normalization function must not be assumed equivalent
until exact parity tests pass.

## 11. HotpotQA Parser

The expected generation format is:

```text
Answer: <short answer>
Explanation: <1-3 concise factual sentences>
```

For an upstream `OK` response:

- require exactly one canonical `Answer:` field;
- extract only the short-answer value;
- strip surrounding whitespace;
- exclude `Explanation` completely;
- classify a missing, duplicate, or empty `Answer` as `PARSE_FAILURE`;
- never derive an answer from explanation prose.

## 12. HotpotQA Status Policy

- `OK`: score official F1 and EM.
- `REFUSAL`: F1 and EM are `0`; also report refusal separately.
- `PARSE_FAILURE`: F1 and EM are `NA`.
- `TRUNCATED`: F1 and EM are `NA`.
- `ERROR`: F1 and EM are `NA` after governed retries.

Comparisons use exact measurable paired intersections and report paired `N`.

## 13. ASQA Official Answer Semantics

ASQA provides:

- an ambiguous question;
- multiple disambiguated QA pairs;
- a disambiguated question for each aspect;
- multiple official short-answer aliases per QA pair;
- long-answer reference annotations.

Canonical correctness follows benchmark-native answer evaluation rather than
generic long-form lexical similarity.

## 14. ASQA Primary Correctness

Official ASQA Disambig-F1, also named QA-F1 in the released scorer, is frozen as
the primary answer-correctness metric.

Its scientific interpretation is: does the generated long-form answer contain
enough correct answer content for the official disambiguated questions to be
answered?

The generated ASQA `Answer` body is used as the QA evaluator's context. Do not
provide:

- retrieved passages;
- retrieval metadata;
- gold contexts;
- relevance scores.

For every official QA-pair aspect:

1. ask or evaluate the official disambiguated question against the generated
   answer context using the frozen official ASQA QA evaluator;
2. obtain the extracted short answer;
3. compare it with all official short-answer aliases using official normalized
   token F1;
4. use the best alias score for that aspect;
5. average across aspects within the original ASQA question;
6. average question-level scores across the dataset.

Preserve equal original-question weighting.

This is reference-answer correctness and completeness. It is not:

- retrieval S-recall;
- faithfulness;
- ACC;
- an LLM judge.

## 15. ASQA Secondary Diagnostic

Freeze one transparent secondary diagnostic:

`answer_side_alias_coverage`

For each official QA-pair aspect:

- score `1` if at least one official alias is found in the generated answer
  using the frozen deterministic token-bounded ASQA alias-matching contract;
- score `0` otherwise.

Average across aspects within each question, then across questions.

Use the same frozen alias inventory and deterministic normalization architecture
already defined for ASQA retrieval matching, but apply it to the generated
answer surface.

Label this explicitly as **answer-side alias coverage**. Do not call it:

- retrieval S-recall;
- official STR-EM unless implementation is proven exactly equivalent;
- primary correctness.

## 16. Retrieval Versus Answer Coverage

The same official aliases may support two distinct constructs.

Retrieval asks whether the retrieved canonical passage body contains evidence
associated with an aspect.

Answer coverage asks whether the generated answer surface explicitly contains
an official answer alias.

Different evaluation surfaces mean these remain different scientific
constructs. Do not merge them.

## 17. ASQA False-Positive and Negation Policy

Literal alias occurrence alone does not establish correctness. For example,
`The answer is not Paris.` still contains the alias `Paris`.

Therefore answer-side alias coverage is diagnostic only. Official Disambig-F1
remains primary because it conditions extraction on the specific disambiguated
question and surrounding generated answer.

QA-F1 can still make extraction or negation errors. Do not add a new
reference-NLI correctness metric because:

- aliases are not complete semantic reference premises;
- ASQA references are not exhaustive descriptions of all valid claims;
- NLI would require separate threshold calibration;
- the frozen NLI architecture serves context faithfulness, not reference
  correctness.

Unsupported or false extra claims are addressed through the separately frozen
faithfulness and claim analyses.

## 18. ASQA Long-Form Overlap

ROUGE-L may be reported only as optional descriptive reference-wording overlap.
Do not make it primary answer correctness.

Exclude from the canonical correctness family:

- BERTScore;
- embedding similarity;
- MAUVE;
- DR;
- generic LLM judging.

## 19. ASQA Physical Evaluator Provenance

The scientific metric is frozen now.

Before Stage 3, implementation and provenance must additionally freeze:

- the exact corrected ASQA scorer source revision;
- the exact QA-model checkpoint and revision;
- the tokenizer;
- the configuration;
- physical file hashes where feasible;
- a compatible Transformers revision and environment;
- maximum sequence length;
- overflow and window behavior;
- stride, if applicable;
- SQuAD2/no-answer mode;
- null threshold;
- aggregation behavior.

The original ASQA release's unpinned dependency installation is insufficient
for final reproducibility. These are Stage-2 implementation gates, not reasons
to reopen the metric.

## 20. ASQA Human Validation

No new project-specific human calibration is required to admit official
Disambig-F1.

Before Stage 3, require development-only implementation validation that:

- reproduces released scorer behavior on available official or public
  fixtures;
- tests multiple aspects;
- tests multiple aliases;
- tests aggregation;
- tests no-answer behavior;
- documents known negation and extractive-QA limitations.

A local human audit may be descriptive only. It must not tune:

- thresholds;
- aliases;
- checkpoint choice;
- scoring rules.

## 21. Cross-LLM Policy

Use identical parser grammar, reference mapping, normalization, official metric
implementation, ASQA QA evaluator, aggregation, and status handling for:

- `llama-3.3-70b`;
- `gemma4-26b`;
- `ministral-3-14b`.

No model-specific parser exception, alias expansion, normalization, semantic
repair, threshold, or evaluator is allowed.

## 22. Cross-Dataset Status Policy

- `OK`: score normally according to the dataset-specific metric.
- `REFUSAL`: this is an explicit failure to answer the semantic task; score
  correctness as `0` where the metric is defined and retain the `REFUSAL`
  status and rate.
- `PARSE_FAILURE`: `NA`.
- `TRUNCATED`: `NA`.
- `ERROR`: `NA` after governed infrastructure retries.

Never silently replace or regenerate a content failure. Never convert `NA`
statuses to zero. The zero for `REFUSAL` reflects absence of an answer, not
infrastructure missingness.

## 23. Correctness Reporting

For every condition, report:

- expected `N`;
- measured `N`;
- correctness mean or means;
- `REFUSAL` count and rate;
- `PARSE_FAILURE` count and rate;
- `TRUNCATED` count and rate;
- `ERROR` count and rate;
- exact paired-intersection `N` for comparisons.

Missingness and failures must remain visible.

## 24. Primary and Secondary Family

Freeze the compact canonical family as follows.

PubMedQA:

- **Primary:** categorical verdict accuracy.

HotpotQA:

- **Primary:** official answer token F1.
- **Secondary:** official answer EM.

ASQA:

- **Primary:** official Disambig-F1 / QA-F1.
- **Secondary diagnostic:** answer-side alias coverage.

Optional descriptive outputs do not enter configuration selection unless
separately predeclared in the statistical protocol.

## 25. Protected-Final Rule

Correctness constructs and parser/scorer semantics must be frozen and
implementation-validated on `DEVELOPMENT` before Stage 3 selection opens.

Do not use:

- the HotpotQA protected-final 500;
- ASQA dev948

to calibrate or change:

- parsing;
- normalization;
- scorer;
- evaluator checkpoint;
- thresholds;
- aliases;
- aggregation.

No correctness rule may change because selection or protected results look
poor.

## 26. Implementation Gaps

The following remain to be implemented or physically frozen, but are not solved
by this document:

- strict dataset-specific parsers;
- PubMedQA decision-accuracy implementation;
- exact HotpotQA official scorer implementation and parity;
- exact BEIR-ID-to-official-HotpotQA-answer join artifact;
- HotpotQA reference source and revision hashes;
- official ASQA scorer implementation;
- corrected ASQA source revision;
- physical QA evaluator snapshot;
- pinned Transformers environment;
- exact ASQA windowing and no-answer behavior;
- answer-side alias-coverage implementation;
- metric-registry updates;
- correctness artifact schema;
- evaluator-bundle provenance;
- status-aware correctness aggregation;
- official-fixture parity tests.

## 27. Frozen Before Observation

This protocol freezes answer-correctness methodology before selection and
protected-final outcomes are opened.

No selection or protected-final results are observed while creating this
protocol.

Remaining scorer, checkpoint, and environment items are implementation and
provenance closures and do not authorize changing the correctness constructs.
