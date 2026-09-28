# Sprint 3 ASQA Retrieval-Metric and Alias-Matcher Protocol

## 1. Purpose

This protocol defines the canonical ASQA passage-level retrieval-diversity
measurement instrument.

One common binary passage-aspect judgment:

```text
J_q(d,i)
```

must drive:

- `S-recall@5`;
- `alpha-nDCG@5`;
- corpus coverability;
- `c*`.

Do not create separate matching implementations for those metrics.

The scientific retrieval question is:

> Does the selected top-5 context expose more distinct legitimate ASQA
> interpretations or aspects than relevance-only retrieval?

Keep this separate from:

- generated-answer correctness;
- answer-side alias coverage;
- faithfulness;
- ACC;
- embedding-space diversity.

## 2. ASQA Aspect Definition

For question `q`:

```text
A_q = the set of official ASQA qa_pairs
```

Exactly one official `qa_pair` corresponds to one gold aspect.

For aspect `i`:

```text
Aliases(q,i) =
all official non-empty strings in qa_pairs[i].short_answers
```

Do not invent:

- new aliases;
- synonyms;
- abbreviations;
- translations;
- semantic expansions;
- manually repaired aliases.

Do not merge distinct `qa_pairs` merely because their aliases overlap. One
passage may cover multiple aspects.

## 3. Official Alias Retention Policy

Retain all official non-empty aliases.

Do not remove an official alias merely because it is:

- short;
- one token;
- a common word;
- a function word;
- potentially ambiguous;
- numerically formatted;
- an acronym.

Specifically, do not introduce:

- a two-or-fewer-character exclusion;
- a stopword or function-word exclusion list;
- a manually curated unsafe-alias list.

Those rules would alter the benchmark's official aspect-evidence inventory.
False-positive risk for short aliases is instead handled by the frozen case
policy below.

If an alias becomes zero tokens under the frozen normalization:

- exclude that individual normalized alias from automated matching;
- log it with reason `ZERO_TOKEN_ALIAS`;
- preserve the raw official alias in provenance.

If multiple official aliases for the same aspect become exactly identical
after normalization:

- deterministically deduplicate the normalized token sequence for matching;
- preserve all corresponding raw aliases in provenance.

## 4. Canonical Corpus and Passage Surface

The primary canonical ASQA retrieval corpus is the standard DPR English
Wikipedia passage collection.

```text
snapshot lineage:       2018-12-20
canonical passage unit: non-overlapping approximately 100-word DPR passages
expected corpus count:  21,015,324
```

The exact source revision, count, and hash still require physical acquisition
verification before execution.

All four retrievers must operate over the same logical passage universe:

- BM25;
- DPR;
- Contriever;
- ColBERTv2.

Physical indexes may differ.

For every `CorpusRecord`, freeze:

```text
CorpusRecord.text
==
CorpusRecord.retrieval_content
==
exact canonical DPR passage BODY
```

The matcher searches only the exact canonical passage body.

Do not include:

- title;
- title plus body;
- question text;
- neighboring passages;
- gold contexts;
- ASQA annotation contexts;
- retrieved metadata.

This body-only rule must be identical across all four canonical retrievers and
the ASQA matcher.

## 5. Binary Judgment

Define:

```text
J_q(d,i) = 1
```

if and only if at least one surviving normalized official alias belonging to
aspect `i` occurs as an exact contiguous normalized token sequence in the
exact canonical passage body `d`.

Otherwise:

```text
J_q(d,i) = 0
```

Properties:

- the judgment is binary only;
- there is no partial credit;
- repeated occurrence of the same alias does not add credit;
- multiple aliases matching the same aspect still yield `J=1`;
- one passage can have `J=1` for multiple aspects.

Do not use unrestricted substring matching.

## 6. Normalization Philosophy

Freeze a conservative deterministic matcher.

The goal is not maximal recall. The priority is to avoid creating false
passage-aspect positives through aggressive normalization.

Reject the earlier SQuAD-style proposal that would use:

- article removal;
- accent or diacritic stripping;
- blanket lowercasing or casefolding for all aliases;
- title-plus-body matching.

Those are not canonical Sprint-3 ASQA retrieval methodology.

## 7. Base Normalization

Apply the same base normalization to an alias string and passage body before
token matching, in this order:

1. Unicode-normalize to NFC.
2. Before character-category processing, apply only these exact apostrophe
   mappings:

   ```text
   U+2018 LEFT SINGLE QUOTATION MARK
     -> U+0027 APOSTROPHE

   U+2019 RIGHT SINGLE QUOTATION MARK
     -> U+0027 APOSTROPHE

   U+02BC MODIFIER LETTER APOSTROPHE
     -> U+0027 APOSTROPHE

   U+FF07 FULLWIDTH APOSTROPHE
     -> U+0027 APOSTROPHE
   ```

   Do not add other apostrophe-like Unicode characters automatically. Any
   future expansion of this map is a matcher-version change and requires a
   prospective amendment before `SELECTION` or protected observation. Record
   the exact map as part of matcher provenance.
3. Process Unicode characters conservatively. Convert Unicode punctuation,
   symbol, and separator characters to ASCII space U+0020, including the
   categories:

   ```text
   Pd
   Po
   Ps
   Pe
   Pi
   Pf
   Pc
   Sm
   Sc
   Sk
   So
   Zs
   Zl
   Zp
   ```

   After the exact mapping above, retain ASCII apostrophe U+0027 only when it
   occurs between alphanumeric characters. For example, retain the apostrophe
   in `O'Connor` and `France's`. Otherwise it becomes a separator under the
   punctuation rule.
4. Collapse consecutive ASCII spaces to one.
5. Strip leading and trailing spaces.
6. Tokenize by splitting on literal ASCII space U+0020 and discard empty
   tokens.

## 8. Normalization Operations Explicitly Prohibited

Do not perform:

- article removal;
- accent stripping;
- diacritic stripping;
- NFKC compatibility folding;
- stemming;
- lemmatization;
- abbreviation expansion;
- synonym expansion;
- word-number conversion;
- semantic numeric normalization;
- date normalization;
- entity linking;
- fuzzy matching;
- edit-distance matching;
- embedding similarity;
- LLM matching;
- alias invention.

Punctuation becomes separators rather than being deleted or glued. For
example, a hyphen creates a token boundary rather than concatenating its two
sides.

Accept conservative false negatives rather than creating undocumented
semantic equivalences.

## 9. Case Policy

This rule is load-bearing.

Determine alias token count after the base normalization above but before any
casefold operation.

If the normalized alias contains exactly one token, use case-sensitive exact
token matching. The corresponding passage-matching surface must remain
case-preserving.

Examples of the motivation include distinctions such as:

```text
May
versus
may

US
versus
us
```

Do not create semantic-disambiguation logic beyond literal case preservation.

If the normalized alias contains two or more tokens, apply Unicode `casefold`
to every alias token and to the corresponding passage tokens before exact
contiguous-sequence matching. Thus, multi-token aliases are case-insensitive
using Unicode `casefold`.

Do not use `lower()` as a substitute for the frozen `casefold` operation. Do
not switch a one-token alias to case-insensitive matching based on observed
ASQA performance.

## 10. Matching Rule

Let normalized alias tokens be:

```text
a = (a_1, ..., a_m)
```

and normalized passage tokens be:

```text
t = (t_1, ..., t_n)
```

A match exists if and only if:

```text
m > 0
```

and there exists a start position `s` such that:

```text
(t_s, ..., t_{s+m-1})
==
(a_1, ..., a_m)
```

under the applicable one-token or multi-token case policy.

The sequence must be:

- contiguous;
- in order;
- token exact.

Do not use:

- bag-of-token matching;
- unordered matching;
- raw substring matching;
- semantic matching.

## 11. Sparse J Representation

`J` is conceptually defined for the complete frozen corpus. Implementation
does not need to materialize a dense:

```text
question x aspect x 21M-passage
```

tensor.

A canonical sparse positive-hit representation is allowed.

For every positive hit, preserve at least:

- question or sample ID;
- aspect index or ID;
- raw official aliases;
- normalized surviving aliases and hash;
- passage ID;
- exact passage-body hash;
- matcher version and hash;
- `J=1`.

`J=0` is implied for all other passage-aspect combinations in the frozen
question, aspect, and corpus universe.

This sparse representation must remain sufficient to derive:

- coverability;
- S-recall;
- alpha-nDCG;
- `c*`.

## 12. Gold and Coverable Aspect Counts

For each question `q`, define:

```text
G_q = |A_q|
```

as the total number of official gold aspects.

Define corpus-coverable aspects:

```text
A_q_plus =
{
  i in A_q :
  exists d in D such that J_q(d,i)=1
}
```

where `D` is the complete currently authorized frozen ASQA corpus.

Define:

```text
B_q = |A_q_plus|
U_q = G_q - B_q
```

and, for `G_q > 0`:

```text
rho_q = B_q / G_q
```

where `rho` is corpus coverability.

Report distributions of:

- `G_q`;
- `B_q`;
- `U_q`;
- `rho_q`.

Do not silently hide uncoverable aspects.

## 13. Why Uncoverable Aspects Are Separate

An aspect may be legitimate in ASQA but absent from the frozen DPR-2018 corpus
under the conservative matcher.

That is corpus, snapshot, or measurement coverability, not automatically
retriever failure. Do not blame a retriever for an aspect that has no
recognized passage anywhere in its frozen corpus.

At the same time, do not erase this limitation. Always report coverability
alongside conditional S-recall.

## 14. Lead Metric: S-Recall@5

For a ranking:

```text
R_q^5 = (d_1, ..., d_5)
```

define retrieved coverable aspects:

```text
C_q@5 =
{
  i in A_q_plus :
  at least one r in {1,...,5} has J_q(d_r,i)=1
}
```

For `B_q > 0`, freeze:

```text
SRecall@5(q) =
|C_q@5| / B_q
```

This is the lead ASQA retrieval-diversity effectiveness metric.

Behavior is set-like:

- one passage covering three distinct aspects covers all three;
- three passages covering the same aspect cover that aspect once;
- top-5 ordering does not affect `SRecall@5`.

If `B_q = 0`, then:

```text
SRecall@5 = NA
```

not zero.

Report the number and proportion of `B_q=0` questions.

## 15. Absolute Gold Coverage Identity

For `G_q > 0` and `B_q > 0`:

```text
absolute_gold_coverage@5 =
|C_q@5| / G_q
```

and:

```text
absolute_gold_coverage@5
=
rho_q * SRecall@5(q)
```

This may be reported as a diagnostic identity. Do not create it as another
headline optimization metric. Its purpose is to make corpus limitations
visible.

## 16. Secondary Metric: Alpha-nDCG@5

Use the same `J` matrix.

For aspect `i` at ranking position `r`, define:

```text
c_(i,r-1) =
sum over earlier positions s < r of J_q(d_s,i)
```

For `alpha` in `[0,1]`:

```text
gain_alpha(d_r | d_<r) =
sum over aspects i in A_q of:

J_q(d_r,i) * (1-alpha)^(c_(i,r-1))
```

Then:

```text
alpha-DCG@5 =
sum r=1..5:

gain_alpha(d_r | d_<r)
/
log2(r+1)
```

A passage may contribute gain for multiple aspects. Repeated coverage receives
geometrically decreasing gain.

## 17. Alpha Values

Freeze canonical:

```text
alpha = 0.5
```

for the mandatory secondary `alpha-nDCG@5`.

Freeze sensitivity analyses:

```text
alpha = 0.3
alpha = 0.7
```

These are sensitivity settings only. Do not select whichever alpha makes a
method look best. Do not use alpha 0.3 or 0.7 to choose configurations.

## 18. Alpha-nDCG Ideal Denominator

Define:

```text
alpha-nDCG@5 =
alpha-DCG@5(system ranking)
/
alpha-IDCG@5
```

when ideal gain is greater than zero.

Construct the ideal list using:

- the same complete frozen ASQA corpus;
- the same `J` matrix;
- not a retriever-specific top-20 candidate pool.

Use the canonical greedy ideal-list convention:

1. At each rank, select the remaining passage with maximum current marginal
   alpha gain.
2. Break ties deterministically by canonical corpus-manifest passage order.
3. Continue until five passages are selected or no positive gain remains.

The implementation must eventually be regression-tested on a known diversity
ranking fixture or trusted alpha-nDCG reference behavior.

If:

```text
alpha-IDCG@5 = 0
```

then:

```text
alpha-nDCG@5 = NA
```

not zero.

## 19. Alpha=0 Interpretation

At `alpha=0`, the repeated-aspect discount disappears.

Passage graded relevance becomes:

```text
rel_q(d) =
number of ASQA aspects covered by d
```

Therefore alpha-nDCG reduces to ordinary graded nDCG under that aspect-count
relevance definition.

Do not claim it becomes binary nDCG unless every passage covers at most one
aspect.

## 20. C-Star Diagnostic

For each passage `d`, define its coverable-aspect set:

```text
C_q(d) =
{
  i in A_q_plus :
  J_q(d,i)=1
}
```

For `B_q > 0`, define:

```text
c*(q) =
minimum number of passages from the complete frozen corpus whose union of
coverage sets equals A_q_plus
```

This is an exact unweighted minimum set-cover diagnostic.

It represents the minimum number of canonical passages necessary to expose
all recoverable ASQA aspects under the frozen matcher and corpus.

## 21. C-Star Must Be Exact

Do not compute canonical `c*` from:

- BM25 top-20;
- DPR top-20;
- Contriever top-20;
- ColBERT top-20;
- selected top-5;
- gold contexts only.

It is a collection-level diagnostic.

Compute it once per:

```text
question x frozen corpus/matcher version
```

Implementation may collapse passages with identical non-empty
aspect-coverage signatures because they are equivalent for the set-cover
cardinality problem.

Use an exact algorithm, such as:

- bitmask dynamic programming;
- breadth-first state search;
- exact integer programming;

provided exact optimality is verified.

Do not silently substitute greedy set cover and still call the result `c*`.

If exact `c*` proves computationally infeasible, stop and prospectively amend
the protocol before canonical protected evaluation.

If `B_q = 0`, then:

```text
c* = NA
```

## 22. C-Star Reporting

Report `c*` jointly with `B_q`.

At minimum, report:

- median `c*` where defined;
- distribution of `c*`;
- proportion with `c*>1`;
- proportion with `B_q>=2` and `c*=1`;
- proportion with `B_q>=2` and `c*>1`.

Interpretation:

- `c*=1` with `B_q=1` is trivial;
- `B_q>=2` and `c*=1` means one canonical passage already covers every
  recoverable aspect;
- `B_q>=2` and `c*>1` demonstrates genuine collection-level multi-passage
  diversification opportunity.

Do not interpret `c*` alone without coverability.

## 23. One J Matrix, Multiple Metrics

The metric layer must consume the same frozen `J` judgments.

Do not allow S-recall, alpha-nDCG, or `c*` to have:

- different alias lists;
- different normalization;
- different title or body policy;
- different case policy;
- different corpus universe.

A matcher change changes all four:

- coverability;
- S-recall;
- alpha-nDCG;
- `c*`.

Therefore, a scientifically meaningful matcher change requires a prospective
protocol amendment and complete version and hash change.

## 24. Development-Only Matcher Validation

Validate the matcher only using ASQA internal `DEVELOPMENT` from the frozen
3482/871 train partition.

Never use:

- ASQA `SELECTION` 871;
- ASQA protected dev948.

Freeze:

```text
validation target = exactly 150 passage-aspect cases
double-reviewed shared subset = exactly 30 cases
```

These are project-scale instrument-validation conventions, not statistical
power claims or universal standards.

Use the following sequence:

1. Implement the already-frozen matcher on ASQA internal `DEVELOPMENT` only.
2. Automated matcher execution may be used to construct deterministic
   candidate strata such as:

   - positive hits;
   - no-hit cases;
   - one-token aliases;
   - multi-token aliases;
   - punctuation or hyphen cases;
   - apostrophe cases;
   - short aliases;
   - near-substring collision cases;
   - numeric or alphanumeric cases;
   - duplicate-normalized-alias cases;
   - zero-token-alias cases, if present.

3. Before any human or manual inspection of the sampled passage-aspect text or
   matcher correctness, freeze:

   - the exact 150-case validation manifest;
   - category or stratum assignment;
   - deterministic sampling rule and seed or hash namespace;
   - the exact 30-case double-review subset.

4. Human reviewers may inspect cases only after that manifest is frozen. Use
   two human reviewers for the shared 30-case double-review subset.

Thus, automated matcher output may define sampling strata. Human inspection or
correctness judgments may not influence which cases enter the validation
manifest.

Do not repeatedly add cases until validation looks favorable.

Report:

- validation sample construction;
- reviewer agreement;
- false-positive examples;
- apparent false negatives;
- edge-case failures;
- matcher version and hash.

Do not modify individual ASQA aliases based on review.

If the 150-case validation exposes a systematic matcher defect, stop before
`SELECTION`. A change requires a prospective matcher amendment and a new
matcher version. Do not patch individual aliases or create
development-specific exceptions.

## 25. No Performance-Informed Matcher Tuning

Matcher design and normalization must never depend on:

- retrieval Recall or S-recall;
- alpha-nDCG performance;
- which diversification method wins;
- generator correctness;
- protected results.

The matcher is a measurement instrument, not a tunable retrieval component.
Do not choose normalization rules because they improve a system score.

## 26. ASQA Evidence Roles

Matcher implementation and validation may use only:

```text
ASQA internal DEVELOPMENT = 3482
```

ASQA internal `SELECTION = 871` is reserved for bounded method selection under
the later statistics protocol.

ASQA dev948 remains:

```text
PROJECT_PROTECTED_FINAL
```

Do not inspect protected matcher or metric outcomes until all applicable
methodology, implementations, configurations, statistics, and evaluator
versions are frozen.

## 27. Answer-Side Alias Coverage

`docs/sprint3/ANSWER_CORRECTNESS_PROTOCOL.md` defines a separate diagnostic:

```text
answer_side_alias_coverage
```

The `answer_side_alias_coverage` diagnostic must reuse:

- the same official ASQA `qa_pair` to `short_answers` inventory;
- the same zero-token handling;
- the same duplicate-normalized-alias handling;
- the same Unicode NFC normalization;
- the same exact apostrophe map;
- the same punctuation, symbol, and separator-to-space policy;
- the same one-token case-sensitive rule;
- the same two-or-more-token Unicode-casefold rule;
- the same contiguous exact token-sequence matching rule.

The only scientific surface difference is:

```text
retrieval metric:
  match against canonical DPR passage BODY

answer_side_alias_coverage diagnostic:
  match against the generated canonical ASQA Answer text
```

Do not merge the two scores. Do not use answer-side alias coverage to define
corpus coverability, `SRecall@5`, alpha-nDCG, or `c*`.

The official ASQA QA-F1 or Disambig-F1 correctness scorer remains separate and
is governed by `docs/sprint3/ANSWER_CORRECTNESS_PROTOCOL.md`.

## 28. What Not to Use

Do not replace canonical ASQA retrieval effectiveness with:

- embedding diversity;
- pairwise embedding distance;
- lexical document diversity;
- generic relevance judge;
- LLM passage relevance;
- generated-answer correctness;
- answer-side alias coverage;
- ACC;
- faithfulness.

Embedding or geometric diversity may exist as a manipulation or mechanistic
diagnostic elsewhere, but it is not the ASQA lead retrieval-effectiveness
endpoint.

## 29. Corpus Fallback Relation

This protocol defines metrics relative to the authorized frozen corpus. The
canonical preference remains full DPR-2018 Wikipedia.

Do not create or authorize a reduced ASQA corpus in this document. ASQA
fallback governance is a separate methodology item.

If a future prospective amendment authorizes a different fixed shared corpus,
the same matcher and metric definitions apply to that corpus, but:

- coverability;
- `J` hits;
- alpha-IDCG;
- `c*`;

must be recomputed and versioned for that corpus.

Never compare those quantities as if corpus identity had not changed.

## 30. Failure and Zero Policy

Metric-computation failure is not a zero score.

Examples include:

- missing corpus identity;
- matcher-version mismatch;
- malformed ASQA aspect schema;
- broken passage identity;
- metric-implementation exception.

These must produce explicit failure or `NA`.

Specific scientifically valid zero cases remain valid:

```text
SRecall@5 = 0
```

only when `B_q > 0` and none of the coverable aspects occurs in the selected
top five.

```text
alpha-nDCG@5 = 0
```

only when `alpha-IDCG > 0` and the system ranking earns zero alpha-DCG gain.

Do not convert `B_q = 0` into either zero metric.

## 31. Provenance

Every canonical ASQA retrieval-metric artifact must bind or reference:

- ASQA source logical ID;
- immutable source revision once acquired;
- `sample_id`;
- evidence role;
- `qa_pair` or aspect identity;
- raw `short_answers`;
- raw alias-set hash;
- normalized alias token sequences;
- normalized alias-set hash;
- zero-token aliases and reason;
- corpus protocol and version;
- corpus manifest and hash;
- passage count;
- passage ID;
- exact passage-body hash;
- title hash separately if stored, while confirming title was not matched;
- matcher logical ID, version, and hash;
- normalization implementation and version;
- apostrophe map and version;
- case-policy version;
- sparse positive `J` hits and hash;
- `G_q`;
- `B_q`;
- `U_q`;
- `rho_q`;
- `SRecall@5`;
- alpha values;
- alpha-DCG;
- alpha-IDCG;
- alpha-nDCG;
- greedy ideal-list tie-break identity;
- `c*`;
- exact `c*` solver and version;
- metric-implementation version;
- Git commit;
- run-registry identity.

## 32. Reporting Limitation

State explicitly in eventual reporting:

> The experiment uses the reproducible DPR December-2018 Wikipedia
> collection.

This is compatible with the JPR retrieval lineage used in ASQA research but
does not exactly reproduce all Wikipedia evidence used during AmbigQA or ASQA
annotation.

Snapshot mismatch is therefore a documented corpus limitation.

Do not claim:

> the exact Wikipedia snapshot ASQA was constructed on.

## 33. Frozen Before Protected Observation

This document records the canonical ASQA retrieval-diversity measurement
methodology before ASQA `SELECTION` outcomes or protected dev948 outcomes are
opened.

No ASQA `SELECTION` or protected result was used to choose:

- alias inventory;
- normalization;
- case policy;
- passage surface;
- S-recall denominator;
- alpha values;
- ideal-list construction;
- `c*` definition;
- zero or `NA` handling.

## 34. Implementation and Execution Items Not Solved Here

The following are downstream work rather than open metric methodology:

- immutable ASQA source-snapshot acquisition;
- physical DPR corpus acquisition;
- verification of the expected 21,015,324 count;
- ASQA `CorpusRecord` builder;
- body-only `retrieval_content` enforcement;
- matcher implementation;
- sparse `J` sidecar implementation;
- `DEVELOPMENT` validation-manifest materialization;
- human matcher validation;
- exact `c*` solver implementation;
- alpha-nDCG regression tests;
- metric-registry synchronization;
- run-registry and provenance integration;
- protected metric execution.

These items do not reopen the metric construct unless validation identifies a
systematic defect requiring a prospective amendment.

## 35. Relation to Other Protocols

This protocol must be read with:

- `docs/sprint3/ASQA_INTERNAL_PARTITION_PROTOCOL.md`;
- `docs/sprint3/CANDIDATE_POOL_TOPK_PROTOCOL.md`;
- `docs/sprint3/DIVERSIFICATION_CONFIGURATION_PROTOCOL.md`;
- `docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL.md`;
- `docs/sprint3/ANSWER_CORRECTNESS_PROTOCOL.md`.

The partition protocol governs `DEVELOPMENT`, `SELECTION`, and protected
evidence roles. The candidate-pool protocol governs top-20 and top-5
mechanics. The diversification protocol governs methods and configurations.
The correctness protocol governs generated-answer evaluation. This protocol
governs passage-level ASQA aspect retrieval measurement.
