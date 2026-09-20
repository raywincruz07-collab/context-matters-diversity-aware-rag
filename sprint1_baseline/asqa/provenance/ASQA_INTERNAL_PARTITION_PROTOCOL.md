# ASQA Internal Train-Partition Protocol

## 1. Purpose

This protocol freezes the internal partition of the official ASQA train
universe before decomposer selection, evaluator calibration, method selection,
or protected-final evaluation.

- Official ASQA train: 4,353 examples.
- Official ASQA dev: 948 examples.

The official ASQA dev948 remains `PROJECT_PROTECTED_FINAL` and is not part of
this internal train partition.

## 2. Evidence Roles

This protocol freezes three distinct evidence roles.

### DEVELOPMENT

`DEVELOPMENT` contains 3,482 examples from official train4353. It may be used
for:

- implementation and debugging;
- smoke tests;
- resource pilots;
- the decomposer bake-off;
- atomic-claim instrument development;
- NLI threshold calibration;
- human evaluator validation;
- matcher validation;
- prompt development;
- method construction;
- development-only error analysis.

### SELECTION

`SELECTION` contains 871 examples from official train4353. It is used only
after development decisions are frozen, for a bounded comparison of
**predeclared** candidate configurations such as:

- retriever/diversifier configurations;
- MMR lambda;
- clustering configurations;
- DPP configurations;
- other explicitly predeclared method hyperparameters.

### PROJECT_PROTECTED_FINAL

`PROJECT_PROTECTED_FINAL` contains all 948 official ASQA dev examples. It is
used only for canonical final evaluation after all applicable methodology,
configurations, prompts, thresholds, and statistical procedures are frozen.

## 3. Split Size

The internal train split is frozen as:

```text
DEVELOPMENT = 3482
SELECTION   = 871
```

This is approximately an 80/20 partition of train4353. The rationale is:

- it preserves a large development pool for evaluator and method construction;
- it provides 871 paired selection questions;
- the planning-scale paired detectable effect is approximately 0.095 SD under
  simple alpha=.05 / 80%-power assumptions;
- the worst-case binary-proportion margin is approximately +/-3 percentage
  points with finite-population correction;
- moving to 75/25 or 70/30 provides relatively modest additional selection
  precision while materially reducing development capacity;
- having no internal selection split would conflate iterative development with
  configuration selection.

These are planning references only. They do not replace the later frozen
statistical-analysis protocol.

## 4. Source Identity

Partitioning uses ASQA's stable original:

```text
sample_id
```

Sequential row position must not be used. Upstream `sample_id` is signed int64.
Its canonical string representation is:

- exact signed base-10 decimal;
- an optional leading `-` only for negative values;
- digits otherwise;
- no leading `+`;
- no leading zero except for the value `0`;
- UTF-8 encoded.

`sample_id` must not be coerced to int32.

## 5. Partition Hash Rule

The partition priority is frozen exactly as:

```text
priority(asqa_sample_id) =
SHA256(
  UTF8(
    "context-matters-rag|sprint3|ASQA|"
    "TRAIN_INTERNAL_PARTITION|20260823|"
    + canonical_decimal_sample_id
  )
).hexdigest()
```

The procedure is:

1. At source acquisition, verify exactly 4,353 unique official train
   `sample_id` values.
2. Verify exactly 948 unique official dev `sample_id` values.
3. Verify that the train and dev `sample_id` sets are disjoint.
4. Preserve every signed-int64 source ID and its canonical decimal string.
5. Compute a lowercase SHA-256 priority for every train ID.
6. Sort ascending by `(lowercase_hex_digest, UTF-8 canonical-ID bytes)`.
7. Assign the first 3,482 IDs to `DEVELOPMENT`.
8. Assign the remaining 871 IDs to `SELECTION`.
9. Do not allow any content field to influence membership.

Membership selection must not use:

- question text;
- long answers;
- short answers;
- `qa_pairs`;
- annotations;
- contexts;
- aspect counts;
- retrieval data;
- model outputs;
- difficulty;
- metric values;
- any semantic or result-derived field.

## 6. Existing Exposure

The repository audit established that:

- no concrete ASQA train `sample_id` has been identified as previously used in
  a retrieval, model, evaluator, or matcher-development run;
- no concrete ASQA dev948 result example has been found in experimental
  artifacts;
- no previous 3,482/871 manifest exists;
- earlier aggregate dataset, schema, and context-coverage auditing is
  dataset-level inspection, not evidence of example-level method selection.

Therefore, at the time this protocol is written:

```text
forced-development exposure set = empty
```

Before actual partition materialization, check whether an off-repository or
manual exposure register identifies concrete ASQA train IDs used for method
construction.

If any such concrete IDs are discovered before materialization, **STOP**. Do
not silently alter the hash split. A prospective amendment must place those IDs
in `DEVELOPMENT` and define how membership and counts are preserved.

## 7. Development Exposure Labels

All future development activities must draw only from `DEVELOPMENT`.

For decomposer-selection ASQA examples, additionally label every used ID:

```text
EVALUATOR_DEVELOPMENT_EXPOSED
```

This is already required by the committed decomposer-selection protocol.
Likewise, record the specific exposure purpose for:

- NLI calibration;
- matcher validation;
- human validation;
- prompt development;
- resource or smoke testing.

Repeated exposure within `DEVELOPMENT` is permitted and must be recorded in
provenance.

## 8. Selection Discipline

The `SELECTION` manifest may be materialized, hashed, and committed
prospectively because ID membership alone is not a result. Selection content
and results must not be used for ordinary development.

Before opening selection outcomes:

- candidate configurations must be predeclared;
- prompts must be frozen for that selection round;
- evaluator instruments and thresholds relevant to the comparison must be
  frozen;
- comparison metrics and decision criteria must be frozen.

Evaluate every predeclared candidate on the same 871 selection IDs. Prefer one
bounded selection round.

Do not:

- introduce new candidates after seeing selection performance;
- retune thresholds on `SELECTION`;
- repeatedly redesign methods until selection improves;
- replace failed questions.

If a genuine methodological defect is discovered, document it explicitly and
stop and reassess rather than silently iterating.

Selection rows must never be pooled with `DEVELOPMENT` or
`PROJECT_PROTECTED_FINAL` statistics.

## 9. PROJECT_PROTECTED_FINAL Dev948

All 948 official ASQA dev examples remain:

```text
PROJECT_PROTECTED_FINAL
```

There is no internal repartition of dev948. Dev948 must not be used for:

- prompt development;
- decomposer selection;
- NLI calibration;
- matcher calibration;
- evaluator validation;
- retriever/diversifier configuration selection;
- hyperparameter tuning;
- threshold selection.

Canonical final ASQA analyses may use dev948 only after all applicable
methodology, configuration, and statistical gates are closed. This includes
final:

- S-recall@5;
- alpha-nDCG@5;
- `c*` analysis;
- answer correctness;
- faithfulness;
- Bidirectional Atomic-Content Coverage.

## 10. Provenance

At materialization, eventually persist:

- requested ASQA source, repository, and configuration;
- immutable dataset/source revision;
- physical source-file hashes;
- exact schema, including signed-int64 `sample_id`;
- train count and unique-ID verification;
- dev count and unique-ID verification;
- train/dev disjointness check;
- train-universe hash;
- protected-dev-universe hash;
- namespace
  `context-matters-rag|sprint3|ASQA|TRAIN_INTERNAL_PARTITION|20260823|`;
- SHA-256 algorithm;
- UTF-8 encoding;
- canonical decimal ID serialization;
- digest/ID-byte tie rule;
- ordered `DEVELOPMENT` IDs, count, and hash;
- ordered `SELECTION` IDs, count, and hash;
- forced-development exposure list and hash;
- role and exposure labels;
- creation timestamp;
- builder version;
- Git commit;
- record and question hashes used only **after** membership selection to bind
  identity.

## 11. Stale Repository Notes

Some repository-audit unresolved notes are stale because later methodology work
has already frozen:

- the ASQA literal-alias matcher architecture;
- S-recall@5 as the lead metric;
- alpha-nDCG@5 with alpha=0.5 as primary and alpha=0.3/0.7 as sensitivity;
- the `c*` architecture;
- the faithfulness construct;
- the generation-protocol architecture.

Those decisions are not reopened in this document. The exact decomposer winner
remains prospectively selected under the committed decomposer-selection
protocol.

## 12. Still Unresolved

The genuinely unresolved downstream items are:

- immutable `din0s/asqa` source revision and physical acquisition identity;
- actual partition-manifest materialization;
- any off-repository concrete exposure discovered before materialization;
- final candidate-pool and top-k;
- exact diversification configuration matrix;
- final statistical-analysis protocol;
- exact selection objective and acceptable-loss rule for method selection.

## 13. FROZEN BEFORE OBSERVATION

This document freezes the 3,482/871 split sizes, ID-only hash construction,
evidence-role separation, development-use policy, bounded selection discipline,
and dev948 protected-final policy before any ASQA selection or protected-final
outcomes are observed.

The actual ASQA train partition IDs and dev948 final results are not materialized
or inspected while writing this document.
