# Sprint 3 ASQA Corpus Authority Note

## 1. Purpose

This governance and supersession note exists only to make current ASQA corpus
authority unambiguous across older and newer Sprint-3 documents.

It closes stale-document ambiguity concerning:

- canonical corpus identity;
- expected corpus count;
- fallback permission;
- development-only reduced corpora;
- corpus-relative metric recomputation.

It does not create new ASQA corpus methodology and does not authorize a
fallback.

This note does not reopen:

- ASQA matcher rules;
- alias rules;
- retrieval metrics;
- dataset partitioning;
- evidence roles;
- `candidate_pool` or `top_k`;
- the final experiment matrix;
- resource thresholds.

## 2. Current Authoritative Canonical ASQA Corpus

The currently authorized canonical ASQA corpus is:

```text
standard DPR English Wikipedia passage collection
snapshot lineage: 2018-12-20
expected passage count: 21,015,324
canonical passage surface: exact DPR passage BODY only
```

Physical acquisition must later verify and bind to provenance:

- immutable source identity and revision;
- actual passage count;
- corpus manifest;
- passage IDs;
- passage boundaries;
- body hashes.

The expected count is an assertion pending physical verification. A mismatch
does not authorize silent substitution of another corpus.

## 3. Same Logical Corpus for All Retrievers

Canonical:

- BM25;
- DPR;
- Contriever;
- ColBERTv2;

must use the same logical corpus universe, including:

- the same passage membership;
- the same passage IDs;
- the same boundaries;
- the same canonical body text.

Physical indexing structures may differ. Corpus identity may not vary by
retriever.

## 4. No Current ASQA Fallback Authorization

No reduced canonical ASQA fallback corpus is currently authorized.

If full DPR-2018 canonical execution proves infeasible:

```text
STOP
```

Do not:

- choose a smaller corpus;
- choose a resource-derived `M`;
- create a random shared subset;
- copy the HotpotQA fallback;
- proceed with protected evaluation on a reduced corpus.

A future corpus change requires a new prospective methodology amendment
created before the affected canonical results are observed.

This note is not that amendment.

## 5. Supersession of the Old Blueprint Fallback Clause

`docs/sprint3/PROJECT_BLUEPRINT.md` contains older ASQA language permitting a
large, frozen, shared Wikipedia subsample after a failed resource pilot.

That ASQA fallback permission is:

```text
SUPERSEDED
```

The later authority is:

- `docs/sprint3/ASQA_RETRIEVAL_METRIC_PROTOCOL.md`;
- `docs/sprint3/FINAL_EXPERIMENT_MATRIX_PROTOCOL.md`.

The final matrix's controlling canonical ASQA rule is:

```text
NO FALLBACK AUTHORIZED
+ STOP
+ PROSPECTIVE AMENDMENT REQUIRED
```

Do not interpret the old blueprint clause as current permission for canonical
ASQA fallback execution.

## 6. Expected-Count Supersession

Older occurrences of:

```text
21,015,300
```

in:

- `docs/sprint3/DATASET_PROTOCOL.md`;
- `docs/sprint3/DATASET_DECISIONS.md`;

are superseded for canonical Sprint-3 ASQA authority by:

```text
21,015,324
```

as frozen in the later ASQA corpus and metric authority.

The exact physical count still requires acquisition verification. The older
files remain unchanged as historical and supporting protocol records.

## 7. Development-Only Reduced Corpora Remain Allowed

Older language allowing small or reduced corpora remains valid only for
noncanonical `DEVELOPMENT` purposes such as:

- unit and integration testing;
- smoke runs;
- debugging;
- profiling;
- feasibility estimation;
- resource estimation.

Such reduced corpora:

- are not canonical ASQA evaluation corpora;
- are not fallback corpora;
- cannot produce protected-final ASQA claims;
- cannot be presented as full-corpus ASQA results;
- cannot determine canonical corpus identity based on performance.

Preserve `DEVELOPMENT`, `SELECTION`, and protected evidence-role boundaries.

## 8. Query-Specific and Gold-Injected Corpora Are Prohibited

Canonical ASQA retrieval may not use:

- query-specific corpora;
- per-question document pools;
- supplied gold pages;
- ASQA annotation passages injected into the corpus;
- AmbigQA or ASQA supplied evidence injected after retrieval;
- gold-document injection into top-20 or top-5;
- retriever-specific corpus membership.

This prohibition remains active regardless of resource pressure.

## 9. Corpus-Relative Metric Dependencies

Canonical ASQA retrieval measurement is corpus-relative.

Any prospectively authorized corpus-identity change requires:

- a new corpus version, manifest, and hash;
- new corpus-dependent ASQA metric-artifact versions;
- complete recomputation and provenance of at least:

- corpus manifest;
- passage identities;
- body hashes;
- passage count;
- sparse positive `J_q(d,i)` hits;
- `A_q_plus`;
- `B_q`;
- `U_q`;
- `rho_q`;
- `C_q@5`;
- `SRecall@5`;
- alpha-DCG inputs and results where corpus-dependent;
- alpha-IDCG;
- `alpha-nDCG@5` for `alpha=0.5`;
- alpha-nDCG sensitivity outputs for `alpha=0.3` and `alpha=0.7`;
- exact `c*`;
- every downstream artifact whose identity depends on those corpus-relative
  quantities.

`SRecall@5` and `alpha-nDCG@5` must be recomputed.

Preserving old denominators or alpha-IDCG values across corpus versions is not
valid.

The deterministic alias-matcher implementation and version may remain
unchanged only if:

- the alias inventory is unchanged;
- normalization is unchanged;
- case policy is unchanged;
- passage-matching logic is unchanged.

Even if the matcher algorithm version remains unchanged, every `J` artifact
must bind the new corpus identity and therefore receive a new corpus-specific
artifact and version identity.

If matcher logic itself changes, that separately requires the prospective
matcher-amendment and version rules already frozen in
`docs/sprint3/ASQA_RETRIEVAL_METRIC_PROTOCOL.md`.

## 10. Authority Order

The ASQA corpus-specific authority order is:

1. later explicit professor or supervisor instruction, if prospectively
   recorded;
2. `docs/sprint3/ASQA_CORPUS_AUTHORITY_NOTE.md` for cross-document ASQA corpus
   authority, supersession, expected-count authority, and current no-fallback
   interpretation;
3. `docs/sprint3/FINAL_EXPERIMENT_MATRIX_PROTOCOL.md` for canonical
   no-fallback and `STOP` execution behavior;
4. `docs/sprint3/ASQA_RETRIEVAL_METRIC_PROTOCOL.md` for canonical corpus, body
   surface, matcher, and corpus-relative metric definitions;
5. `docs/sprint3/PROJECT_BLUEPRINT.md` where not superseded;
6. older dataset protocols and decision notes where compatible;
7. research, audit, and implementation notes as non-authoritative unless
   separately frozen.

This note does not override the scientific definitions in the ASQA metric
protocol or the execution matrix. It records which later authorities
supersede older conflicting language.

When an older document conflicts with the later final-matrix or ASQA metric
protocol on canonical ASQA corpus or fallback behavior, the later specific
authority controls.

## 11. HotpotQA Is Scientifically Separate

HotpotQA has a separate frozen resource and fallback architecture.

Its resource gate and potential:

- shared deterministic fallback;
- resource-derived `M`;
- `M_min` requirement;

do not apply to ASQA.

Do not copy HotpotQA fallback behavior into ASQA by analogy. This note does not
resolve HotpotQA `M_min`.

## 12. What This Note Does Not Do

This note does not:

- authorize an ASQA fallback;
- define a fallback size;
- define a resource gate;
- define memory or runtime limits;
- alter DPR-2018 passage construction;
- alter matcher normalization;
- alter official aliases;
- alter SRecall;
- alter alpha-nDCG;
- alter `c*`;
- alter ASQA partitioning;
- alter final matrix counts;
- alter confirmatory contrasts.

It is an authority and supersession closure only.

## 13. Relation to Authoritative Files

- `docs/sprint3/ASQA_RETRIEVAL_METRIC_PROTOCOL.md` defines the canonical ASQA
  corpus, body surface, matcher, and corpus-relative metric semantics.
- `docs/sprint3/FINAL_EXPERIMENT_MATRIX_PROTOCOL.md` controls current canonical
  no-fallback and stop behavior.
- `docs/sprint3/PROJECT_BLUEPRINT.md` is older blueprint authority where not
  superseded; its ASQA fallback clause is superseded here.
- `docs/sprint3/DATASET_PROTOCOL.md` provides compatible provenance, integrity,
  gold-injection, and development-subset rules; its older expected count is
  superseded.
- `docs/sprint3/DATASET_DECISIONS.md` records earlier ASQA decisions and
  development-subset restrictions; its older expected count is superseded.
- `docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL.md` governs when corpus identity,
  physical acquisition, and execution gates must close.
- `docs/sprint3/HOTPOTQA_RESOURCE_GATE_PROTOCOL.md` governs HotpotQA only and
  does not authorize ASQA fallback behavior.

## 14. Pre-Observation Status

This governance note is created before canonical ASQA `SELECTION` outcomes or
protected-final outcomes are inspected.

No such result was used to choose:

- canonical full-corpus identity;
- the no-fallback rule;
- expected-count authority;
- the development-only reduced-corpus distinction;
- supersession ordering.

## 15. After Creation

After creating `docs/sprint3/ASQA_CORPUS_AUTHORITY_NOTE.md`:

1. show the complete new-file diff;
2. run `git status --short`;
3. confirm no other file changed;
4. do not commit;
5. stop.
