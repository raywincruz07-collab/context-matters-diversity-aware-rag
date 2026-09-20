# Sprint 3 HotpotQA Full-Corpus Authority Amendment

## 1. Authority and Motivation

This prospective amendment must be read with:

- `docs/sprint3/HOTPOTQA_FINAL_EVALUATION_PROTOCOL.md`;
- `docs/sprint3/HOTPOTQA_RESOURCE_GATE_PROTOCOL.md`;
- `docs/sprint3/FINAL_EXPERIMENT_MATRIX_PROTOCOL.md`;
- `docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL.md`;
- `docs/sprint3/PROJECT_BLUEPRINT.md`.

Prior methodology:

- strongly preferred the complete BEIR HotpotQA corpus;
- provisionally allowed a deterministic global fallback only if
  `M >= M_min`;
- intentionally left `M_min` unresolved pending substantive justification.

A subsequent pre-observation literature and benchmark-methodology review found
no established HotpotQA or BEIR:

- minimum corpus size;
- minimum corpus fraction;
- accepted reduced-corpus fallback protocol;
- scientifically authoritative value for `M_min`.

Therefore assigning a value such as:

- `25,997`;
- `100,000`;
- `500,000`;
- `1,000,000`;
- 5%;
- 10%;
- 20%;
- 25%;
- 50%;

would create a project convention rather than apply an established benchmark
validity criterion.

The project prioritizes benchmark validity over preserving a resource-driven
fallback.

## 2. Canonical HotpotQA Corpus

The only canonical Sprint-3 HotpotQA retrieval corpus is:

```text
complete BEIR HotpotQA corpus
```

Expected document count:

```text
5,233,329
```

This count remains subject to physical source, revision, and count verification
before execution.

The canonical corpus must remain:

- global;
- query-independent;
- shared identically by BM25, DPR, Contriever, and ColBERTv2;
- free of query-specific gold or qrel injection.

Physical indexes may differ. Logical corpus membership may not differ by
retriever.

## 3. Retirement of M_min

Freeze:

```text
M_min is no longer an active Sprint-3 methodological parameter.
```

Do not define:

```text
M_min = 100%
```

or:

```text
M_min = 5,233,329
```

Instead record:

```text
M_min retired because no scientifically defensible reduced-corpus
admissibility threshold was established.
```

There is no canonical fallback-size decision to make. Do not estimate or tune
`M_min` later.

## 4. Revocation of Canonical Fallback

This amendment prospectively supersedes every earlier HotpotQA clause that
authorizes a reduced canonical fallback corpus based on:

- the largest resource-supported `M`;
- SHA-256 fallback membership;
- `M >= M_min`;
- resource-derived fallback sizing.

For canonical HotpotQA `SELECTION` and `PROJECT_PROTECTED_FINAL` evaluation:

```text
NO REDUCED FALLBACK CORPUS IS AUTHORIZED
```

If the complete corpus fails the frozen feasibility and resource gate:

```text
STOP
status = HOTPOTQA_CANONICAL_FULL_CORPUS_INFEASIBLE
```

Do not:

- select a smaller `M`;
- use the largest corpus that happens to fit;
- use the historical 25,997-document corpus;
- use a 100,000-, 500,000-, or 1,000,000-document pilot corpus as canonical
  evaluation;
- select a fixed fraction;
- use qrels to construct a reduced corpus;
- inject gold documents;
- continue protected evaluation on a modified corpus.

A future reduced-corpus scientific study requires a new prospective methodology
amendment. It must be reported as a `MODIFIED` benchmark setting and must not be
silently presented as canonical BEIR HotpotQA.

## 5. Resource Gate Remains Active for the Full Corpus

This amendment does not remove the resource gate. Preserve the frozen
feasibility architecture for deciding whether the complete corpus can be
executed safely.

Preserve:

```text
projection multiplier = 1.25
resource reserve       = 20%
```

For projected resource requirement `R` and declared resource limit `L`, retain
the frozen pass rule for every applicable frozen resource dimension:

```text
1.25 * R <= 0.80 * L
```

Preserve retriever-specific:

- disk projections;
- RAM projections;
- VRAM projections;
- build and index projections;
- search and runtime projections;
- recovery and reload considerations;
- budget and time constraints.

A fixable engineering defect does not establish scientific infeasibility. It
must first be diagnosed and fixed using permitted development evidence and then
remeasured according to the existing resource-gate protocol.

## 6. Resource Pilot Role

Preserve the existing frozen resource pilot.

Nested deterministic corpus sizes:

```text
100,000
500,000
1,000,000
```

HotpotQA `DEVELOPMENT` queries:

```text
220 total
20 warm-up
200 measured
```

Retrievers:

- BM25;
- DPR;
- Contriever;
- ColBERTv2.

Frozen retrieval cardinalities:

```text
candidate_pool = 20
top_k          = 5
```

The pilot remains resource and engineering evidence only. Its purpose after
this amendment is to:

- measure scaling;
- estimate complete-corpus feasibility;
- validate index, build, and search engineering;
- estimate disk, RAM, VRAM, runtime, and budget requirements.

The pilot sizes are not:

- candidate canonical corpora;
- candidate `M_min` values;
- fallback options;
- scientific corpus-size treatments.

Do not evaluate retrieval effectiveness to select corpus size.

## 7. Development Reduced Corpora Remain Permitted

Reduced corpora remain permitted for:

- unit testing;
- smoke testing;
- debugging;
- profiling;
- resource measurement;
- deterministic feasibility pilots.

They may not produce:

- canonical HotpotQA `SELECTION` evidence;
- protected-final evidence;
- standard BEIR HotpotQA claims.

Always label them `DEVELOPMENT` or `RESOURCE` evidence.

## 8. Historical 25,997-Document Corpus

The historical approximately 25,997-document Sprint-2 corpus is not an
admissible canonical fallback.

It was constructed historically for the earlier sampled-query setup and used
query- and gold-informed corpus construction.

It remains:

```text
HISTORICAL_OBSERVED
```

It may be discussed as historical Sprint-2 evidence only. It must not:

- determine canonical corpus membership;
- determine corpus size;
- justify `M_min`;
- enter canonical Sprint-3 `SELECTION`;
- enter protected-final evaluation.

## 9. Why No Numeric or Fractional Floor Is Frozen

Authoritative HotpotQA and BEIR sources define their normal corpus-level
evaluation using the fixed Wikipedia/BEIR corpus.

The pre-observation literature review did not establish a benchmark-authorized
minimum reduced-corpus size or fraction.

Corpus reduction changes the retrieval universe and distractor competition.
An arbitrary reduced size would therefore change the experimental setting
without an accepted benchmark-validity threshold.

This amendment does not claim:

- that every possible reduced corpus is scientifically useless;
- that literature proves all retriever rankings would change;
- that a universal theorem requires exactly 5,233,329 documents.

The narrower defensible claim is:

> No adequately justified reduced-corpus equivalence rule was found for this
> project's canonical HotpotQA/BEIR comparison.

Therefore the project chooses the complete corpus for canonical validity.

## 10. Literature-Evidence Caution

Supporting literature and research are used conservatively.

Do not rely on an engineering GitHub downsampling example as benchmark
authority.

Do not use LongRAG as evidence that random document subsampling is valid or
invalid. LongRAG changes retrieval-unit granularity by grouping content from
the Wikipedia corpus; it is not equivalent to this project's proposed random
document removal.

The core decision instead rests on:

- canonical HotpotQA and BEIR corpus definitions;
- the absence of an authoritative reduced-corpus equivalence or minimum rule;
- the desire to avoid an arbitrary project-defined corpus-validity threshold.

## 11. Selection and Protected Execution

Canonical HotpotQA `SELECTION` may open only after:

- complete corpus identity is frozen;
- physical corpus acquisition is verified;
- all four retriever implementations are ready;
- resource feasibility for the complete corpus passes;
- every other applicable Stage-2 requirement passes.

If complete-corpus feasibility does not pass, canonical HotpotQA `SELECTION`
does not open.

Protected-final HotpotQA cannot proceed without a valid earlier HotpotQA
selection stage where required by the frozen methodology. Reduced-corpus
results must not replace the blocked canonical stage.

## 12. Supersession

This amendment supersedes only HotpotQA clauses that authorize or depend on a
canonical reduced fallback corpus.

In particular, it supersedes these concepts for canonical Sprint-3 HotpotQA:

- canonical fallback corpus;
- fallback membership;
- resource-derived fallback `M`;
- `M_min`;
- `M >= M_min`;
- fallback activation after full-corpus failure.

Preserve every compatible rule concerning:

- full-corpus identity;
- the protected sample;
- evidence roles;
- corpus sharing across retrievers;
- prohibition of gold injection;
- `candidate_pool=20`;
- `top_k=5`;
- the retriever set;
- diversification;
- generation;
- statistics;
- selection;
- resource-feasibility measurement.

The superseded documents remain unchanged as protocol and decision-history
records.

## 13. Timing Ambiguity Is Closed

Earlier resource-gate wording created an ordering issue concerning whether
`M_min` had to be resolved before the resource pilot.

After this amendment, there is no `M_min`.

The frozen `DEVELOPMENT` resource pilot may proceed once its ordinary
development and resource prerequisites are satisfied. Its sole canonical
decision is:

> Can the complete BEIR HotpotQA corpus be executed under the frozen resource
> constraints?

There is no second fallback-size decision.

## 14. Provenance

Before full-corpus canonical execution, bind:

- this amendment's commit and hash;
- HotpotQA full-corpus source and revision;
- verified physical corpus count;
- corpus manifest and hash;
- passage IDs and content contract;
- resource-pilot manifest and hash;
- hardware identity;
- disk, RAM, and VRAM limits;
- time and budget limits;
- measured pilot outputs;
- retriever-specific full-corpus projections;
- full-corpus feasibility decision;
- Git commit;
- run-registry identity.

If feasibility fails, preserve the failure evidence. Do not erase an
infeasibility result.

## 15. Pre-Observation Status

This amendment is frozen before canonical HotpotQA `SELECTION` outcomes or
protected-final outcomes are inspected.

No canonical HotpotQA `SELECTION` or protected-final result was used to choose:

- the full-corpus-only policy;
- retirement of `M_min`;
- fallback revocation;
- the resource-pilot interpretation.

The decision was made prospectively from benchmark-methodology considerations
rather than observed comparative performance.

## 16. What This Amendment Does Not Do

This amendment does not:

- alter protected HotpotQA `N=500`;
- alter BEIR dev `SELECTION N=5,447`;
- alter historical or exposed classifications;
- alter retrievers;
- alter diversification methods;
- alter `candidate_pool` or `top_k`;
- alter generation;
- alter correctness;
- alter faithfulness;
- alter ACC;
- alter statistical procedures;
- alter confirmatory contrasts;
- change the resource-gate equation;
- guarantee that complete-corpus execution is feasible.

## 17. After Creation

After creating
`docs/sprint3/HOTPOTQA_FULL_CORPUS_AUTHORITY_AMENDMENT.md`:

1. show the complete new-file diff;
2. run `git status --short`;
3. confirm no other file changed;
4. do not commit;
5. stop.
