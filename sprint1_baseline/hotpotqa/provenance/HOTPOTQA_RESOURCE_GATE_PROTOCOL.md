# HotpotQA Resource-Feasibility Gate Protocol

## 1. Purpose

This document freezes the prospective resource-feasibility decision architecture
for choosing between:

- the full canonical BEIR HotpotQA corpus; or
- the already-authorized deterministic global fallback corpus.

The decision is purely resource- and operability-based. It must not depend on:

- Recall;
- MRR;
- nDCG;
- answer correctness;
- faithfulness;
- gold data or qrels;
- protected-final outcomes;
- comparative retriever performance.

This document does not claim that the full corpus passes or fails.

## 2. Canonical Scientific Policy

The preferred corpus is the full BEIR HotpotQA corpus.

Its expected scale is approximately 5,233,329 passages, subject to exact source
revision and count verification.

All four retrievers must operate over the same logical corpus:

- BM25;
- DPR;
- Contriever;
- ColBERTv2.

The following values are already frozen:

```text
candidate_pool = 20
final_top_k    = 5
```

The historical Sprint-2 25,997-document pool is `HISTORICAL_OBSERVED` only and
is prohibited as the canonical Sprint-3 corpus.

## 3. Global Corpus Rule

Every retriever must independently satisfy the resource gate.

If one retriever genuinely cannot use the full corpus, that retriever must not
receive a smaller private corpus. The only permitted outcomes are:

1. obtain sufficient resources;
2. apply a scientifically neutral engineering fix;
3. move all four retrievers to one globally frozen fallback corpus;
4. declare the canonical HotpotQA experiment infeasible if fallback `M` would
   violate `M_min`.

HotpotQA `SELECTION` and `PROJECT_PROTECTED_FINAL` must use the same logical
corpus policy.

## 4. Resource Dimensions

The gate considers only the smallest necessary set of resource dimensions:

- persistent disk;
- temporary disk;
- host RAM;
- GPU VRAM where applicable;
- index/build wall time;
- canonical candidate-production wall time;
- cloud or institutional compute budget;
- project-calendar feasibility;
- restart/resume feasibility.

Performance metrics must not be added to these dimensions.

## 5. Already-Frozen Safety Rule

Preserve the previously frozen resource-safety architecture:

- 20% capacity reserve;
- projection multiplier `1.25`.

For projected resource requirement `R` and declared usable limit `L`, the
conservative pass condition for projected capacity dimensions is:

```text
1.25 * R <= 0.80 * L
```

This intentionally provides a conservative resource buffer. These values must
not be changed by this document.

## 6. Resource Pilot

Freeze a `DEVELOPMENT`-only resource pilot using the following nested,
deterministic corpus sizes:

- 100,000 documents;
- 500,000 documents;
- 1,000,000 documents.

These are resource-characterization points, not scientific treatments.

Use query-independent deterministic corpus prefixes based on the already-frozen
fallback document ordering. Before indexing, restore selected documents to
canonical corpus order where required by the retriever contract.

Use all four retrievers with their frozen scientific configurations. Do not
change scientific retriever settings merely to make a measured pilot look
faster.

## 7. Resource Query Workload

Freeze 220 deterministic HotpotQA `DEVELOPMENT` queries selected using an
ID-only hash rule:

- the first 20 are warm-up queries and are excluded from reported timing;
- the next 200 form the measured resource workload.

The candidate pool remains exactly 20.

Do not load or use the following for resource-gate decision-making:

- qrels;
- answers;
- correctness;
- protected-final IDs or results.

The exact hash namespace and selection implementation remain a Stage-2
implementation and provenance item that must be committed before the pilot
runs.

## 8. Pilot Measurements

Record the following where applicable:

- exact document count;
- text and token count;
- source/corpus loading time;
- encoding time;
- index-construction time;
- final index size;
- peak temporary disk;
- peak host RSS;
- peak GPU VRAM;
- fingerprint and completion-validation time;
- reload time;
- p50 query latency;
- p95 query latency;
- queries per second;
- candidate-artifact writing rate;
- bytes per query;
- OOM and failure behavior;
- swap and thrashing behavior;
- interruption and recovery behavior.

No retrieval-effectiveness metric may enter the resource decision.

## 9. Retriever-Specific Projection

### BM25

- Use indexed document and token workload.
- Resource projection uses the more conservative observed per-document or
  per-token rate.
- The current full-corpus query implementation must be projected at least as
  `N log N` because it performs full scoring followed by full `lexsort`.

### DPR

- Embedding and storage scaling is linear in document count under the frozen
  768-dimensional float32 representation.
- Use `N * 768 * 4` bytes as a hard vector-payload lower bound.
- Exact `IndexFlatIP` search is projected linearly in `N` for fixed dimension
  and query count.

### Contriever

- Use the same general linear vector and storage logic as DPR.
- Use Contriever's own observed throughput and memory measurements.

### ColBERTv2

- Project using total tokenizer-token workload.
- Do not assume PLAID build or index scaling is linear.
- Use the more conservative of:
  1. linear-per-token extrapolation;
  2. a log-log fit across the three pilot points.

### Candidate production

Project from the measured pool-20 workload using the exact frozen Stage-3 and
Stage-5 query registry. Do not multiply retrieval work by diversifier or LLM
counts.

## 10. Projection Rule

Use all three pilot sizes.

For each resource dimension:

1. calculate the retriever-appropriate projection to the full corpus;
2. select the conservative applicable estimate;
3. apply the frozen `1.25` multiplier;
4. compare against the declared 80%-usable resource capacity.

Do not fit models to retrieval-quality outcomes.

## 11. Full-Corpus Pass Policy

A retriever passes full-corpus feasibility only when all applicable resource
constraints pass.

Where projections indicate feasibility, final closure additionally requires an
actual canonical full-corpus build and validation before `SELECTION` opens.

Require:

- exact corpus and manifest validation;
- full index completion;
- completion and fingerprint validation;
- successful reload;
- successful `DEVELOPMENT` pool-20 search;
- resource limits respected;
- wall-time and calendar limits respected;
- budget and quota respected;
- recovery strategy demonstrated.

A direct full build need not be attempted only when the already-frozen
projection prospectively proves a hard resource or budget violation.

## 12. Fixable Engineering Defect

A fixable engineering defect is something correctable without altering:

- corpus membership;
- document content or boundaries;
- retriever scoring semantics;
- scientific configuration;
- returned ranking semantics.

Examples include:

- avoidable corpus duplication;
- an unnecessarily materialized Python lookup;
- non-streaming manifest construction;
- a wrong cache or persistent path;
- a missing runtime dependency;
- missing checkpoint or resume plumbing;
- an avoidable duplicate temporary artifact;
- an exact, scientifically equivalent top-20 extraction optimization;
- batch or resource engineering proven to preserve encoded and scored values.

A fixable defect does not authorize fallback.

The required process is:

1. fix on `DEVELOPMENT`;
2. demonstrate equivalence where scientific output could be affected;
3. create a clean commit;
4. repeat affected pilot measurements and projection.

## 13. Genuine Resource Infeasibility

Resource infeasibility exists only when at least one retriever violates a
predeclared hard resource constraint after:

1. verifying measurements;
2. excluding transient infrastructure failure;
3. considering reasonable scientifically neutral engineering fixes;
4. considering resources available within the prospectively declared budget;
5. applying the frozen safety and projection rule.

Valid causes include:

- the minimum scientifically equivalent artifact does not fit disk;
- minimum required RAM or VRAM exceeds available hardware;
- build or retrieval runtime exceeds the declared compute or calendar window;
- required spend or quota exceeds the declared budget;
- restart or resume cannot complete inside the declared limits.

The following are not valid causes:

- poor retrieval quality;
- inconvenient engineering;
- one retriever being slower than another;
- historical Sprint-2 performance;
- a desire to reduce cost after seeing selection results.

## 14. Fallback M_min

No numerical `M_min` is frozen by this document.

Repository evidence does not scientifically justify a numerical `M_min`.
Historical 25,997 is not a valid basis, and an arbitrary fraction of 5.23
million must not be invented.

`M_min` remains:

```text
PENDING PROSPECTIVE SUPERVISOR / SUBSTANTIVE ADJUDICATION
```

If no defensible `M_min` is approved before the resource gate is executed,
full-corpus failure means that the canonical HotpotQA study is infeasible. It
does not authorize an arbitrarily small fallback.

## 15. Fallback M Rule

If fallback is prospectively authorized and the full corpus genuinely fails:

```text
M = largest common resource-supported corpus size satisfying every frozen
    hard constraint for all four retrievers
```

Then require:

```text
M >= M_min
```

Use exactly the first `M` documents under the already-frozen global fallback
SHA-256 order, then restore canonical corpus ordering where required.

Freeze:

- one `M`;
- one corpus manifest;
- one logical fallback corpus.

All four retrievers must use exactly that corpus. `M` must never be selected or
resized using retrieval performance.

Do not deliberately retain:

- historical gold documents;
- qrel documents;
- Sprint-2 pool documents.

## 16. Fallback Decision Record

Before invoking fallback, preserve:

- decision schema and version;
- timestamp;
- Git commit and worktree state;
- corpus source, revision, and count;
- corpus-manifest hash;
- resource-pilot corpus and query manifest hashes;
- hardware;
- RAM and VRAM;
- storage limits;
- environment and package locks;
- retriever and index configurations;
- `candidate_pool=20`;
- pilot measurements;
- raw resource logs;
- index inventories;
- projection calculations;
- frozen safety factors;
- declared budgets and limits;
- engineering alternatives considered;
- reason codes;
- per-retriever pass or fail status;
- global decision;
- `M_min`;
- chosen `M`, if applicable;
- fallback-manifest hash.

Include the explicit statement:

> No retrieval-quality, answer, faithfulness, gold/qrel, or protected-result
> metric entered the resource decision.

## 17. Failure and Resume

For a cloud or infrastructure interruption:

- resume only validated artifacts under identical scientific identity;
- otherwise restart the affected index.

Completed valid caches or indexes may be reused only if their identity and hash
pass validation. Partial artifacts must not be silently treated as complete.

If the pilot predicts feasibility but a full build fails:

1. stop;
2. classify the failure;
3. do not invoke fallback automatically;
4. determine whether the cause is infrastructure, a fixable defect, or a genuine
   resource breach;
5. follow the corresponding governed process.

One-retriever failure never authorizes a retriever-specific corpus. Never mix
pre-fix and corrected index artifacts.

## 18. Stage-Gate Alignment

The full-versus-fallback corpus decision is a Stage-2 hard blocker.

The required order is:

1. freeze the exact Hotpot source and corpus implementation contract;
2. declare numerical resource budgets, limits, and `M_min`;
3. materialize resource-pilot manifests;
4. run the `DEVELOPMENT`-only resource pilot;
5. apply frozen projections;
6. perform full-build validation if feasible, or formally invoke fallback;
7. freeze the final canonical corpus manifest and physical index identities;
8. only then open HotpotQA `SELECTION`.

HotpotQA `SELECTION` and `PROJECT_PROTECTED_FINAL` must use the same logical
corpus. Historical Sprint-2 evidence remains separate.

## 19. Current Numeric Inputs Still Required

The following remain unresolved and must be fixed before the resource pilot
begins:

- usable persistent-disk capacity;
- usable temporary disk;
- host RAM;
- GPU VRAM;
- maximum cloud or institutional compute budget;
- maximum GPU-hours or quota;
- maximum acceptable index-build wall time;
- maximum candidate-production wall time;
- remaining project-calendar window;
- restart allowance;
- intended persistent-index lifecycle;
- numerical `M_min`;
- exact RunPod hardware, storage, and pricing.

## 20. Implementation and Provenance Items Still Required

The following remain required but are not solved by this document:

- exact BEIR Hotpot source revision and count;
- title, body, and `retrieval_content` contract;
- full-corpus manifest;
- streaming loader;
- Hotpot candidate producers;
- resource-pilot hash namespace and manifests;
- measured BM25 scaling;
- measured PLAID scaling;
- full-corpus resource outcome;
- fallback `M`, if needed;
- final Stage-3 and Stage-5 query-row registry;
- long-build recovery behavior.

## 21. Frozen Before Resource Observation

This protocol freezes the scientific resource-decision architecture before
resource-pilot outcomes or selection and protected-final outcomes are observed.

It does not assert that the full corpus is feasible or infeasible.

Numeric hardware and budget limits and `M_min` remain explicit prospective
blockers.

No retrieval-quality metric may ever be used to invoke or size the fallback.
