# HotpotQA Protected-Final Evaluation Protocol

## 1. Purpose

This document freezes the HotpotQA protected-final query policy and canonical
corpus policy before protected-final retrieval.

Historical Sprint 2 remains `HISTORICAL_OBSERVED`. It is not re-labelled by
this protocol.

## 2. Populations

Define:

```text
T = exact BEIR HotpotQA test universe
H = historical exposed 500-query manifest
E = T \ H
```

Before materialization, all of the following identity checks are required:

```text
|T| = 7405
|H| = 500
H ⊂ T
|E| = 6905
E ∩ H = ∅
E ∪ H = T
```

The historical manifest SHA-256 is:

```text
9db98acceced785e77695ee55f361adb4c0e8f15f248e831901d3092c739e3b6
```

The 6,905 IDs in `E` are the eligible `PROJECT_PROTECTED_FINAL`
population. Selection may use only exact query IDs and test membership.

Selection must not use:

- question text;
- answers;
- qrel scores;
- gold document IDs;
- retrieval outputs;
- model outputs;
- difficulty estimates;
- any result-derived property.

## 3. Protected-Final Sample Size

The protected-final HotpotQA query sample size is frozen as:

```text
N = 500
```

The reasons for selecting 500 queries are:

- it provides a substantial paired question-level sample;
- it gives an approximately ±4.2 percentage-point worst-case proportion
  margin over the 6,905-query population as a planning reference;
- it gives an approximate paired standardized detectable-effect scale around
  0.125 SD under simple 80%-power and alpha = 0.05 planning assumptions;
- it balances statistical usefulness against the very large retriever ×
  diversification × LLM generation and evaluation workload;
- it is appropriate for a three-dataset, 12-ECTS project.

Any earlier estimate such as "31,500 generations" is only a rough planning
illustration based on an assumed condition count. It is not the frozen
full-study generation count because the exact diversification configuration
matrix is not yet frozen.

## 4. Query Sample Construction

The domain-separated query priority is frozen exactly as:

```text
priority(beir_query_id) =
SHA256(
  UTF8(
    "context-matters-rag|sprint3|HotpotQA|"
    "PROJECT_PROTECTED_FINAL|20260822|"
    + beir_query_id
  )
).hexdigest()
```

The construction procedure is:

1. Obtain `T` from distinct exact query-ID strings in the pinned BEIR
   test-membership source.
2. Load `H` from the historical manifest identified above.
3. Verify every population identity gate in Section 2.
4. Construct `E = T \ H`.
5. Preserve query IDs byte-for-byte.
6. Compute the priority over every ID in `E`.
7. Sort ascending by `(lowercase_hex_digest, UTF-8 query-ID bytes)`.
8. Select the first 500 IDs.
9. Assign final manifest positions `0..499` while preserving the exact BEIR ID
   as `source_sample_id`.
10. Bind query text only after selection and record its SHA-256.
11. Persist:
    - the complete 7,405-ID universe identity and hash;
    - the complete 6,905-ID eligible-order identity and hash;
    - the selected 500-ID manifest and hash.

The same 500 selected queries must be used across every retriever,
diversification method, LLM, and context-comparison condition where HotpotQA
protected-final data are applicable. Failures do not trigger replacement.

This document freezes the construction rule now. The actual selected
protected-final IDs must not be materialized or inspected while this document
is written.

## 5. Primary Corpus

The primary HotpotQA retrieval corpus is frozen as the complete BEIR HotpotQA
corpus, with an expected size of 5,233,329 passages, subject to exact
verification at corpus-manifest acquisition time.

All four retrievers must receive the exact same logical corpus:

- BM25;
- DPR;
- Contriever;
- ColBERTv2.

The shared logical corpus requires:

- the same document membership;
- the same source IDs;
- the same passage boundaries;
- the same title/body definition;
- the same `retrieval_content`.

Physical indexes may differ by retriever. The historical 25,997-document Sprint
2 pooled corpus is prohibited for protected-final canonical retrieval.

## 6. Full-Corpus Manifest Requirements

Before any canonical indexing, the following must be frozen and verified:

- immutable source corpus revision;
- exact document count;
- unique exact source document IDs;
- canonical document ordering;
- repository document-ID mapping;
- title;
- body;
- exact shared `retrieval_content`;
- per-record hashes;
- aggregate corpus-manifest hash.

The Hotpot-specific title/body/`retrieval_content` construction remains an
`UNRESOLVED IMPLEMENTATION-CONTRACT DETAIL`. It must be frozen before corpus
acquisition or indexing. This document does not invent that mapping.

## 7. Full-Corpus-First Policy

The full corpus is methodologically preferred. A reduced corpus must not be
selected because:

- retrieval quality looks poor;
- a method performs better on a small pool;
- the historical setup used a smaller corpus.

A full-corpus feasibility gate is allowed only for genuine resource
constraints. A fixable implementation defect is not evidence of scientific
infeasibility.

## 8. Resource Feasibility Gate

This document freezes the gate architecture but does not invent numeric
resource budgets.

Before the resource benchmark, a later pre-observation resource protocol must
freeze:

- available persistent-disk budget;
- host RAM budget;
- GPU VRAM budget;
- maximum GPU-hours;
- maximum monetary cost;
- maximum acceptable index-build wall time;
- maximum acceptable retrieval/search wall time;
- deterministic pilot corpus sizes;
- candidate-pool setting used for latency measurement;
- minimum scientifically acceptable fallback corpus size `M_min`;
- a 20% capacity reserve;
- a fixed projection method.

Benchmarking must use query-independent deterministic corpus subsets or
prefixes and BEIR train/development queries only. It must not use:

- protected-final queries;
- protected answers;
- qrel scores;
- gold documents;
- protected retrieval metrics.

At multiple increasing corpus sizes, measure:

- final disk usage;
- peak temporary disk usage;
- host RSS;
- GPU memory;
- corpus/manifest build time;
- encoding/index time;
- fingerprint/reload time;
- p50/p95 retrieval latency;
- throughput;
- OOM, failure, and swap behavior;
- resumability.

Project full-corpus requirements conservatively using a fixed 1.25 multiplier
on measured or projected requirements, in addition to the predeclared 20%
capacity reserve.

The full corpus passes only if all four retrievers can satisfy the frozen
resource and schedule constraints. If one retriever cannot satisfy the resource
gate, a retriever-specific corpus is prohibited. The allowed responses are:

1. obtain sufficient resources;
2. make a scientifically neutral scalable implementation improvement;
3. invoke the one shared fallback policy in Section 9.

## 9. Fallback Corpus Policy

Fallback is allowed only if the predeclared full-corpus resource gate fails.
The fallback must be one global, deterministic, query-independent,
retriever-independent, diversification-independent, and LLM-independent
subsample of the same BEIR HotpotQA corpus.

Fallback membership must never use:

- query IDs;
- question text;
- qrels;
- answers;
- gold documents;
- retrieval outputs;
- relevance scores;
- generated answers.

Historical-500 gold or pooled documents must not be preserved deliberately. A
historical document may enter only through the global hash-selection rule.

If fallback is used, the resulting experiments must be clearly labelled as
reduced-corpus experiments and must not be described as full BEIR corpus
retrieval.

## 10. Fallback Construction

The fallback membership priority rule is frozen exactly as:

```text
priority(document_id) =
SHA256(
  UTF8(
    "context-matters-rag|sprint3|HotpotQA|"
    "GLOBAL_CORPUS_FALLBACK|20260822|"
    + document_id
  )
).hexdigest()
```

The construction rules are:

- preserve the exact canonical BEIR source document-ID string;
- sort by `(lowercase_hex_digest, UTF-8 document-ID bytes)`;
- once `M` is resource-derived, choose the first `M` memberships in this order;
- restore selected records to canonical full-corpus document order for corpus
  positions;
- preserve exact title, body, and `retrieval_content`;
- use exactly the same fallback manifest everywhere.

## 11. Fallback Size M

This document does not freeze a numeric `M`. Instead, it freezes:

```text
M = largest m <= full corpus size
    such that every retriever and the combined persistent artifacts
    satisfy the PREDECLARED disk/RAM/VRAM/time/cost limits,
    the 20% capacity reserve,
    and the fixed 1.25 resource projection multiplier.
```

`M` may be determined only from resource measurements. It may not be selected
using:

- recall;
- qrel coverage;
- downstream correctness;
- retrieval quality;
- LLM quality;
- faithfulness;
- any performance metric.

A minimum acceptable `M_min` must be fixed in the later resource protocol
before the benchmark is run. If the resource-derived `M` is less than `M_min`,
stop rather than accepting a scientifically inadequate corpus.

## 12. Historical 500 Bridge

The historical-500 bridge is frozen as optional secondary analysis only. If it
is later run, its label is:

```text
CANONICAL_REPLICATION_BRIDGE
```

It may compare the exposed historical 500 against the new canonical corpus and
protocol. It must never:

- tune methods;
- choose an MMR lambda;
- select DPP or clustering settings;
- choose a prompt or LLM;
- choose candidate pool or top-k;
- calibrate evaluator thresholds;
- enter protected-final statistics;
- be pooled with protected-final rows.

Historical Sprint 2 outputs remain unchanged and retain the
`HISTORICAL_OBSERVED` label.

## 13. Stale Repository Notes

The repository audit reported some items as unresolved that have subsequently
been methodologically frozen elsewhere, including:

- canonical generation protocol architecture;
- faithfulness/hallucination construct architecture.

This document does not reopen those decisions. The exact decomposer winner
remains development-selected under the separate committed decomposer-selection
protocol.

## 14. Still Unresolved After This Document

This document does not silently resolve:

- actual protected-final manifest materialization;
- exact HotpotQA corpus source revision and count verification;
- canonical HotpotQA title/body/`retrieval_content` mapping;
- numeric resource budgets;
- resource pilot sizes;
- `M_min`;
- actual resource-feasibility outcome;
- actual fallback `M`, if fallback is ever triggered;
- final candidate pool and top-k;
- exact diversification configuration matrix;
- final statistical protocol.

## 15. Frozen Before Observation

This protocol freezes `N = 500`, the ID-only sample rule, full-corpus primary
policy, shared fallback architecture, fallback membership rule, and optional
historical bridge policy before protected-final retrieval results are observed.

No protected-final query IDs are materialized or inspected as part of writing
this document.
