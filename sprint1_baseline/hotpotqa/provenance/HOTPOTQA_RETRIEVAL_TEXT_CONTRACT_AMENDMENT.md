# Sprint 3 HotpotQA Retrieval-Text Contract Amendment

## 1. Purpose

This prospective amendment freezes the exact canonical document text supplied
to every Sprint-3 HotpotQA retriever.

It supersedes only older HotpotQA wording that leaves the title, body, or
`retrieval_content` construction unresolved. It does not reopen any other
HotpotQA methodology.

## 2. Canonical Source Fields

The source fields are the BEIR HotpotQA `title` and `text` strings.

The canonical serialization version is:

```text
hotpot_retrieval_content_v1
```

## 3. Canonical Retrieval-Content Formula

For every BEIR HotpotQA document, freeze:

```text
retrieval_content = (title + " " + text).strip()
```

The injected title/body separator is exactly one ASCII space, U+0020.

The same exact pre-tokenizer `retrieval_content` string must be supplied to:

- BM25;
- DPR;
- Contriever;
- ColBERTv2.

No retriever-specific title/body construction is permitted.

## 4. Empty and Whitespace-Only Fields

Apply the following consequences of the canonical formula:

- If `title` is empty or whitespace-only:

  ```text
  retrieval_content = text.strip()
  ```

- If `text` is empty or whitespace-only:

  ```text
  retrieval_content = title.strip()
  ```

- If both fields are empty or whitespace-only:

  ```text
  retrieval_content = ""
  ```

  Retain the document in canonical corpus membership. Do not silently drop or
  replace it. Any backend inability to represent it is an implementation or
  resource blocker, not permission to alter corpus membership.

## 5. Character and Whitespace Preservation

Preserve all internal source characters and embedded whitespace or newlines.
Only the outer `.strip()` implied by the frozen formula is applied.

Do not:

- lowercase;
- normalize Unicode;
- collapse internal whitespace;
- remove punctuation;
- detect or remove duplicated titles;
- inject separators other than the one ASCII space specified by the formula;
- use retriever-specific title/body construction.

The one U+0020 separator is the exact character inserted between the unmodified
source `title` and `text` values before the final outer `.strip()`. Existing
source whitespace adjacent to that separator is not independently normalized.

## 6. DPR Input Contract

DPR continues to receive an empty separate-title field in the current frozen
scientific adapter. The complete canonical `retrieval_content` string is
supplied as DPR's passage-text input.

This is deliberate controlled-comparison behavior. Do not pass the BEIR title
through DPR's separate title field.

## 7. Retriever-Specific Processing

Model- or tokenizer-specific tokenization, truncation, encoding,
representation, indexing, and scoring remain retriever-specific. Those
operations occur after construction of the shared `retrieval_content` and do
not change its scientific identity.

## 8. Scientific Rationale

BEIR exposes title and text separately, and BEIR retrieval implementations
commonly incorporate both. The project requires identical supplied document
content across retrievers.

Accordingly, this shared serialization is a prospective controlled-comparison
convention aligned with BEIR document semantics.

This amendment does not claim that the serialization exactly reproduces every
native BEIR retriever's internal field handling.

The contract is not selected from retrieval performance, answer performance,
SELECTION outcomes, or protected-final outcomes.

## 9. Preserved Methodology

This amendment preserves without change:

- the complete BEIR HotpotQA corpus as the only canonical corpus;
- the expected count of 5,233,329 documents, subject to physical verification;
- exact source document IDs;
- document membership and passage boundaries;
- the prohibition on query-specific corpus construction or gold/qrel injection;
- `candidate_pool=20`;
- `top_k=5`;
- all four retrievers and their frozen checkpoints/configurations;
- the full-corpus resource gate and its stop behavior;
- dataset partitions and evidence roles;
- diversification methodology;
- generation methodology;
- all other frozen Sprint-3 methodology.

## 10. Provenance and Artifact Identity

Canonical HotpotQA corpus and retrieval provenance must bind or reference:

- serialization version `hotpot_retrieval_content_v1`;
- exact source title;
- exact source text;
- exact canonical `retrieval_content` hash;
- canonical corpus manifest and hash;
- each retriever's index identity.

The corpus manifest must make the source title, source text, and prepared
`retrieval_content` identities independently auditable.

## 11. Change and Invalidation Rule

Any future change to this serialization requires:

- a prospective methodology amendment made before applicable outcomes are
  inspected;
- a new corpus-surface/serialization version;
- new `retrieval_content` hashes and corpus-manifest identity;
- rebuilding all four canonical retriever indexes;
- rerunning all dependent canonical retrieval and candidate artifacts;
- invalidating and rematerializing every downstream artifact whose identity
  depends on those retrieval outputs.

Do not reuse an index or retrieval artifact created under a different
serialization version.

## 12. Supersession

This amendment supersedes only earlier HotpotQA statements that classify the
title/body/`retrieval_content` mapping as unresolved.

It does not supersede or modify the full-corpus authority, resource-gate,
dataset, retriever, diversification, generation, evaluation, selection, or
stage-gate rules except where those documents require this exact text contract
to be frozen before indexing.

## 13. Frozen Before Observation

This amendment is frozen before Stage-3 SELECTION or protected-final outcomes
are inspected.

No SELECTION or protected-final result was used to choose the serialization,
its separator, its whitespace behavior, or its shared cross-retriever
application.
