# HotpotQA Official-Test Evaluation Amendment — 2026-09-05

## Status

This prospective amendment is frozen before any new canonical full-corpus
HotpotQA retrieval results for the evaluation population below are observed.

## Query population

Earlier project protocols defined a project-specific protected-final sample
of 500 HotpotQA test questions.

For the remaining project experiments, HotpotQA will instead use the complete
official BEIR HotpotQA test split:

- official BEIR test: 7,405 questions
- historically exposed test questions: 500
- previously unexposed test questions: 6,905
- canonical evaluation population: all 7,405 questions

The canonical evidence label is `OFFICIAL_TEST_FULL`.

Because 500 questions were used in earlier project work, the complete
7,405-question result must not be described as entirely unseen or entirely
protected-final.

## Unchanged retrieval contract

The following remain unchanged:

- corpus: complete BEIR HotpotQA corpus
- document count: 5,233,329
- retrieval text: `(title + " " + text).strip()`
- retrievers: BM25, DPR, Contriever, ColBERTv2
- candidate pool: 20
- selected context: top 5
- historical 25,997-document pooled corpus remains prohibited

## Resource pilots

The completed resource pilots remain valid development-only engineering
evidence:

- corpus sizes: 100k, 500k, 1M
- deterministic DEVELOPMENT queries: 220
- warmup: 20
- measured: 200
- candidate pool: 20

They do not need to be rerun.

## Scope

The full 7,405-query population applies to new canonical HotpotQA experiments
for Sprint 1, Sprint 2, and Sprint 3 so that paired comparisons use the same
official test population.

For Sprint-1 baseline generation:

`7,405 questions × 5 conditions × 3 generators = 111,075 generations`.

## Superseded clauses

For HotpotQA query-population selection only, this amendment supersedes
earlier clauses requiring `PROJECT_PROTECTED_FINAL N=500` or sampling 500
queries from the 6,905-query unexposed complement.

All non-conflicting corpus, retrieval, diversification, generation,
evaluation, and provenance contracts remain unchanged.

Any future change to this 7,405-query decision requires another prospective
amendment before inspecting the affected canonical outcomes.
