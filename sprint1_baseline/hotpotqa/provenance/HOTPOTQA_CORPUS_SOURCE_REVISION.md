# HotpotQA Canonical Corpus Source Revision

## Status

Prospectively frozen before canonical HotpotQA SELECTION or
PROJECT_PROTECTED_FINAL retrieval outcomes are observed.

## Canonical source

Source repository:

`BeIR/hotpotqa`

Hugging Face revision:

`a7e8bab212f5a89f9be1bc9b654aa6dfa317f32b`

Configuration:

`corpus`

Split:

`corpus`

Expected document count:

`5,233,329`

Source fields:

- `_id`
- `title`
- `text`

The exact BEIR `_id` string is the canonical source document identifier.

## Canonical retrieval surface

Serialization version:

`hotpot_retrieval_content_v1`

For every document:

`retrieval_content = (title + " " + text).strip()`

This is governed by
`docs/sprint3/HOTPOTQA_RETRIEVAL_TEXT_CONTRACT_AMENDMENT.md`.

The same logical corpus and exact retrieval_content must be supplied to:

- BM25
- DPR
- Contriever
- ColBERTv2

Physical indexes may differ. Corpus membership and supplied scientific text
may not differ by retriever.

## Verification performed

Before this record was created, the pinned revision was resolved with
Hugging Face datasets metadata using the `corpus` configuration.

Observed:

- builder: `parquet`
- config: `corpus`
- split: `corpus`
- num_examples: `5,233,329`
- fields: `_id`, `title`, `text`

No HotpotQA SELECTION or PROJECT_PROTECTED_FINAL retrieval effectiveness
outcome was used to select this source revision.

## Physical acquisition still required

This record freezes the upstream source revision only.

After physical corpus acquisition, separately verify and bind:

- actual row count
- unique exact `_id` values
- canonical row ordering
- source title/text identities
- retrieval_content identities
- per-record hashes or equivalent auditable identity
- aggregate logical corpus hash
- physical artifact inventory and hashes

A metadata count alone is not a substitute for physical corpus verification.

## Historical Sprint-2 separation

`src/data_prep_hotpot_beir.py` and the historical approximately
25,997-document gold/negative pooled corpus remain HISTORICAL_OBSERVED.

They are not the canonical Sprint-3 HotpotQA corpus and must not be reused as
such.

## Canonical corpus policy

The complete BEIR HotpotQA corpus is the only authorized canonical Sprint-3
HotpotQA retrieval corpus.

No reduced canonical fallback is authorized.
