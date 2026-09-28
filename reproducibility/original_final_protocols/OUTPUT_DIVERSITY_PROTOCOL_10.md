# Phase 4.5 Output Diversity Protocol 10

Date: 2026-09-27

## Scope

Primary dataset: ASQA WITH_CONTEXT only.

Output diversity is measured within each generated ASQA answer using the
already-frozen sentence segmentation from Phase 4.2.

Answers with fewer than two sentences are ineligible for intra-answer
pairwise semantic diversity and are reported separately.

## Frozen population

Source:
`exports/phase42_faithfulness_workload_20260926/asqa_phase42_workload.jsonl`

Total measurable ASQA answers:
68,130

Answers with >=2 sentences:
44,993

Answers with <2 sentences:
23,137

Only the 44,993 eligible answers receive an intra-answer semantic-diversity
score.

## Embedding model

Model:
`facebook/contriever`

Revision:
`2bd46a25019aeea091fd42d1f0fd4801675cf699`

This is the same embedding model and revision frozen for Phase 4.4 retrieval
diversity.

## Sentence embedding semantics

For every sentence:

1. tokenize with the pinned Contriever tokenizer/model revision
2. compute `last_hidden_state`
3. apply attention-mask-aware mean pooling
4. L2-normalize the pooled embedding

No sentence rewriting, paraphrasing, summarization, or filtering is permitted.

## Output-diversity metric

For an answer containing n >= 2 sentence embeddings:

1. construct the cosine-similarity matrix
2. take only the upper triangle with k=1
3. for each sentence pair compute:

   cosine distance = 1 - cosine similarity

4. output diversity is the arithmetic mean of all pairwise cosine distances

This intentionally mirrors the frozen Phase 4.4 retrieval-diversity geometry.

For n < 2, the answer is marked ineligible rather than assigning artificial
semantic diversity.

## Reporting

Per answer preserve:

- dataset
- sample_id
- retriever
- condition
- logical_model_id
- physical_model_id
- status
- sentence_count
- eligible
- output_diversity

Aggregate summaries will be computed by retriever × condition × logical model.

## Scientific role

Phase 4.5 measures semantic variety within generated long-form ASQA answers.

It does not measure:
- correctness
- faithfulness
- retrieval diversity
- diversity across unrelated questions
- Self-BLEU across different samples

No scientific result has been inspected or used to tune this protocol.
