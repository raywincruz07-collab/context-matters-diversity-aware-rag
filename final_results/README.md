# Final Results

This directory provides a compact professor-facing entry point to the frozen
final results of:

**CONTEXT MATTERS — Evaluating Diversity-Aware Retrieval for RAG**

The scientific experiments are complete and frozen. Nothing in this directory
represents a new retrieval run, generation run, diversification run, embedding
run, metric recomputation, NLI run, or plot regeneration.

## Recommended review order

1. `MASTER_RESULTS.csv`
2. `figures/`
3. `FIGURE_INDEX.csv`
4. `../reproducibility/final_notebooks/`
5. `../sprint3_evaluation/`
6. `../reproducibility/source_of_truth/`

## Master results

`MASTER_RESULTS.csv` is the consolidated final reporting table.

It contains:

- 264 final experiment cells
- 50 columns
- PubMedQA: 96 cells
- HotpotQA: 96 cells
- ASQA: 72 cells

Each row corresponds to:

`dataset × retriever × condition × logical generator`

The table consolidates frozen final evidence for:

- correctness
- faithfulness
- hallucination
- coverage
- retrieval diversity
- ASQA output diversity

The master table was constructed only by joining frozen final summary
artifacts. No scientific metric was recomputed.

Canonical SHA-256:

`b77009d4e5dc3900ba58b4b70d4aea8e3755b017afd1ef83aeb1f5c3ac8d953d`

`MASTER_RESULTS_AUDIT.json` records construction provenance and validation.

## Core figures

`figures/` contains the 22 frozen professor-facing core figures used for final
reporting and presentation.

The mapping is:

| Figure | Content |
|---|---|
| F01 | PubMedQA correctness by condition |
| F02 | HotpotQA correctness by condition |
| F03 | ASQA correctness by condition |
| F04 | PubMedQA correctness by generator |
| F05 | HotpotQA correctness by generator |
| F06 | ASQA correctness by generator |
| F07 | Faithfulness by condition |
| F08 | Hallucination by condition |
| F09 | Faithfulness by generator |
| F10 | Hallucination by generator |
| F11 | PubMedQA coverage by condition |
| F12 | HotpotQA coverage by condition |
| F13 | ASQA answer-side coverage by condition |
| F14 | Retrieval diversity by condition |
| F15 | ASQA output diversity by condition |
| F16 | ASQA output diversity by generator |
| F17 | Diversity versus correctness: paired changes |
| F18 | Diversity versus faithfulness: paired changes |
| F19 | Diversity versus hallucination: paired changes |
| F20 | Context/output-diversity association summary |
| F21 | Context/output-diversity operating points |
| F22 | ASQA paraphrase robustness summary |

`FIGURE_INDEX.csv` records the corresponding frozen source names, file sizes,
and SHA-256 hashes.

## Experimental scope

Final datasets:

- PubMedQA — 1,000 questions
- HotpotQA — 7,405 official test questions
- ASQA — 948 dev questions

Final retrievers:

- PubMedQA: BM25, DPR, Contriever, ColBERTv2
- HotpotQA: BM25, DPR, Contriever, ColBERTv2
- ASQA: BM25, DPR, Contriever

Final context-selection conditions:

- `none`
- `mmr_0`
- `mmr_0.25`
- `mmr_0.5`
- `mmr_0.75`
- `kmeans_k2`
- `agglo_k3`
- `dpp_map`

Candidate pool: Top-20

Final generator context: Top-5 passages

Physical generators:

- Gemma 4 26B
- Llama 3.3 70B
- Qwen 3.6 36B

The historical logical identifier `ministral-3-14b` maps to physical
`qwen3.6-36b`.

Total frozen generation artifacts represented by the final experiment matrix:

**875,136**

## Interpretation

The final evidence supports a trade-off interpretation:

> Diversification reliably changes the semantic composition of retrieved
> evidence, while downstream correctness, faithfulness, hallucination and
> output-diversity effects depend on the dataset, retriever, generator and
> operating point.

The results do not support a universal claim that diversification is always
beneficial or always harmful.

## Important reporting limitations

- Final ASQA does not include ColBERTv2.
- Protected-final ASQA `S-Recall@5`, `alpha-nDCG@5`, coverability and `c*`
  are not available as frozen final numeric artifacts.
- ASQA Phase 4.3 `STR-EM` / `STR-Hit` are answer-side measures and must not
  be presented as retrieval-side aspect metrics.
- Retrieval diversity is a manipulation / diagnostic measure, not independent
  proof of retrieval quality.
- Faithfulness and hallucination are not mathematical complements.
- Phase 5.3 associations are descriptive, not causal.
- Phase 5.4 paraphrase robustness is a 30-pair ASQA DEVELOPMENT analysis.

For the canonical limitation record, see:

`../reproducibility/source_of_truth/KNOWN_LIMITATIONS.md`

For scientific provenance, see:

`../reproducibility/source_of_truth/PROVENANCE_MAP.md`

## Executed notebooks

The three canonical professor-facing notebooks with preserved executed outputs
are:

- `../reproducibility/final_notebooks/01_sprint1_baseline_results.ipynb`
- `../reproducibility/final_notebooks/02_sprint2_diversification_results.ipynb`
- `../reproducibility/final_notebooks/03_sprint3_final_evaluation_results.ipynb`

## Detailed result artifacts

The original frozen Phase 4 and Phase 5 summaries remain under:

`../sprint3_evaluation/`

The consolidated master table in this directory does not replace those
artifacts; it provides a convenient reporting and review layer over them.

## Large artifacts

Large corpora, retrieval indexes, embedding caches, raw generation collections,
large NLI outputs, and other heavy scientific artifacts remain external to Git
by design.

Their provenance is represented by manifests, hashes, source identities, counts,
and the repository reproducibility documentation.

See:

`../REPRODUCIBILITY.md`
