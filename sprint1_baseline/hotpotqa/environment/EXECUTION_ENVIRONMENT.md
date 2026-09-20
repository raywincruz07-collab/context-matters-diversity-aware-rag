# HotpotQA Sprint 1 — Execution Environment Provenance

This directory preserves environment information relevant to reproducing the
HotpotQA Sprint 1 baseline experiments.

## Full retrieval execution

The four baseline retrievers were executed over the frozen full HotpotQA corpus
of 5,233,329 documents and the complete official test population of 7,405
queries with candidate pool size 20.

- BM25 canonical output was preserved on bwUniCluster.
- DPR, Contriever, and ColBERT canonical outputs were preserved in the RunPod
  workspace backup.
- The completed ColBERT retrieval summary records an NVIDIA GeForce RTX 5090.
- The completed ColBERT execution log directly references:

  `/workspace/venvs/context-matters-cu130/lib/python3.12/site-packages/`

- The preserved RunPod environment provenance records:
  - Python 3.12.3
  - NVIDIA GeForce RTX 5090
  - driver 580.159.03
  - compute capability 12.0
  - PyTorch 2.13.0+cu130
  - CUDA 13.0

## Dependency locks

`requirements-lock-colbert-runpod-cu130.txt` is retained as the closest frozen
reproduction lock for the ColBERT-compatible CUDA 13.0 stack.

A preserved RunPod virtual environment named
`context-matters-colbert-cu130` matched that lock exactly for all 153 locked
packages.

The full HotpotQA ColBERT execution itself used the separate
`context-matters-cu130` environment. Its preserved package inventory retained
the same key scientific stack:

- torch 2.13.0+cu130
- transformers 4.57.6
- huggingface_hub 0.36.2
- colbert-ai 0.2.22
- RAGatouille 0.0.9.post2
- sentence-transformers 5.7.0
- datasets 5.0.1
- faiss-cpu 1.15.0
- numpy 1.26.4
- pandas 2.2.2
- scikit-learn 1.9.0
- accelerate 1.14.0

Relative to `requirements-lock-colbert-runpod-cu130.txt`, the preserved
`context-matters-cu130` environment differed in five ancillary package versions:

- charset-normalizer 3.5.1 instead of 3.4.9
- filelock 3.32.5 instead of 3.29.0
- idna 3.19 instead of 3.18
- regex 2026.9.3 instead of 2026.7.19
- typing-extensions 4.16.0 instead of 4.15.0

Accordingly, the included lock file is a frozen reproducibility reference and
must not be misrepresented as a byte-for-byte `pip freeze` of the exact
environment used by the full HotpotQA ColBERT execution.

`requirements-lock-colbert-local-cpu.txt` is retained for the validated local
CPU-compatible ColBERT environment.

## bwUniCluster resource-pilot environment

Historical HotpotQA resource-pilot SLURM scripts referenced the shared
bwUniCluster environment:

`/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3/envs/colbert`

A later inspection of that environment reported Python 3.11.16,
PyTorch 2.9.1+cu128, Transformers 4.57.6, huggingface_hub 0.36.2, and
colbert-ai 0.2.22. This environment is evidence for the bwUniCluster
resource-pilot/runtime path and must not be conflated with the canonical
RunPod CUDA 13.0 full-retrieval environment.

## Generation model provenance

The generation configuration is preserved in
`../configs/maki_model_bindings_v7.json`.

Recorded logical-to-physical mappings across all 111,075 HotpotQA generation
records are:

- `gemma4-26b` -> `gemma4-26b`
- `llama-3.3-70b` -> `llama-3.3-70b`
- `ministral-3-14b` -> `qwen3.6-36b`

The `ministral-3-14b` logical condition must remain named exactly as recorded;
the physical serving model is preserved separately as provenance.
