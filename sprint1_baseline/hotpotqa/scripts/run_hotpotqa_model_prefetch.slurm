#!/bin/bash
#SBATCH --job-name=hp_models
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3/logs/hp_models_%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3/logs/hp_models_%j.err

set -euo pipefail

WS="/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3"
PYTHON="$WS/envs/colbert/bin/python"

export HF_HOME="$WS/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

mkdir -p "$HF_HOME"

"$PYTHON" - <<'PY'
from transformers import (
    AutoModel,
    AutoTokenizer,
    DPRContextEncoder,
    DPRQuestionEncoder,
)

print("=== DPR QUESTION ===", flush=True)
qid = "facebook/dpr-question_encoder-single-nq-base"
qrev = "d04a52f6d2f96c60117a925e8c24c4043a75f265"
AutoTokenizer.from_pretrained(qid, revision=qrev)
DPRQuestionEncoder.from_pretrained(qid, revision=qrev)

print("=== DPR CONTEXT ===", flush=True)
cid = "facebook/dpr-ctx_encoder-single-nq-base"
crev = "bb21a3c2b1656d60c6a8e920283bc40dabddadb8"
AutoTokenizer.from_pretrained(cid, revision=crev)
DPRContextEncoder.from_pretrained(cid, revision=crev)

print("=== CONTRIEVER ===", flush=True)
mid = "facebook/contriever"
mrev = "2bd46a25019aeea091fd42d1f0fd4801675cf699"
AutoTokenizer.from_pretrained(mid, revision=mrev)
AutoModel.from_pretrained(mid, revision=mrev)

print("MODEL PREFETCH: PASS", flush=True)
PY
