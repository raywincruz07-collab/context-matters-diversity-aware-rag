#!/bin/bash
#SBATCH --job-name=hp_colbert
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3/logs/hp_colbert_%A_%a.out
#SBATCH --error=/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3/logs/hp_colbert_%A_%a.err

set -euo pipefail

# Required by ColBERT residual CUDA extension JIT compilation.
module load devel/cuda/12.8

WS="/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3"
REPO="$WS/context-matters-rag-sprint3"
PYTHON="$WS/envs/colbert/bin/python"

SIZES=(100000 500000 1000000)
SIZE="${SIZES[$SLURM_ARRAY_TASK_ID]}"

export HF_HOME="$WS/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"

echo "=== HOTPOT COLBERT RESOURCE PILOT ==="
echo "job=$SLURM_JOB_ID"
echo "task=$SLURM_ARRAY_TASK_ID"
echo "corpus_size=$SIZE"
echo "host=$(hostname)"
echo "commit=$(git rev-parse HEAD)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
date

nvidia-smi || true

"$PYTHON" scripts/run_hotpotqa_colbert_resource_pilot.py \
  --corpus-size "$SIZE" \
  --dataset-cache-dir "$WS/data/hotpotqa/hf_cache" \
  --positions-dir "$WS/data/hotpotqa/resource_pilots/corpus_positions" \
  --query-manifest "$REPO/artifacts/resource_pilots/hotpotqa_resource_queries_v1.json" \
  --checkpoint "$WS/models/colbertv2.0" \
  --cache-root "$WS/data/hotpotqa/resource_pilots/retriever_cache" \
  --output-root "$WS/data/hotpotqa/resource_pilots/results"

echo "=== COMPLETE ==="
date
