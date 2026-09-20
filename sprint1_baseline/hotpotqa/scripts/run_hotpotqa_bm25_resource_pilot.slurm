#!/bin/bash
#SBATCH --job-name=hp_bm25
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3/logs/hp_bm25_%A_%a.out
#SBATCH --error=/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3/logs/hp_bm25_%A_%a.err

set -euo pipefail

WS="/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3"
REPO="$WS/context-matters-rag-sprint3"
PYTHON="$WS/envs/colbert/bin/python"

SIZES=(100000 500000 1000000)
SIZE="${SIZES[$SLURM_ARRAY_TASK_ID]}"

cd "$REPO"

echo "=== HOTPOT BM25 RESOURCE PILOT ==="
echo "job=$SLURM_JOB_ID"
echo "array_task=$SLURM_ARRAY_TASK_ID"
echo "corpus_size=$SIZE"
echo "host=$(hostname)"
echo "commit=$(git rev-parse HEAD)"
date

"$PYTHON" scripts/run_hotpotqa_resource_pilot.py \
  --retriever bm25 \
  --corpus-size "$SIZE" \
  --dataset-cache-dir "$WS/data/hotpotqa/hf_cache" \
  --positions-dir "$WS/data/hotpotqa/resource_pilots/corpus_positions" \
  --query-manifest "$REPO/artifacts/resource_pilots/hotpotqa_resource_queries_v1.json" \
  --cache-root "$WS/data/hotpotqa/resource_pilots/retriever_cache" \
  --output-root "$WS/data/hotpotqa/resource_pilots/results"

echo "=== COMPLETE ==="
date
