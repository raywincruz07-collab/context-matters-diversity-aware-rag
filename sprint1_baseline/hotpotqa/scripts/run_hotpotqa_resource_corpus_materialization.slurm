#!/bin/bash
#SBATCH --job-name=hp_corpus_sel
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3/logs/hp_corpus_sel_%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3/logs/hp_corpus_sel_%j.err

set -euo pipefail

WS="/pfs/work9/workspace/scratch/ma_rthummar-context_matters_sprint3"
REPO="$WS/context-matters-rag-sprint3"
PYTHON="$WS/envs/colbert/bin/python"

cd "$REPO"

echo "=== START ==="
date
hostname
echo "commit: $(git rev-parse HEAD)"

"$PYTHON" scripts/materialize_hotpotqa_resource_corpus_positions.py \
  --cache-dir "$WS/data/hotpotqa/hf_cache" \
  --output-dir "$WS/data/hotpotqa/resource_pilots/corpus_positions"

echo "=== OUTPUTS ==="
ls -lh "$WS/data/hotpotqa/resource_pilots/corpus_positions"

echo "=== COMPLETE ==="
date
