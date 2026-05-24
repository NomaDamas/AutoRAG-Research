#!/bin/bash
#SBATCH --job-name=vidorev3_smoke
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/jeffrey/Projects-dev/AutoRAG-Research/experiments/vidorev3_colflor/logs/smoke_%j.out
#SBATCH --error=/data/home/jeffrey/Projects-dev/AutoRAG-Research/experiments/vidorev3_colflor/logs/smoke_%j.err
set -euo pipefail
cd /data/home/jeffrey/Projects-dev/AutoRAG-Research
export VIRTUAL_ENV=/data/home/jeffrey/Projects-dev/AutoRAG-Research/.venv310
export PATH="$VIRTUAL_ENV/bin:$PATH"
export LD_LIBRARY_PATH=$(find "$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia" -type d -name lib | paste -sd: -):${LD_LIBRARY_PATH:-}
export HF_HOME=${HF_HOME:-/data/home/jeffrey/.hf_cache}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
$VIRTUAL_ENV/bin/python scripts/evaluate_vidorev3_colflor_direct.py \
  --model-name smoke_baseline \
  --model-path /data/home/jeffrey/banana_colflor_baseline/output/training/colflor-vdr-50k-b32-baseline/best \
  --domain hr \
  --output-dir experiments/vidorev3_colflor/smoke_results \
  --limit-queries 2 \
  --limit-corpus 5 \
  --query-batch-size 1 \
  --image-batch-size 1
