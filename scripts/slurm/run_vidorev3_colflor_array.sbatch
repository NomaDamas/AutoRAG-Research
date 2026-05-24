#!/bin/bash
#SBATCH --job-name=vidorev3_colflor
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=/data/home/jeffrey/Projects-dev/AutoRAG-Research/experiments/vidorev3_colflor/logs/%A_%a.out
#SBATCH --error=/data/home/jeffrey/Projects-dev/AutoRAG-Research/experiments/vidorev3_colflor/logs/%A_%a.err

set -euo pipefail
cd /data/home/jeffrey/Projects-dev/AutoRAG-Research
export VIRTUAL_ENV=/data/home/jeffrey/Projects-dev/AutoRAG-Research/.venv310
export PATH="$VIRTUAL_ENV/bin:$PATH"
export LD_LIBRARY_PATH=$(find "$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia" -type d -name lib | paste -sd: -):${LD_LIBRARY_PATH:-}
export HF_HOME=${HF_HOME:-/data/home/jeffrey/.hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data/home/jeffrey/.hf_cache/transformers}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p experiments/vidorev3_colflor/logs experiments/vidorev3_colflor/results
MODELS_JSON=experiments/vidorev3_colflor/models.json
DOMAINS=experiments/vidorev3_colflor/domains.txt
NUM_DOMAINS=$(wc -l < "$DOMAINS")
TASK=${SLURM_ARRAY_TASK_ID:?}
MODEL_IDX=$(( TASK / NUM_DOMAINS ))
DOMAIN_IDX=$(( TASK % NUM_DOMAINS + 1 ))
MODEL_NAME=$($VIRTUAL_ENV/bin/python - <<PY
import json
m=json.load(open('$MODELS_JSON'))[$MODEL_IDX]
print(m['name'])
PY
)
MODEL_PATH=$($VIRTUAL_ENV/bin/python - <<PY
import json
m=json.load(open('$MODELS_JSON'))[$MODEL_IDX]
print(m['path'])
PY
)
DOMAIN=$(sed -n "${DOMAIN_IDX}p" "$DOMAINS")

echo "TASK=$TASK MODEL_IDX=$MODEL_IDX DOMAIN_IDX=$DOMAIN_IDX MODEL=$MODEL_NAME DOMAIN=$DOMAIN PATH=$MODEL_PATH"
$VIRTUAL_ENV/bin/python scripts/evaluate_vidorev3_colflor_direct.py \
  --model-name "$MODEL_NAME" \
  --model-path "$MODEL_PATH" \
  --domain "$DOMAIN" \
  --output-dir experiments/vidorev3_colflor/results \
  --query-batch-size ${QUERY_BATCH_SIZE:-16} \
  --image-batch-size ${IMAGE_BATCH_SIZE:-8} \
  --top-k 10 \
  --device cuda
