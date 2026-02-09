#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: $0 <embedding-model>}"
M=$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

CONFIGS=(
  hr
  finance_en
  industrial
  pharmaceuticals
  computer_science
  energy
  physics
  finance_fr
)

for c in "${CONFIGS[@]}"; do
  DB="vidorev3_${c}_image_test_${M}"
  pueue add -l "vidorev3-${c}" \
    "autorag-research ingest --name=vidorev3 --extra config-name=${c} --embedding-model=${MODEL} \
     && autorag-research data dump --db-name=${DB} \
     && autorag-research data upload ${DB}.dump vidorev3 ${c}_image_${MODEL}"
done

echo "Queued ${#CONFIGS[@]} ViDoReV3 tasks."
