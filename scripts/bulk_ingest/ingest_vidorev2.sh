#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: $0 <embedding-model>}"
M=$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

DATASETS=(
  esg_reports_v2
  biomedical_lectures_v2
  economics_reports_v2
)

for ds in "${DATASETS[@]}"; do
  DB="vidorev2_${ds}_test_${M}"
  pueue add -l "vidorev2-${ds}" \
    "autorag-research ingest --name=vidorev2 --extra dataset-name=${ds} --embedding-model=${MODEL} \
     && autorag-research data dump --db-name=${DB} \
     && autorag-research data upload ${DB}.dump vidorev2 ${ds}_${MODEL}"
done

echo "Queued ${#DATASETS[@]} ViDoReV2 tasks."
