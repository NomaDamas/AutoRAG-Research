#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: $0 <embedding-model>}"
M=$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

DATASETS=(
  ArxivQA
  ChartQA
  MP-DocVQA
  InfoVQA
  PlotQA
  SlideVQA
)

for ds in "${DATASETS[@]}"; do
  d=$(echo "${ds}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
  DB="visrag_${d}_train_${M}"
  pueue add -l "visrag-${ds}" \
    "autorag-research ingest --name=visrag --extra dataset-name=${ds} --subset=train --embedding-model=${MODEL} \
     && autorag-research data dump --db-name=${DB} \
     && autorag-research data upload ${DB}.dump visrag ${d}_${MODEL}"
done

echo "Queued ${#DATASETS[@]} VisRAG tasks."
