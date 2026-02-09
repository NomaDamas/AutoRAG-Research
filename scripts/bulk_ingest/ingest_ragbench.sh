#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: $0 <embedding-model>}"
M=$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

CONFIGS=(
  covidqa
  cuad
  delucionqa
  emanual
  expertqa
  finqa
  hagrid
  hotpotqa
  msmarco
  pubmedqa
  tatqa
  techqa
)

for c in "${CONFIGS[@]}"; do
  DB="ragbench_${c}_test_${M}"
  pueue add -l "ragbench-${c}" \
    "autorag-research ingest --name=ragbench --extra config=${c} --embedding-model=${MODEL} \
     && autorag-research data dump --db-name=${DB} \
     && autorag-research data upload ${DB}.dump ragbench ${c}_${MODEL}"
done

echo "Queued ${#CONFIGS[@]} RAGBench tasks."
