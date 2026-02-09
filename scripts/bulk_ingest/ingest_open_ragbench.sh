#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: $0 <embedding-model>}"
M=$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

DB="open_ragbench_test_${M}"
pueue add -l "open-ragbench" \
  "autorag-research ingest --name=open-ragbench --db-name=${DB} --embedding-model=${MODEL} \
   && autorag-research data dump --db-name=${DB} \
   && autorag-research data upload ${DB}.dump open-ragbench ${MODEL}"

echo "Queued 1 Open-RAGBench task."
