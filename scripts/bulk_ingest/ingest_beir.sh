#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: $0 <embedding-model>}"
M=$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

DATASETS=(
  trec-covid
  nfcorpus
  nq
  hotpotqa
  fiqa
  arguana
  webis-touche2020
  cqadupstack
  quora
  dbpedia-entity
  scidocs
  fever
  climate-fever
  scifact
)

for ds in "${DATASETS[@]}"; do
  d="${ds//-/_}"
  DB="beir_${d}_test_${M}"
  pueue add -l "beir-${ds}" \
    "autorag-research ingest --name=beir --extra dataset-name=${ds} --embedding-model=${MODEL} --db-name=${DB} \
     && autorag-research data dump --db-name=${DB} \
     && autorag-research data upload ${DB}.dump beir ${ds}_${MODEL}; ret=\$?; rm -f ${DB}.dump; exit \$ret"
done

echo "Queued ${#DATASETS[@]} BEIR tasks."
