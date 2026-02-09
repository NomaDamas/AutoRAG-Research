#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: $0 <embedding-model>}"
M=$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

# MTEB retrieval task names (case-sensitive for the CLI)
TASKS=(
  NFCorpus
  SciFact
  MSMARCO
  ArguAna
  FiQA2018
  TRECCOVID
  Touche2020
  NQ
  HotpotQA
  FEVER
  DBPedia
  ClimateFEVER
  QuoraRetrieval
  CQADupstackRetrieval
  SciDocs
)

for task in "${TASKS[@]}"; do
  t=$(echo "${task}" | tr '[:upper:]' '[:lower:]')
  DB="mteb_${t}_test_${M}"
  pueue add -l "mteb-${task}" \
    "autorag-research ingest --name=mteb --extra task-name=${task} --embedding-model=${MODEL} --db-name=${DB} \
     && autorag-research data dump --db-name=${DB} \
     && autorag-research data upload ${DB}.dump mteb ${t}_${MODEL}; ret=\$?; rm -f ${DB}.dump; exit \$ret"
done

echo "Queued ${#TASKS[@]} MTEB tasks."
