#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: $0 <embedding-model>}"
M=$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

DATASETS=(
  arxivqa_test_subsampled
  docvqa_test_subsampled
  infovqa_test_subsampled
  tabfquad_test_subsampled
  tatdqa_test
  shiftproject_test
  syntheticDocQA_artificial_intelligence_test
  syntheticDocQA_energy_test
  syntheticDocQA_government_reports_test
  syntheticDocQA_healthcare_industry_test
)

for ds in "${DATASETS[@]}"; do
  d=$(echo "${ds}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
  DB="vidore_${d}_test_${M}"
  pueue add -l "vidore-${ds}" \
    "autorag-research ingest --name=vidore --extra dataset-name=${ds} --embedding-model=${MODEL} --db-name=${DB} \
     && autorag-research data dump --db-name=${DB} \
     && autorag-research data upload ${DB}.dump vidore ${d}_${MODEL}; ret=\$?; rm -f ${DB}.dump; exit \$ret"
done

echo "Queued ${#DATASETS[@]} ViDoRe tasks."
