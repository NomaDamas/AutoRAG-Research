#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: $0 <embedding-model>}"
M=$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

LANGUAGES=(
  arabic
  bengali
  english
  finnish
  indonesian
  japanese
  korean
  russian
  swahili
  telugu
  thai
)

for lang in "${LANGUAGES[@]}"; do
  DB="mrtydi_${lang}_test_${M}"
  pueue add -l "mrtydi-${lang}" \
    "autorag-research ingest --name=mrtydi --extra language=${lang} --embedding-model=${MODEL} --db-name=${DB} \
     && autorag-research data dump --db-name=${DB} \
     && autorag-research data upload ${DB}.dump mrtydi ${lang}_${MODEL}; ret=\$?; rm -f ${DB}.dump; exit \$ret"
done

echo "Queued ${#LANGUAGES[@]} MrTyDi tasks."
