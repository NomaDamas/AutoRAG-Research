#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: $0 <embedding-model>}"
M=$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

# Format: "domain:mode"
# Domains with both short and long modes
# leetcode, aops, theoremqa_theorems, theoremqa_questions have short mode only
TASKS=(
  "biology:short"
  "earth_science:short"
  "economics:short"
  "psychology:short"
  "robotics:short"
  "stackoverflow:short"
  "sustainable_living:short"
  "pony:short"
  "leetcode:short"
  "aops:short"
  "theoremqa_theorems:short"
  "theoremqa_questions:short"
  "biology:long"
  "earth_science:long"
  "economics:long"
  "psychology:long"
  "robotics:long"
  "stackoverflow:long"
  "sustainable_living:long"
  "pony:long"
)

for entry in "${TASKS[@]}"; do
  IFS=: read -r domain mode <<< "${entry}"
  DB="bright_${domain}_${mode}_test_${M}"
  EXTRA="--extra domain=${domain} --extra document-mode=${mode}"
  pueue add -l "bright-${domain}-${mode}" \
    "autorag-research ingest --name=bright ${EXTRA} --embedding-model=${MODEL} --db-name=${DB} \
     && autorag-research data dump --db-name=${DB} \
     && autorag-research data upload ${DB}.dump bright ${domain}_${mode}_${MODEL}; ret=\$?; rm -f ${DB}.dump; exit \$ret"
done

echo "Queued ${#TASKS[@]} BRIGHT tasks."
