#!/bin/bash

# BeIR datasets (scifact 제외)
DATASETS=(
    "msmarco"
    "trec-covid"
    "nfcorpus"
    "nq"
    "hotpotqa"
    "fiqa"
    "arguana"
    "webis-touche2020"
    "cqadupstack"
    "quora"
    "dbpedia-entity"
    "scidocs"
    "fever"
    "climate-fever"
)

EMBEDDING_MODEL="google/embeddinggemma-300m"
API_BASE="http://mkseoul2.iptime.org:12800/v1"

for DATASET in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Processing dataset: $DATASET"
    echo "=========================================="

    python3 scripts/ingest_text_only_retrieval_dataset.py \
        --dataset-type beir \
        --dataset-name "$DATASET" \
        --embedding-model-name "$EMBEDDING_MODEL" \
        --api-base "$API_BASE" \
        --dump-path "./dump_files/${DATASET}-embeddinggemma-300m.dump"

    if [ $? -eq 0 ]; then
        echo "✓ $DATASET completed successfully"
    else
        echo "✗ $DATASET failed"
    fi
    echo ""
done

echo "All datasets processed!"
