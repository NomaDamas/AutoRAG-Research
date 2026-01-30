# Core Concepts

## Dataset

A dataset contains:

- **Documents**: Content to search (text or images)
- **Queries**: Questions to answer
- **Ground Truth**: Which documents are relevant to each query

Stored in PostgreSQL with vector embeddings.

## Retrieval Pipeline

Takes a query, returns relevant documents.

```python
results = retrieval_pipeline.retrieve(query="What causes fever?", top_k=10)
# Returns: [{"doc_id": 42, "score": 0.95}, ...]
```

## Generation Pipeline

Takes a query + retrieved documents, generates an answer.

```python
answer = generation_pipeline.generate(query="What causes fever?", top_k=5)
# Returns: "Fever is caused by..."
```

## Metric

Measures pipeline quality against ground truth.

**Retrieval metrics**: Did we find the right documents? (Recall, NDCG, MRR)

**Generation metrics**: Is the answer correct? (ROUGE, BERTScore)

## Executor

Orchestrates pipeline execution and metric evaluation.

```yaml
# experiment.yaml
db_name: beir_scifact_test

pipelines:
  retrieval: [bm25]

metrics:
  retrieval: [recall, ndcg]
```
