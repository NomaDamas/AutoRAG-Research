# Retrieval Pipelines

Algorithms that take a query and return relevant documents.

## Available Pipelines

| Pipeline | Algorithm | Modality |
|----------|-----------|----------|
| [BM25](bm25.md) | Sparse (term frequency) | Text |
| [Hybrid](hybrid.md) | RRF / Convex Combination | Text |

## Base Class

All retrieval pipelines extend `BaseRetrievalPipeline`:

```python
from autorag_research.pipelines.retrieval import BaseRetrievalPipeline


class MyRetrievalPipeline(BaseRetrievalPipeline):
    def _get_retrieval_func(self):
        def retrieve(queries: list[str], top_k: int) -> list[list[dict]]:
            # Return list of results per query
            # Each result: {"doc_id": ..., "score": ...}
            pass

        return retrieve

    def _get_pipeline_config(self):
        return {"type": "my_pipeline"}
```

## Methods

| Method | Description |
|--------|-------------|
| `retrieve(query, top_k)` | Single query retrieval |
| `run(top_k, batch_size)` | Batch retrieval for all queries |
