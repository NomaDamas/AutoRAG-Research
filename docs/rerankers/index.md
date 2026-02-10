# Rerankers

Re-score retrieved documents to improve ranking quality.

## Available Rerankers

| Reranker | Type | Provider |
|----------|------|----------|
| [Cohere](cohere.md) | API | Cohere |
| [Jina](jina.md) | API | Jina AI |
| [VoyageAI](voyageai.md) | API | Voyage AI |
| [MixedbreadAI](mixedbreadai.md) | API | Mixedbread AI |
| [RankGPT](rankgpt.md) | LLM | Any LLM |
| [UPR](upr.md) | LLM | Any LLM |

## Installation

API rerankers require optional dependencies:

```bash
pip install autorag-research[reranker]
```

## Base Class

All rerankers extend `BaseReranker`:

```python
from autorag_research.rerankers import BaseReranker, RerankResult


class MyReranker(BaseReranker):
    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        # Return sorted results by relevance
        pass

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        # Async version
        pass
```

## Methods

| Method | Description |
|--------|-------------|
| `rerank(query, docs, top_k)` | Single query reranking |
| `arerank(query, docs, top_k)` | Async single query |
| `rerank_documents(queries, docs_list, top_k)` | Batch reranking |
| `arerank_documents(queries, docs_list, top_k)` | Async batch |

## Usage with Injection

```python
from autorag_research.injection import load_reranker, with_reranker

# Load from config
reranker = load_reranker("cohere")

# Or use decorator
@with_reranker()
def my_func(reranker):
    return reranker.rerank("query", ["doc1", "doc2"])
```
