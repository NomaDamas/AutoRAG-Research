# RankGPT

LLM-based listwise reranking.

## Overview

| Field | Value |
|-------|-------|
| Type | LLM |
| Algorithm | Listwise ranking |
| Paper | [Sun et al., 2023](https://arxiv.org/abs/2304.09542) |

## How It Works

1. Presents all documents to LLM with query
2. LLM outputs ranking: `[1] > [3] > [2]`
3. Uses sliding window for large document sets

## Configuration

```yaml
_target_: autorag_research.rerankers.rankgpt.RankGPTReranker
model_name: gpt-4o-mini
max_passages_per_call: 20
window_size: 10
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| llm | BaseLanguageModel | required | LangChain LLM instance |
| max_passages_per_call | int | 20 | Max docs per LLM call |
| window_size | int | 10 | Sliding window step |

## Usage

```python
from langchain_openai import ChatOpenAI
from autorag_research.rerankers import RankGPTReranker

llm = ChatOpenAI(model="gpt-4o-mini")
reranker = RankGPTReranker(llm=llm)
results = reranker.rerank("query", documents, top_k=5)
```

## When to Use

Good for:

- High-quality ranking needed
- Small document sets (<100)
- When API rerankers unavailable

Consider API rerankers for:

- Large scale (cost)
- Low latency requirements
