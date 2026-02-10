# UPR

Unsupervised Passage Reranker using question generation.

## Overview

| Field | Value |
|-------|-------|
| Type | LLM |
| Algorithm | Question generation |
| Paper | [Sachan et al., 2022](https://arxiv.org/abs/2204.07496) |

## How It Works

1. Generate a question from each passage
2. Compare generated questions to original query
3. Rank by similarity score

## Configuration

```yaml
_target_: autorag_research.rerankers.upr.UPRReranker
model_name: gpt-4o-mini
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| llm | BaseLanguageModel | required | LangChain LLM instance |
| use_logprobs | bool | False | Use log probabilities |

## Usage

```python
from langchain_openai import ChatOpenAI
from autorag_research.rerankers import UPRReranker

llm = ChatOpenAI(model="gpt-4o-mini")
reranker = UPRReranker(llm=llm)
results = reranker.rerank("query", documents, top_k=5)
```

## When to Use

Good for:

- Zero-shot reranking
- No training data available
- Research/experimentation

Consider RankGPT for:

- Better accuracy
- Direct comparison
