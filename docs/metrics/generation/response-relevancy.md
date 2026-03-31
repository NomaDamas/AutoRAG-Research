# Response Relevancy

RAGAS-style answer relevance metric without requiring the `ragas` package.

## Overview

| Field | Value |
|-------|-------|
| Type | Generation |
| Range | Typically [0, 1] (cosine math can produce [-1, 1]) |
| Higher is better | Yes |

## Description

This metric mirrors RAGAS `ResponseRelevancy` core logic:

1. Generate `strictness` synthetic questions from the model answer.
2. Compute cosine similarity between each synthetic question embedding and the original user query embedding.
3. Average similarities.
4. If all generations are flagged noncommittal, final score is forced to `0`.

The default instruction text is copied from RAGAS response-relevance prompt logic.

## Configuration

```yaml
_target_: autorag_research.evaluation.metrics.generation.ResponseRelevancyConfig
llm: openai-gpt5-mini
embedding_model: openai-large
strictness: 3
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| llm | str | `openai-gpt5-mini` | LLM config used to generate reverse questions |
| embedding_model | str | `openai-large` | Embedding model used for cosine similarity |
| strictness | int | 3 | Number of generated questions per answer |
| prompt_template | str | built-in RAGAS-style prompt | Prompt template for question generation |
