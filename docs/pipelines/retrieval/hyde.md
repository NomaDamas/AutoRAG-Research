# HyDE (Hypothetical Document Embeddings)

Dense retrieval using LLM-generated hypothetical documents.

## Overview

| Field | Value |
|-------|-------|
| Type | Retrieval |
| Algorithm | Dense (vector similarity with hypothetical documents) |
| Modality | Text |
| Paper | [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) |

## How It Works

1. Receives a query
2. Uses LLM to generate a hypothetical document that would answer the query
3. Embeds the hypothetical document (not the original query)
4. Performs vector similarity search with the hypothetical embedding

This bridges the semantic gap between queries and documents by generating document-like text.

## Configuration

```yaml
_target_: autorag_research.pipelines.retrieval.hyde.HyDEPipelineConfig
name: hyde_gpt4
llm: openai-gpt4
embedding: openai-small
prompt_template: |
  Please write a passage to answer the question.
  Question: {query}
  Passage:
top_k: 10
batch_size: 100
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| name | str | required | Unique pipeline instance name |
| llm | str | required | LLM config name (from configs/llm/) |
| embedding | str | required | Embedding config name (from configs/embedding/) |
| prompt_template | str | see below | Template with {query} placeholder |
| top_k | int | 10 | Results per query |
| batch_size | int | 100 | Queries per batch |

**Default prompt template:**
```
Please write a passage to answer the question.
Question: {query}
Passage:
```

## Custom Prompts

The paper recommends domain-specific prompts. Examples:

**Web search (DL19/20, DBPedia):**
```yaml
prompt_template: |
  Please write a passage to answer the question
  Question: {query}
  Passage:
```

**SciFact:**
```yaml
prompt_template: |
  Please write a scientific paper passage to support/refute the claim
  Claim: {query}
  Passage:
```

**TREC-COVID:**
```yaml
prompt_template: |
  Please write a scientific paper passage to answer the question
  Question: {query}
  Passage:
```

**FiQA:**
```yaml
prompt_template: |
  Please write a financial article passage to answer the question
  Question: {query}
  Passage:
```

**TREC-NEWS:**
```yaml
prompt_template: |
  Please write a news passage about the topic.
  Topic: {query}
  Passage:
```

**ArguAna:**
```yaml
prompt_template: |
  Please write a counter argument for the passage
  Passage: {query}
  Counter Argument:
```

**Mr.TyDi (Multilingual):**
```yaml
prompt_template: |
  Please write a passage in Korean to answer the question in detail.
  Question: {query}
  Passage:
```

## Usage

### Python API

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from autorag_research.orm.connection import DBConnection
from autorag_research.pipelines.retrieval.hyde import HyDERetrievalPipeline

db = DBConnection.from_config()
session_factory = db.get_session_factory()

pipeline = HyDERetrievalPipeline(
    session_factory=session_factory,
    name="hyde_gpt4",
    llm=ChatOpenAI(model="gpt-4"),
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    # Optional: custom prompt for domain-specific documents
    prompt_template="Write a Wikipedia passage about: {query}\n\nPassage:",
)

# Single query
results = pipeline.retrieve("What is machine learning?", top_k=10)

# Batch processing
stats = pipeline.run(top_k=10)
```

### With Config

```python
from autorag_research.pipelines.retrieval.hyde import HyDEPipelineConfig

config = HyDEPipelineConfig(
    name="hyde_gpt4",
    llm="openai-gpt4",      # Auto-converted to LLM instance
    embedding="openai-small", # Auto-converted to Embeddings instance
    top_k=10,
)
```

## When to Use

Good for:

- Zero-shot retrieval (no labeled data needed)
- Bridging query-document semantic gap
- Complex questions requiring reasoning

Consider other methods when:

- Low latency is critical (LLM adds latency)
- Embedding model cost is a concern
- Pre-computed query embeddings are available
