# BasicRAG

Simple single-call RAG: retrieve once, build prompt, generate once.

## Overview

| Field | Value |
|-------|-------|
| Type | Generation |
| Algorithm | Retrieve + Generate |
| Modality | Text |

## How It Works

1. Retrieve top-k documents using configured retrieval pipeline
2. Build context from retrieved documents
3. Generate answer using LLM with prompt template

## Configuration

```yaml
_target_: autorag_research.pipelines.generation.basic_rag.BasicRAGPipelineConfig
name: basic_rag
retrieval_pipeline_name: bm25
llm: gpt-4o-mini
prompt_template: |
  Context:
  {context}

  Question: {question}

  Answer:
top_k: 5
batch_size: 100
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| name | str | required | Unique pipeline instance name |
| retrieval_pipeline_name | str | required | Name of retrieval pipeline to use |
| llm | str or BaseLLM | required | LLM instance or config name |
| prompt_template | str | default | Template with `{context}` and `{question}` |
| top_k | int | 10 | Documents to retrieve |
| batch_size | int | 100 | Queries per batch |

## Prompt Template Variables

| Variable | Description |
|----------|-------------|
| `{context}` | Retrieved document contents |
| `{question}` | Original query |

## When to Use

Good for:

- Simple Q&A tasks
- Baseline RAG implementation
- Quick prototyping

Consider advanced pipelines for:

- Multi-hop reasoning
- Iterative retrieval
- Complex answer synthesis
