# Vector Search

Dense retrieval based on vector similarity.

## Overview

| Field | Value |
|-------|-------|
| Type | Retrieval |
| Algorithm | Dense vector similarity |
| Modality | Text |
| Extension | [VectorChord](https://github.com/tensorchord/VectorChord) |

## How It Works

Ranks documents by computing similarity between query and document embeddings.

Supports two search modes:

1. **Single-vector**: Standard dense retrieval with one embedding per document
2. **Multi-vector**: Late interaction (MaxSim) with multiple token-level embeddings

Uses VectorChord PostgreSQL extension for efficient vector search.

## Score Metric

All scores are **relevance scores** in the range **[-1, 1]** where higher values indicate greater relevance.

### Single-Vector Mode (Cosine Similarity)

Uses cosine similarity between query and document embeddings:

$$
\text{score} = 1 - \text{cosine\_distance} = \cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}
$$

Where:

- $\mathbf{q}$ is the query embedding vector
- $\mathbf{d}$ is the document embedding vector

### Multi-Vector Mode (Normalized Late Interaction)

Uses MaxSim operation normalized by the number of query vectors:

$$
\text{score} = \frac{1}{n} \sum_{i=1}^{n} \max_{j} (\mathbf{q}_i \cdot \mathbf{d}_j)
$$

Where:

- $n$ is the number of query token vectors
- $\mathbf{q}_i$ is the $i$-th query token embedding
- $\mathbf{d}_j$ is the $j$-th document token embedding

This normalization ensures scores remain in [-1, 1] regardless of query length, making them comparable across different queries and compatible with single-vector scores for hybrid search.

## Configuration

```yaml
_target_: autorag_research.pipelines.retrieval.vector_search.VectorSearchPipelineConfig
name: vector_search
search_mode: single
top_k: 10
batch_size: 100
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| name | str | required | Unique pipeline instance name |
| search_mode | str | `single` | Embedding mode (`single` or `multi`) |
| top_k | int | 10 | Results per query |
| batch_size | int | 100 | Queries per batch |

## Search Modes

| Mode | Embedding Field | Algorithm | Use Case |
|------|-----------------|-----------|----------|
| single | `query.embedding` | Cosine similarity | Standard dense retrieval |
| multi | `query.embeddings` | MaxSim (late interaction) | Fine-grained token matching |

## Prerequisites

Queries must have pre-computed embeddings before running the pipeline:

```python
from autorag_research.data_ingestor import DataIngestor

ingestor = DataIngestor(session_factory)
ingestor.embed_all(embedding_model)  # Populates embedding/embeddings fields
```

## When to Use

Good for:

- Semantic similarity search
- Paraphrase and synonym matching
- Cross-lingual retrieval (with multilingual embeddings)
- Fine-grained matching (multi-vector mode)

Consider BM25 for:

- Exact keyword matching
- Low latency requirements
- No embedding model available
