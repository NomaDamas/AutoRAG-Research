# Hybrid Retrieval

Combine multiple retrieval pipelines by its relevance scores. There are two types of fusion strategies: RRF and CC (Convex Combination; Weighted Sum)

## Overview

| Field | Value |
|-------|-------|
| Type | Retrieval |
| Methods | RRF, Convex Combination |
| Pipelines | Any 2 retrieval pipelines |

## Methods

### Reciprocal Rank Fusion (RRF)

Combines results based on rank positions, ignoring raw scores.

**Formula:** `RRF(d) = sum(1/(k + rank_i(d)))`

**Advantages:**

- Score-scale independent
- Robust to different retrieval methods
- No normalization needed

### Convex Combination (CC)

Combines normalized scores with configurable weights.

**Formula:** `combined = weight * norm(score_1) + (1-weight) * norm(score_2)`

**Normalization Methods:**

| Method | Description |
|--------|-------------|
| `mm` | Min-max scaling to [0, 1] using actual min/max |
| `tmm` | Theoretical min with actual max (e.g., BM25 min=0, cosine min=-1) |
| `z` | Z-score standardization |
| `dbsf` | 3-sigma distribution-based |

## Configuration

### RRF Pipeline

```yaml
_target_: autorag_research.pipelines.retrieval.hybrid.HybridRRFRetrievalPipelineConfig
name: hybrid_rrf
retrieval_pipeline_1_name: vector_search
retrieval_pipeline_2_name: bm25
rrf_k: 60
top_k: 10
```

### CC Pipeline

```yaml
_target_: autorag_research.pipelines.retrieval.hybrid.HybridCCRetrievalPipelineConfig
name: hybrid_cc
retrieval_pipeline_1_name: vector_search
retrieval_pipeline_2_name: bm25
weight: 0.5
normalize_method: mm
top_k: 10
```

## Options

### Common Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| name | str | required | Unique pipeline name |
| retrieval_pipeline_1_name | str | required | First pipeline name |
| retrieval_pipeline_2_name | str | required | Second pipeline name |
| top_k | int | 10 | Results per query |
| batch_size | int | 100 | Queries per batch |

### RRF Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| rrf_k | int | 60 | RRF constant (higher = more top-rank emphasis) |

### CC Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| weight | float | 0.5 | Weight for pipeline_1 (0=full pipeline_2, 1=full pipeline_1) |
| normalize_method | str | mm | Normalization: mm, tmm, z, dbsf |
| pipeline_1_min | float | None | Theoretical min score for tmm (pipeline_1) |
| pipeline_2_min | float | None | Theoretical min score for tmm (pipeline_2) |

## Usage

### Programmatic

```python
from autorag_research.pipelines.retrieval import (
    HybridRRFRetrievalPipeline,
    VectorSearchRetrievalPipeline,
    BM25RetrievalPipeline,
)

# Create sub-pipelines
vector = VectorSearchRetrievalPipeline(session_factory, "vector")
bm25 = BM25RetrievalPipeline(session_factory, "bm25")

# Create hybrid with instantiated pipelines
hybrid = HybridRRFRetrievalPipeline(
    session_factory=session_factory,
    name="hybrid_rrf",
    retrieval_pipeline_1=vector,
    retrieval_pipeline_2=bm25,
    rrf_k=60,
)

# Or create with pipeline names (auto-loaded from YAML)
hybrid = HybridRRFRetrievalPipeline(
    session_factory=session_factory,
    name="hybrid_rrf",
    retrieval_pipeline_1="vector_search",  # Loads from configs/
    retrieval_pipeline_2="bm25",
    rrf_k=60,
)

results = hybrid.retrieve("What is machine learning?", top_k=10)
```

### CLI

```bash
autorag run --pipeline hybrid_rrf --top-k 10
```

## When to Use

**Use RRF when:**

- Combining different retrieval paradigms (dense + sparse)
- Score scales differ significantly
- Want robust, parameter-free fusion

**Use CC when:**

- Fine-tuning weight between pipelines
- Score distributions are known
- Want explicit control over fusion balance

## References

- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) - Cormack et al., 2009
- [Hybrid Search](https://arxiv.org/abs/2210.11934) - Survey on hybrid retrieval methods

## Citation

```bibtex
@inproceedings{cormack2009reciprocal,
  title={Reciprocal rank fusion outperforms condorcet and individual rank learning methods},
  author={Cormack, Gordon V and Clarke, Charles LA and Buettcher, Stefan},
  booktitle={Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval},
  pages={758--759},
  year={2009}
}

@article{bruch2023analysis,
  title={An analysis of fusion functions for hybrid retrieval},
  author={Bruch, Sebastian and Gai, Siyu and Ingber, Amir},
  journal={ACM Transactions on Information Systems},
  volume={42},
  number={1},
  pages={1--35},
  year={2023},
  publisher={ACM New York, NY}
}
```
