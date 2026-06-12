# GQR Hybrid Retrieval

Guided Query Refinement (GQR) is a hybrid retrieval wrapper that uses one primary retriever for the first-pass candidate scores and one complementary retriever as fixed guidance. At every optimization step, it recomputes the current primary distribution, blends that live distribution with the complementary distribution, and updates the primary query representation so the final ranking moves toward consensus.

## When to use

Use GQR when you want dense-first retrieval to benefit from a complementary lexical or sparse signal without adding a separate reranker dependency.

Typical pairing:

- **Primary**: `vector_search`
- **Complementary**: `bm25`
- **Candidate pool**: `union` for true hybrid coverage, or `primary` for primary-bounded reranking

## Configuration

```yaml
_target_: autorag_research.pipelines.retrieval.gqr_hybrid.GQRHybridRetrievalPipelineConfig
name: gqr_hybrid
primary_retrieval_pipeline_name: vector_search
complementary_retrieval_pipeline_name: bm25
fetch_k_multiplier: 2
n_steps: 25
learning_rate: 0.1
temperature: 1.0
mixture_alpha: 0.5
candidate_pool_mode: union
scorer_mode: auto
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `primary_retrieval_pipeline_name` | string | required | First-pass retriever that provides the primary scores and, when available, the query embedding. |
| `complementary_retrieval_pipeline_name` | string | required | Guidance retriever used to build the complementary target distribution. |
| `fetch_k_multiplier` | int | `2` | Fetches `top_k * fetch_k_multiplier` from each child before reranking. |
| `n_steps` | int | `25` | Optimization steps for embedding-level or score-space refinement. |
| `learning_rate` | float | `0.1` | Step size for refinement. |
| `temperature` | float | `1.0` | Softmax temperature used for raw primary and complementary score distributions. |
| `mixture_alpha` | float | `0.5` | Guidance weight: `0.0` follows primary scores, `1.0` follows complementary scores. |
| `candidate_pool_mode` | `union` or `primary` | `union` | `union` includes candidates from both retrievers; `primary` keeps only primary candidates. |
| `scorer_mode` | `auto`, `single`, or `multi` | `auto` | Primary scorer used for embedding-level refinement. `auto` uses multi-vector MaxSim when the primary pipeline exposes `search_mode: multi`; otherwise it uses single-vector cosine. |

## Scoring contract

GQR result scores are **ranking scores**: higher is better within the returned list, but the numeric magnitude is not calibrated across execution modes or against other pipelines. Distribution construction uses the child retrievers' raw native scores plus a conservative missing-candidate floor for union candidates absent from one child result set; scores are not z-score normalized before softmax.

GQR has three scoring paths:

1. **Single-vector embedding-level refinement** runs when `scorer_mode` resolves to `single`, a query embedding exists, and every selected candidate has a chunk embedding. Scores are dense cosine values from the refined query embedding, matching the Eq. 3 dense primary scorer scope.
2. **Multi-vector embedding-level refinement** runs when `scorer_mode` resolves to `multi`, a stored query multi-vector embedding exists, and every selected candidate has chunk multi-vector embeddings. Scores are normalized MaxSim values, `(1 / n_query_vectors) * sum_i max_k z_i · d_jk`, matching the Eq. 4 late-interaction primary scorer scope and the service's `multi` score convention.
3. **Score-space fallback** runs when query embeddings are unavailable or when any selected candidate lacks the required embedding representation. In partial-embedding pools, GQR falls back for the entire pool so one result list does not mix embedding scores with raw primary-score values. This is the documented ALLOWED-2 degraded path that preserves the per-step consensus objective without a full primary scorer API.
For both paths, the complementary distribution is fixed for the candidate pool, while the primary distribution and consensus target are recomputed every optimization step.

Use ordering and retrieval metrics for evaluation. Do not compare raw GQR score magnitudes across embedding-backed and fallback-backed runs.

## Text-query behavior

For ad-hoc text queries, GQR first asks its child retrievers for text results. If the primary vector retriever cannot embed raw text, GQR falls back to text-capable complementary results and then applies score-space refinement. Multi-vector by-text queries also use score-space refinement because AutoRAG-Research has no standard text-to-query-matrix interface across embedding models.

For ID/batch retrieval, stored query and chunk single-vector or multi-vector embeddings enable the embedding-level path when all selected candidates have the representation required by the resolved scorer mode.

## Dependency loading note

GQR currently resolves `primary_retrieval_pipeline_name` and `complementary_retrieval_pipeline_name` through the same child-loader helper used by hybrid retrieval. This keeps the runnable YAML config compatible with existing retrieval pipeline loading while preserving explicit primary/complementary roles in the GQR config.
