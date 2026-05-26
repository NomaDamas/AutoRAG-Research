# Hybrid Deep Searcher

Hybrid Deep Searcher (HDS) is an inference-only generation baseline that alternates sequential LLM planning with
parallel fan-out retrieval. At each turn, the LLM proposes subqueries, the configured retrieval pipeline gathers
evidence for those subqueries, and HDS merges/deduplicates evidence before either continuing or producing an answer.

```yaml
_target_: autorag_research.pipelines.generation.hybrid_deep_searcher.HybridDeepSearcherPipelineConfig
name: hybrid_deep_searcher
retrieval_pipeline_name: bm25
llm: openai-gpt4o-mini
max_turns: 3
max_parallel_queries: 4
retrieval_concurrency: 4
```

## Retrieval backend expectations

HDS sends LLM-generated subqueries directly to `retrieval_pipeline.retrieve(...)`. The retriever must therefore be
text-query-capable for strings that are not already rows in the query table. The shipped example uses `bm25` because it
can retrieve from generated text without an embedding model. If you switch to vector or hybrid retrieval, configure an
embedding-capable text path first.

## Concurrency

HDS has bounded logical fan-out. Effective retrieval pressure is approximately:

`generation max_concurrency x retrieval_concurrency`

Retriever-internal fan-out can multiply that further. Start with conservative values for larger datasets or shared
databases, then raise the limits only after observing database and provider capacity.

## Scope

This pipeline is meant for inference/evaluation comparisons with other generation baselines such as IRCoT and question
decomposition. It does not implement HDS-specific training infrastructure or paper-scale benchmark reproduction.
