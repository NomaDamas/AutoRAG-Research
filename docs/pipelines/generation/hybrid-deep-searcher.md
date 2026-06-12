# Hybrid Deep Searcher

Hybrid Deep Searcher (HDS) is an inference-only generation baseline that follows the paper-style loop: the model emits
`<think>...</think>` reasoning plus parallel query blocks, the environment executes those queries, and query-labelled
search results are appended to a rolling interaction log before the next reasoning turn. Final answers are read from
`\boxed{...}`; `<answer>...</answer>` is accepted for provider/model compatibility.

```yaml
_target_: autorag_research.pipelines.generation.hybrid_deep_searcher.HybridDeepSearcherPipelineConfig
name: hybrid_deep_searcher
retrieval_pipeline_name: bm25
llm: openai-gpt4o-mini
max_turns: 3
max_parallel_queries: 4
max_search_calls: 8
retrieval_concurrency: 4
```

## Protocol

HDS parses search actions from paper tokens:

```text
<|begin search queries|> q1; q2
q3 <|end search queries|>
```

Queries are split on semicolons and newlines, list prefixes are stripped, duplicates are removed, and fan-out is capped
by `max_parallel_queries`. Each turn also respects `max_search_calls`: only the first remaining-budget queries are
executed, and budget exhaustion forces finalization with the final prompt when `fallback_to_final_prompt` is enabled.

Retrieved contents are returned to the model as one query-labelled environment block:

```text
<|begin search results|>
q1: joined top contents for q1
q2: joined top contents for q2
<|end search results|>
```

`evidence_budget` caps the number of per-query result contents serialized into each block. Retrieved document IDs are
deduplicated globally for `retrieved_chunk_ids` metadata, but prompt context preserves the query-to-result pairing.

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
decomposition. It does not implement HDS-specific training infrastructure, web-search summarization, or paper-scale
benchmark reproduction.
