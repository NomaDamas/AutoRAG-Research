# Common Query Templates

This document provides curated SQL query templates for common analysis tasks in AutoRAG-Research.

**IMPORTANT**: Always exclude vector/embedding columns: `embedding`, `embeddings`, `bm25_tokens`

**Parameterized Queries**: Templates use `:param_name` syntax for safe value substitution.
Pass parameters via `--params '{"param_name": "value"}'` to the query executor.

## Pipeline Performance Comparison

### Top pipelines by metric score

```sql
SELECT
    p.name AS pipeline_name,
    m.name AS metric_name,
    s.metric_result,
    s.created_at
FROM summary s
JOIN pipeline p ON s.pipeline_id = p.id
JOIN metric m ON s.metric_id = m.id
WHERE m.name = :metric_name
ORDER BY s.metric_result DESC
LIMIT :top_n;
```

**Parameters**: `:metric_name` (e.g., 'bleu', 'retrieval_precision'), `:top_n` (e.g., 10)

**Example output**:
```
pipeline_name       | metric_name | metric_result | created_at
--------------------|-------------|---------------|------------
hybrid_search_v2    | bleu        | 0.85         | 2025-01-15
naive_rag           | bleu        | 0.72         | 2025-01-14
```

---

### Average metrics across all queries

```sql
SELECT
    p.name AS pipeline_name,
    m.name AS metric_name,
    AVG(er.metric_result) AS avg_score,
    COUNT(*) AS query_count
FROM evaluation_result er
JOIN pipeline p ON er.pipeline_id = p.id
JOIN metric m ON er.metric_id = m.id
GROUP BY p.name, m.name
ORDER BY avg_score DESC;
```

**Description**: Shows average metric scores per pipeline across all queries.

---

### Pipeline ranking by multiple metrics

```sql
SELECT
    p.name AS pipeline_name,
    MAX(CASE WHEN m.name = 'bleu' THEN s.metric_result END) AS bleu,
    MAX(CASE WHEN m.name = 'retrieval_precision' THEN s.metric_result END) AS precision,
    MAX(CASE WHEN m.name = 'retrieval_recall' THEN s.metric_result END) AS recall
FROM summary s
JOIN pipeline p ON s.pipeline_id = p.id
JOIN metric m ON s.metric_id = m.id
WHERE m.name IN ('bleu', 'retrieval_precision', 'retrieval_recall')
GROUP BY p.name
ORDER BY bleu DESC, precision DESC;
```

**Description**: Compare pipelines across multiple metrics using pivot table.

---

## Per-Query Analysis

### Query-level metric breakdown

```sql
SELECT
    q.query AS question,
    p.name AS pipeline_name,
    m.name AS metric_name,
    er.metric_result,
    er.created_at
FROM evaluation_result er
JOIN query q ON er.query_id = q.id
JOIN pipeline p ON er.pipeline_id = p.id
JOIN metric m ON er.metric_id = m.id
WHERE q.id = :query_id
ORDER BY m.name, p.name;
```

**Parameters**: `:query_id` (specific query to analyze)

**Description**: Shows all metric scores for a specific query across all pipelines.

---

### Compare generated vs ground truth

```sql
SELECT
    q.query AS question,
    q.ground_truths,
    p.name AS pipeline_name,
    exe.generation_result,
    (exe.token_usage->>'total_tokens')::int AS total_tokens
FROM executor_result exe
JOIN query q ON exe.query_id = q.id
JOIN pipeline p ON exe.pipeline_id = p.id
WHERE q.id = :query_id
ORDER BY p.name;
```

**Parameters**: `:query_id`

**Description**: Compare generation outputs against ground truth answers.

**JSONB Note**: Use `token_usage->>'field'` to extract text, cast to `::int` for numbers.

---

### Find worst-performing queries

```sql
SELECT
    q.query AS question,
    p.name AS pipeline_name,
    m.name AS metric_name,
    er.metric_result,
    q.ground_truths
FROM evaluation_result er
JOIN query q ON er.query_id = q.id
JOIN pipeline p ON er.pipeline_id = p.id
JOIN metric m ON er.metric_id = m.id
WHERE m.name = :metric_name
  AND p.name = :pipeline_name
ORDER BY er.metric_result ASC
LIMIT :bottom_n;
```

**Parameters**: `:metric_name`, `:pipeline_name`, `:bottom_n`

**Description**: Identify queries with lowest scores for error analysis.

---

## Retrieval Results Analysis

### Top retrieved chunks for a query

```sql
SELECT
    q.query AS question,
    c.content AS chunk_text,
    crr.score AS retrieval_score,
    crr.rank,
    p.name AS pipeline_name
FROM chunk_retrieved_result crr
JOIN query q ON crr.query_id = q.id
JOIN chunk c ON crr.chunk_id = c.id
JOIN pipeline p ON crr.pipeline_id = p.id
WHERE q.id = :query_id
  AND p.name = :pipeline_name
ORDER BY crr.rank ASC
LIMIT :top_k;
```

**Parameters**: `:query_id`, `:pipeline_name`, `:top_k`

**Description**: Shows actual retrieved chunks with scores and rankings.

---

### Retrieval score distribution

```sql
SELECT
    p.name AS pipeline_name,
    AVG(crr.score) AS avg_score,
    MIN(crr.score) AS min_score,
    MAX(crr.score) AS max_score,
    COUNT(*) AS total_retrievals
FROM chunk_retrieved_result crr
JOIN pipeline p ON crr.pipeline_id = p.id
GROUP BY p.name
ORDER BY avg_score DESC;
```

**Description**: Analyze retrieval score statistics per pipeline.

---

### Ground truth comparison (retrieval)

```sql
SELECT
    q.query AS question,
    p.name AS pipeline_name,
    COUNT(DISTINCT rr.chunk_id) AS total_relevant_chunks,
    COUNT(DISTINCT CASE WHEN crr.chunk_id IS NOT NULL THEN rr.chunk_id END) AS retrieved_relevant_chunks,
    CAST(COUNT(DISTINCT CASE WHEN crr.chunk_id IS NOT NULL THEN rr.chunk_id END) AS FLOAT) /
        NULLIF(COUNT(DISTINCT rr.chunk_id), 0) AS recall
FROM query q
JOIN retrieval_relation rr ON q.id = rr.query_id
JOIN pipeline p ON p.id = :pipeline_id
LEFT JOIN chunk_retrieved_result crr ON
    rr.query_id = crr.query_id
    AND rr.chunk_id = crr.chunk_id
    AND crr.pipeline_id = p.id
WHERE q.id = :query_id
GROUP BY q.query, p.name;
```

**Parameters**: `:query_id`, `:pipeline_id`

**Description**: Calculate recall by comparing retrieved chunks against ground truth.

---

## Token Usage Analysis

### Token usage by pipeline

```sql
SELECT
    p.name AS pipeline_name,
    COUNT(*) AS query_count,
    SUM((exe.token_usage->>'prompt_tokens')::int) AS total_prompt_tokens,
    SUM((exe.token_usage->>'completion_tokens')::int) AS total_completion_tokens,
    SUM((exe.token_usage->>'total_tokens')::int) AS total_tokens,
    AVG((exe.token_usage->>'total_tokens')::int) AS avg_tokens_per_query
FROM executor_result exe
JOIN pipeline p ON exe.pipeline_id = p.id
WHERE exe.token_usage IS NOT NULL
GROUP BY p.name
ORDER BY total_tokens DESC;
```

**Description**: Aggregate token usage statistics per pipeline.

**JSONB Extraction**: Cast to `::int` after extracting with `->>`.

---

### Most expensive queries

```sql
SELECT
    q.query AS question,
    p.name AS pipeline_name,
    (exe.token_usage->>'total_tokens')::int AS total_tokens,
    (exe.token_usage->>'prompt_tokens')::int AS prompt_tokens,
    (exe.token_usage->>'completion_tokens')::int AS completion_tokens,
    exe.execution_time
FROM executor_result exe
JOIN query q ON exe.query_id = q.id
JOIN pipeline p ON exe.pipeline_id = p.id
WHERE exe.token_usage IS NOT NULL
ORDER BY (exe.token_usage->>'total_tokens')::int DESC
LIMIT :top_n;
```

**Parameters**: `:top_n`

**Description**: Find queries with highest token consumption.

---

### Token usage over time

```sql
SELECT
    DATE(exe.created_at) AS date,
    p.name AS pipeline_name,
    SUM((exe.token_usage->>'total_tokens')::int) AS daily_tokens,
    COUNT(*) AS query_count
FROM executor_result exe
JOIN pipeline p ON exe.pipeline_id = p.id
WHERE exe.token_usage IS NOT NULL
  AND exe.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(exe.created_at), p.name
ORDER BY date DESC, daily_tokens DESC;
```

**Description**: Track token usage trends over last 30 days.

---

## Execution Performance

### Slowest queries by execution time

```sql
SELECT
    q.query AS question,
    p.name AS pipeline_name,
    exe.execution_time,
    (exe.token_usage->>'total_tokens')::int AS total_tokens,
    exe.created_at
FROM executor_result exe
JOIN query q ON exe.query_id = q.id
JOIN pipeline p ON exe.pipeline_id = p.id
ORDER BY exe.execution_time DESC
LIMIT :top_n;
```

**Parameters**: `:top_n`

**Description**: Identify performance bottlenecks.

---

### Average execution time by pipeline

```sql
SELECT
    p.name AS pipeline_name,
    COUNT(*) AS query_count,
    AVG(exe.execution_time) AS avg_execution_time,
    MIN(exe.execution_time) AS min_execution_time,
    MAX(exe.execution_time) AS max_execution_time
FROM executor_result exe
JOIN pipeline p ON exe.pipeline_id = p.id
GROUP BY p.name
ORDER BY avg_execution_time DESC;
```

**Description**: Compare pipeline performance across all queries.

---

## JSONB Extraction Patterns

### Extract nested token usage details

```sql
SELECT
    p.name AS pipeline_name,
    q.query AS question,
    exe.token_usage->>'prompt_tokens' AS prompt_tokens_text,
    (exe.token_usage->>'prompt_tokens')::int AS prompt_tokens_int,
    exe.token_usage->'embedding_tokens' AS embedding_tokens_json,
    exe.token_usage AS full_token_usage
FROM executor_result exe
JOIN pipeline p ON exe.pipeline_id = p.id
JOIN query q ON exe.query_id = q.id
WHERE exe.token_usage IS NOT NULL
LIMIT 5;
```

**JSONB Operators**:
- `->` extracts as JSON (preserves type)
- `->>` extracts as text
- Cast with `::int`, `::float`, etc.

---

### Parse pipeline config JSONB

```sql
SELECT
    name AS pipeline_name,
    config->>'type' AS pipeline_type,
    config->>'model' AS model_name,
    (config->>'top_k')::int AS top_k,
    config AS full_config
FROM pipeline
WHERE config IS NOT NULL
LIMIT 10;
```

**Description**: Extract structured data from pipeline configuration.

---

## Complex Multi-Table JOINs

### Full pipeline execution report

```sql
SELECT
    p.name AS pipeline_name,
    q.query AS question,
    exe.generation_result,
    m.name AS metric_name,
    er.metric_result AS score,
    (exe.token_usage->>'total_tokens')::int AS tokens,
    exe.execution_time
FROM executor_result exe
JOIN query q ON exe.query_id = q.id
JOIN pipeline p ON exe.pipeline_id = p.id
LEFT JOIN evaluation_result er ON
    er.query_id = exe.query_id
    AND er.pipeline_id = exe.pipeline_id
LEFT JOIN metric m ON er.metric_id = m.id
WHERE p.name = :pipeline_name
ORDER BY q.query, m.name;
```

**Parameters**: `:pipeline_name`

**Description**: Comprehensive view of pipeline execution with all metrics.

---

### Retrieval + Generation + Evaluation combined

```sql
SELECT
    q.query AS question,
    p.name AS pipeline_name,
    STRING_AGG(DISTINCT c.content, ' | ') AS retrieved_chunks,
    exe.generation_result,
    MAX(CASE WHEN m.name = 'bleu' THEN er.metric_result END) AS bleu,
    MAX(CASE WHEN m.name = 'rouge' THEN er.metric_result END) AS rouge
FROM query q
JOIN pipeline p ON p.id = :pipeline_id
LEFT JOIN chunk_retrieved_result crr ON
    crr.query_id = q.id
    AND crr.pipeline_id = p.id
LEFT JOIN chunk c ON crr.chunk_id = c.id
LEFT JOIN executor_result exe ON
    exe.query_id = q.id
    AND exe.pipeline_id = p.id
LEFT JOIN evaluation_result er ON
    er.query_id = q.id
    AND er.pipeline_id = p.id
LEFT JOIN metric m ON er.metric_id = m.id
WHERE q.id = :query_id
GROUP BY q.query, p.name, exe.generation_result;
```

**Parameters**: `:query_id`, `:pipeline_id`

**Description**: Complete RAG pipeline trace for a single query.

---

## Metadata Queries

### List all pipelines

```sql
SELECT
    id,
    name,
    pipeline_type,
    created_at,
    config->>'model' AS model,
    config->>'type' AS config_type
FROM pipeline
ORDER BY created_at DESC;
```

---

### List all metrics

```sql
SELECT
    id,
    name,
    metric_type,
    created_at
FROM metric
ORDER BY metric_type, name;
```

**Note**: `metric_type` is either 'retrieval' or 'generation'.

---

### Count queries by dataset

```sql
SELECT
    dataset_name,
    COUNT(*) AS query_count,
    COUNT(DISTINCT ground_truths) AS unique_ground_truths
FROM query
GROUP BY dataset_name
ORDER BY query_count DESC;
```

---

## Notes

- **Always exclude**: `embedding`, `embeddings`, `bm25_tokens` columns
- **JSONB extraction**: Use `->>` for text, cast to type for numbers
- **Performance**: Add `LIMIT` clauses to prevent large result sets
- **Parameterization**: Use `:param_name` with `--params '{"param_name": "value"}'` for safe value substitution
- **NULL handling**: Use `NULLIF()` and `COALESCE()` for division and defaults
- **Example with params**: `python query_executor.py -q "SELECT * FROM pipeline WHERE name = :name" -p '{"name": "naive_rag"}'`
