---
name: autorag-query
description: |
  Query AutoRAG-Research pipeline results using natural language. Converts questions to SQL,
  executes safely (SELECT-only), returns formatted results. Auto-detects DB connection from
  configs/db.yaml or env vars. Use for pipeline comparison, metrics analysis, token usage.
allowed-tools:
  - Bash
  - Read
---

# AutoRAG-Query: Text2SQL Agent Skill

Query AutoRAG pipeline results with natural language. Converts to SQL, executes safely, returns tables/JSON/CSV.

## Quick Example

**User**: "Which pipeline has the best BLEU score?"

**Agent**:
1. Read `references/schema.sql` (understand tables)
2. Generate SQL:
   ```sql
   SELECT p.name, s.metric_result
   FROM summary s
   JOIN pipeline p ON s.pipeline_id = p.id
   JOIN metric m ON s.metric_id = m.id
   WHERE m.name = 'bleu'
   ORDER BY s.metric_result DESC LIMIT 1;
   ```
3. Execute: `uv run python .agents/skills/autorag-query/scripts/query_executor.py --query "..."`
4. Present: "**hybrid_search_v2** has best BLEU: **0.85**"

## Workflow

1. **Parse intent**: What data? (metrics/pipelines/queries) What operation? (rank/aggregate/filter)
2. **Load schema**: Read `references/schema.sql` - key tables:
   - `summary`: Aggregated pipeline metrics (best for rankings)
   - `evaluation_result`: Per-query scores (detailed analysis)
   - `executor_result`: Generation outputs with `token_usage` JSONB
   - `chunk_retrieved_result`: Retrieval scores/ranks
3. **Generate SQL** following rules:
   - ✅ SELECT-only, ⛔ Never: INSERT/UPDATE/DELETE/DROP/CREATE
   - ⛔ **Exclude vector columns**: `embedding`, `embeddings`, `bm25_tokens` (cause type errors)
   - Add `LIMIT 100` if not specified
   - Use JOINs: `query_id → query.id`, `pipeline_id → pipeline.id`, `metric_id → metric.id`
   - JSONB: `token_usage->>'field'` (text) or `(token_usage->>'field')::int` (cast)
4. **Execute**: `uv run python .agents/skills/autorag-query/scripts/query_executor.py --query "..." [--format json|csv|table]`
5. **Present**: Summarize findings, show table, highlight insights

## Key Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `pipeline` | Pipeline definitions | `id`, `name`, `pipeline_type` |
| `metric` | Metric definitions | `id`, `name`, `metric_type` (retrieval/generation) |
| `query` | Search queries | `id`, `query`, `ground_truths`, `dataset_name` |
| `executor_result` | Generation outputs | `query_id`, `pipeline_id`, `generation_result`, `token_usage` (JSONB), `execution_time` |
| `evaluation_result` | Per-query scores | `query_id`, `pipeline_id`, `metric_id`, `metric_result` |
| `summary` | Aggregated metrics | `pipeline_id`, `metric_id`, `metric_result` |
| `chunk_retrieved_result` | Retrieval outputs | `query_id`, `pipeline_id`, `chunk_id`, `score`, `rank` |

**Relationships**: `query_id → query.id`, `pipeline_id → pipeline.id`, `metric_id → metric.id`, `chunk_id → chunk.id`

## Common Queries

See `references/common-queries.md` for 20+ templates.

**Pipeline ranking**:
```sql
SELECT p.name, s.metric_result
FROM summary s
JOIN pipeline p ON s.pipeline_id = p.id
JOIN metric m ON s.metric_id = m.id
WHERE m.name = 'bleu'
ORDER BY s.metric_result DESC;
```

**Token usage**:
```sql
SELECT p.name,
       SUM((exe.token_usage->>'total_tokens')::int) AS total_tokens,
       AVG((exe.token_usage->>'total_tokens')::int) AS avg_per_query
FROM executor_result exe
JOIN pipeline p ON exe.pipeline_id = p.id
WHERE exe.token_usage IS NOT NULL
GROUP BY p.name
ORDER BY total_tokens DESC;
```

**Retrieval results**:
```sql
SELECT c.content, crr.score, crr.rank
FROM chunk_retrieved_result crr
JOIN chunk c ON crr.chunk_id = c.id
WHERE crr.query_id = :query_id AND crr.pipeline_id = :pipeline_id
ORDER BY crr.rank LIMIT 10;
```

## JSONB Extraction

**executor_result.token_usage**:
```json
{"prompt_tokens": 150, "completion_tokens": 50, "total_tokens": 200}
```

**Extract**:
- Text: `token_usage->>'prompt_tokens'` → `"150"`
- Integer: `(token_usage->>'total_tokens')::int` → `200`
- JSON: `token_usage->'embedding_tokens'` → preserves type

**pipeline.config**: `config->>'model'` → `"gpt-4"`

## Critical Rules

1. ⛔ **Always exclude**: `embedding`, `embeddings`, `bm25_tokens` columns (cause type errors)
2. ✅ **SELECT-only**: Script validates and rejects DDL/DML
3. 📏 **Add LIMIT**: Prevent large result sets
4. 🔗 **Use JOINs**: Connect via foreign keys
5. ⚡ **Timeout**: 10s default (add WHERE filters if slow)

## Script Usage

```bash
uv run python .agents/skills/autorag-query/scripts/query_executor.py \
  --query "SELECT ..." \
  --format table|json|csv \
  --timeout 10 \
  --limit 10000 \
  --database autorag_research  # optional
```

**Connection**: Auto-loads from `configs/db.yaml` or `POSTGRES_*` env vars using `DBConnection` class.

**Output formats**:
- `table`: ASCII table (default)
- `json`: JSON array
- `csv`: CSV with headers

**Row count**: Printed to stderr: `(N rows)`

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| "Forbidden keyword" | Non-SELECT query | Use SELECT-only |
| "Vector type error" | Selected vector columns | Exclude `embedding`, `embeddings`, `bm25_tokens` from SELECT |
| "Query timeout" | Query too slow | Add WHERE/LIMIT |
| "Connection failed" | Missing credentials | Check `configs/db.yaml` or set env vars |

## Advanced: Window Functions & Pivots

**Ranking**:
```sql
SELECT p.name, m.name, s.metric_result,
       RANK() OVER (PARTITION BY m.name ORDER BY s.metric_result DESC) AS rank
FROM summary s
JOIN pipeline p ON s.pipeline_id = p.id
JOIN metric m ON s.metric_id = m.id;
```

**Pivot**:
```sql
SELECT p.name,
       MAX(CASE WHEN m.name = 'bleu' THEN s.metric_result END) AS bleu,
       MAX(CASE WHEN m.name = 'rouge' THEN s.metric_result END) AS rouge
FROM summary s
JOIN pipeline p ON s.pipeline_id = p.id
JOIN metric m ON s.metric_id = m.id
GROUP BY p.name;
```

## References

- **Schema**: `references/schema.sql` - full DB schema with comments
- **Templates**: `references/common-queries.md` - 20+ query examples
- **Executor**: `scripts/query_executor.py` - safe SQL execution script

**Installation**: Works from `.agents/skills/autorag-query/` (auto-detected by agents).
