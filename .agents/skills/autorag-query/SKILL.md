---
name: autorag-query
description: |
  Query AutoRAG-Research pipeline results, metrics, and evaluation data using natural language.
  Converts text queries to SQL, executes safely (SELECT-only), and returns formatted results.
  Use for: comparing pipeline performance, analyzing metrics across queries, exploring retrieval
  results, token usage analysis. Database connection auto-detected from configs/db.yaml or env vars.
allowed-tools:
  - Bash
  - Read
  - Write
---

# AutoRAG-Query: Text2SQL Agent Skill

## Overview

AutoRAG-Query enables AI agents to query AutoRAG-Research pipeline results using natural language. Instead of requiring users to navigate a GUI, agents can directly query the PostgreSQL database to answer questions about:

- **Pipeline performance**: Which pipeline has the best BLEU score?
- **Metric analysis**: Compare precision/recall across retrieval pipelines
- **Retrieval results**: What chunks were retrieved for a specific query?
- **Token usage**: Which pipeline is most cost-efficient?
- **Execution performance**: Which queries are slowest?

**Key Features**:
- 🔒 **Safe**: SELECT-only queries with validation
- 🔌 **Auto-connection**: Detects credentials from `configs/db.yaml` or environment variables
- 📊 **Formatted output**: Tables, JSON, or CSV
- ⚡ **Timeout protection**: 10-second default timeout
- 🧠 **Schema-aware**: Full database schema reference included

## Quick Start Example

**User asks**: "Which pipeline has the best BLEU score?"

**Agent workflow**:

1. **Parse intent**: User wants top pipeline ranked by BLEU metric
2. **Load schema**: Read `references/schema.sql` to understand tables
3. **Identify tables**: `summary` (aggregated metrics), `pipeline`, `metric`
4. **Generate SQL**:
   ```sql
   SELECT p.name, s.metric_result
   FROM summary s
   JOIN pipeline p ON s.pipeline_id = p.id
   JOIN metric m ON s.metric_id = m.id
   WHERE m.name = 'bleu'
   ORDER BY s.metric_result DESC
   LIMIT 1;
   ```
5. **Execute**:
   ```bash
   python .agents/skills/autorag-query/scripts/query_executor.py \
     --query "SELECT p.name, s.metric_result FROM summary s JOIN pipeline p ON s.pipeline_id = p.id JOIN metric m ON s.metric_id = m.id WHERE m.name = 'bleu' ORDER BY s.metric_result DESC LIMIT 1"
   ```
6. **Format result**:
   ```
   name              | metric_result
   ------------------|---------------
   hybrid_search_v2  | 0.85
   ```

**Agent response**: "The pipeline with the best BLEU score is **hybrid_search_v2** with a score of **0.85**."

---

## Text2SQL Workflow

Follow this systematic workflow to convert natural language to SQL:

### Step 1: Parse User Intent

Identify the key components:
- **What data?** (metrics, pipelines, queries, retrieval results, token usage)
- **What operation?** (ranking, aggregation, comparison, filtering)
- **What conditions?** (specific pipeline, metric, query ID, date range)

**Examples**:
- "Which pipeline..." → Ranking by metric
- "Show me token usage..." → Aggregation by pipeline
- "Compare generation results..." → Side-by-side comparison
- "Find queries with..." → Filtering with conditions

### Step 2: Load Database Schema

**MANDATORY**: Read `references/schema.sql` before generating SQL.

Key tables you'll use frequently:
- `pipeline`: Pipeline definitions (id, name, pipeline_type, config)
- `metric`: Metric definitions (id, name, metric_type: retrieval/generation)
- `query`: Search queries with ground truth (id, query, ground_truths, dataset_name)
- `executor_result`: Generation outputs (query_id, pipeline_id, generation_result, token_usage, execution_time)
- `evaluation_result`: Per-query metric scores (query_id, pipeline_id, metric_id, metric_result)
- `summary`: Aggregated pipeline metrics (pipeline_id, metric_id, metric_result)
- `chunk_retrieved_result`: Retrieval outputs (query_id, pipeline_id, chunk_id, score, rank)

**Critical relationships**:
- `query_id` → `query.id`
- `pipeline_id` → `pipeline.id`
- `metric_id` → `metric.id`
- `chunk_id` → `chunk.id`

### Step 3: Generate SQL Query

**MANDATORY RULES**:

1. ✅ **SELECT-only**: Never use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER
2. ⛔ **Exclude vector columns**: Never select `embedding`, `embeddings`, `bm25_tokens`
   - These cause type resolution errors
   - If user asks about embeddings, explain they're stored but not queryable
3. 📏 **Add LIMIT**: If user doesn't specify, add `LIMIT 100`
4. 🔗 **Use JOINs**: Connect tables via foreign keys (query_id, pipeline_id, metric_id)
5. 📦 **Extract JSONB**:
   - Use `token_usage->>'field'` for text extraction
   - Cast to type: `(token_usage->>'total_tokens')::int`
6. 🧹 **Handle NULLs**: Use `COALESCE()` and `NULLIF()` for safe operations

**SQL Generation Template**:
```sql
SELECT
    [columns, excluding vector types]
FROM [primary_table]
JOIN [related_tables] ON [foreign_key = primary_key]
WHERE [conditions]
ORDER BY [ranking_column] [ASC|DESC]
LIMIT [reasonable_limit];
```

### Step 4: Validate Query

Before execution, check:
- ✅ Contains SELECT keyword
- ⛔ No DDL/DML keywords (INSERT, UPDATE, DELETE, DROP, CREATE, ALTER)
- ⛔ No system functions (pg_read_file, pg_execute, etc.)
- ✅ Uses JOINs for relationships (not subqueries when avoidable)
- ✅ Includes LIMIT clause

**The query_executor.py script validates automatically**, but checking saves time.

### Step 5: Execute Query

```bash
python .agents/skills/autorag-query/scripts/query_executor.py \
  --query "[SQL_QUERY]" \
  --format [table|json|csv] \
  --timeout 10 \
  --limit 10000
```

**Arguments**:
- `--query`: SQL query string (required)
- `--format`: Output format (default: table)
  - `table`: ASCII table with headers
  - `json`: JSON array of objects
  - `csv`: CSV format with headers
- `--timeout`: Query timeout in seconds (default: 10)
- `--limit`: Maximum rows to return (default: 10000, 0=unlimited)

**Connection detection**:
1. **Primary**: Loads from `configs/db.yaml` using `DBConnection.from_config()`
2. **Fallback**: Loads from environment variables (`POSTGRES_HOST`, `POSTGRES_PORT`, etc.)

### Step 6: Handle Errors

**Common errors and solutions**:

| Error | Cause | Solution |
|-------|-------|----------|
| "Forbidden keyword" | Query contains INSERT/UPDATE/DELETE | Use SELECT-only |
| "Vector type error" | Selected vector/embedding columns | Exclude vector columns |
| "Query timeout" | Query too slow | Add WHERE filters, LIMIT clause |
| "Connection failed" | Missing credentials | Check `configs/db.yaml` or set env vars |

**Vector column auto-retry**: If query fails with vector type error, script automatically retries after removing vector columns.

### Step 7: Format and Present Results

**Table format** (default):
```
name              | metric_result | created_at
------------------|---------------|------------
hybrid_search_v2  | 0.85         | 2025-01-15
naive_rag         | 0.72         | 2025-01-14

(2 rows)
```

**JSON format**:
```json
[
  {
    "name": "hybrid_search_v2",
    "metric_result": 0.85,
    "created_at": "2025-01-15T10:30:00"
  }
]
```

**CSV format**:
```csv
name,metric_result,created_at
hybrid_search_v2,0.85,2025-01-15T10:30:00
naive_rag,0.72,2025-01-14T10:30:00
```

**Presenting to user**:
- Summarize key findings in natural language
- Show formatted table/JSON for details
- Highlight insights (e.g., "Pipeline X is 15% better than Y")
- Suggest follow-up queries if relevant

---

## Common Use Cases

Consult `references/common-queries.md` for 20+ curated query templates.

### 1. Pipeline Performance Comparison

**Questions**:
- "Which pipeline has the best BLEU score?"
- "Rank pipelines by retrieval precision"
- "Show me average metrics across all pipelines"

**Key tables**: `summary`, `pipeline`, `metric`

**Example template**:
```sql
SELECT p.name, s.metric_result
FROM summary s
JOIN pipeline p ON s.pipeline_id = p.id
JOIN metric m ON s.metric_id = m.id
WHERE m.name = :metric_name
ORDER BY s.metric_result DESC
LIMIT 10;
```

### 2. Metric Analysis

**Questions**:
- "What's the average BLEU score across all queries?"
- "Show metric breakdown by query"
- "Find queries with lowest scores"

**Key tables**: `evaluation_result`, `query`, `metric`, `pipeline`

**Example template**:
```sql
SELECT
    p.name AS pipeline_name,
    AVG(er.metric_result) AS avg_score,
    COUNT(*) AS query_count
FROM evaluation_result er
JOIN pipeline p ON er.pipeline_id = p.id
JOIN metric m ON er.metric_id = m.id
WHERE m.name = :metric_name
GROUP BY p.name
ORDER BY avg_score DESC;
```

### 3. Token Usage Tracking

**Questions**:
- "Which pipeline uses the most tokens?"
- "Show token usage by query"
- "What's the cost per query?"

**Key tables**: `executor_result`, `pipeline`, `query`

**JSONB extraction required**: `token_usage->>'total_tokens'`

**Example template**:
```sql
SELECT
    p.name AS pipeline_name,
    SUM((exe.token_usage->>'total_tokens')::int) AS total_tokens,
    AVG((exe.token_usage->>'total_tokens')::int) AS avg_tokens_per_query
FROM executor_result exe
JOIN pipeline p ON exe.pipeline_id = p.id
WHERE exe.token_usage IS NOT NULL
GROUP BY p.name
ORDER BY total_tokens DESC;
```

### 4. Retrieval Result Inspection

**Questions**:
- "What chunks were retrieved for query X?"
- "Show retrieval scores for a specific query"
- "Compare retrieval results across pipelines"

**Key tables**: `chunk_retrieved_result`, `chunk`, `query`, `pipeline`

**Example template**:
```sql
SELECT
    c.content AS chunk_text,
    crr.score AS retrieval_score,
    crr.rank
FROM chunk_retrieved_result crr
JOIN chunk c ON crr.chunk_id = c.id
WHERE crr.query_id = :query_id
  AND crr.pipeline_id = :pipeline_id
ORDER BY crr.rank ASC
LIMIT 10;
```

### 5. Ground Truth Comparison

**Questions**:
- "Compare generated answers vs ground truth"
- "Which queries have multiple ground truth answers?"
- "Show generation results for query X"

**Key tables**: `query`, `executor_result`, `pipeline`

**Example template**:
```sql
SELECT
    q.query AS question,
    q.ground_truths,
    p.name AS pipeline_name,
    exe.generation_result
FROM executor_result exe
JOIN query q ON exe.query_id = q.id
JOIN pipeline p ON exe.pipeline_id = p.id
WHERE q.id = :query_id
ORDER BY p.name;
```

---

## Schema Reference

### Key Tables Summary

**Content Storage**:
- `document`: Source documents
- `page`: Document pages
- `chunk`: Text chunks with embeddings (**⚠️ exclude `embedding` column**)
- `image_chunk`: Image chunks with embeddings (**⚠️ exclude `embedding` column**)

**Query & Ground Truth**:
- `query`: Search queries with ground truth answers
- `retrieval_relation`: Query-to-chunk relevance (ground truth)

**Pipeline Definitions**:
- `pipeline`: Pipeline configurations (type: retrieval/generation)
- `metric`: Metric definitions (type: retrieval/generation)

**Results**:
- `executor_result`: Generation pipeline outputs (generation_result, token_usage, execution_time)
- `evaluation_result`: Per-query metric scores
- `summary`: Aggregated pipeline metrics
- `chunk_retrieved_result`: Retrieval outputs (chunk_id, score, rank)
- `image_chunk_retrieved_result`: Image retrieval outputs

### Important Relationships

```
query.id ←─── query_id ───→ executor_result
                           └─→ evaluation_result
                           └─→ chunk_retrieved_result

pipeline.id ←─── pipeline_id ───→ executor_result
                                └─→ evaluation_result
                                └─→ chunk_retrieved_result

metric.id ←─── metric_id ───→ evaluation_result
                            └─→ summary

chunk.id ←─── chunk_id ───→ chunk_retrieved_result
                          └─→ retrieval_relation
```

### JSONB Fields

**executor_result.token_usage**:
```json
{
  "prompt_tokens": 150,
  "completion_tokens": 50,
  "total_tokens": 200,
  "embedding_tokens": 100
}
```

**Extraction patterns**:
- Text: `token_usage->>'prompt_tokens'` → `"150"`
- Integer: `(token_usage->>'prompt_tokens')::int` → `150`
- JSON: `token_usage->'embedding_tokens'` → `100`

**pipeline.config** (arbitrary JSONB):
```json
{
  "type": "hybrid",
  "model": "gpt-4",
  "top_k": 10,
  "alpha": 0.5
}
```

**Extraction**: `config->>'model'` → `"gpt-4"`

### Critical: Vector Column Exclusion

**⚠️ ALWAYS EXCLUDE**:
- `chunk.embedding` (pgvector type)
- `image_chunk.embedding` (pgvector type)
- `chunk.bm25_tokens` (bm25vector type)

**Why?**: These types cause DuckDB type resolution errors when queried.

**Auto-retry**: The query_executor.py script automatically retries after removing vector columns if it detects a vector type error.

**If user asks about embeddings**: Explain that embeddings are stored in the database but cannot be queried via this skill. Suggest using the ORM layer directly in Python code if they need to work with embedding vectors.

---

## Advanced Topics

### Multi-Table JOINs

When combining data from multiple tables, follow these patterns:

**Example: Full pipeline execution report**
```sql
SELECT
    p.name AS pipeline_name,
    q.query AS question,
    exe.generation_result,
    m.name AS metric_name,
    er.metric_result AS score,
    (exe.token_usage->>'total_tokens')::int AS tokens
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

**JOIN guidelines**:
- Use `INNER JOIN` when both tables must have matching rows
- Use `LEFT JOIN` when the right table may not have matches (e.g., metrics may not exist for all queries)
- Always specify ON conditions explicitly
- Avoid subqueries when JOINs are clearer

### JSONB Extraction Patterns

**Basic extraction**:
```sql
-- Text extraction
token_usage->>'prompt_tokens'

-- Nested JSON
token_usage->'embedding_tokens'

-- Type casting
(token_usage->>'total_tokens')::int
(config->>'alpha')::float
```

**NULL handling**:
```sql
-- Default value if NULL
COALESCE((token_usage->>'total_tokens')::int, 0)

-- Check if key exists
token_usage ? 'embedding_tokens'

-- Extract with default
(token_usage->>'total_tokens')::int, 0)
```

**Aggregations on JSONB**:
```sql
SELECT
    AVG((token_usage->>'total_tokens')::int) AS avg_tokens,
    SUM((token_usage->>'prompt_tokens')::int) AS total_prompt_tokens
FROM executor_result
WHERE token_usage IS NOT NULL;
```

### Aggregations and Window Functions

**GROUP BY aggregations**:
```sql
SELECT
    p.name,
    COUNT(*) AS query_count,
    AVG(er.metric_result) AS avg_score,
    MIN(er.metric_result) AS min_score,
    MAX(er.metric_result) AS max_score
FROM evaluation_result er
JOIN pipeline p ON er.pipeline_id = p.id
GROUP BY p.name
ORDER BY avg_score DESC;
```

**Window functions for ranking**:
```sql
SELECT
    p.name AS pipeline_name,
    m.name AS metric_name,
    s.metric_result,
    RANK() OVER (PARTITION BY m.name ORDER BY s.metric_result DESC) AS rank
FROM summary s
JOIN pipeline p ON s.pipeline_id = p.id
JOIN metric m ON s.metric_id = m.id
ORDER BY m.name, rank;
```

**Pivot tables (CASE + MAX)**:
```sql
SELECT
    p.name AS pipeline_name,
    MAX(CASE WHEN m.name = 'bleu' THEN s.metric_result END) AS bleu,
    MAX(CASE WHEN m.name = 'rouge' THEN s.metric_result END) AS rouge,
    MAX(CASE WHEN m.name = 'precision' THEN s.metric_result END) AS precision
FROM summary s
JOIN pipeline p ON s.pipeline_id = p.id
JOIN metric m ON s.metric_id = m.id
GROUP BY p.name;
```

---

## Security and Best Practices

### Security Enforcement

The query_executor.py script enforces these rules:

1. ✅ **SELECT-only**: Only SELECT statements allowed
2. ⛔ **No DDL/DML**: Rejects INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE
3. ⛔ **No system functions**: Rejects pg_read_file, pg_execute, pg_ls_dir, etc.
4. ⏱️ **Timeout**: 10-second default (configurable)
5. 📊 **Result limit**: 10,000 rows max (configurable)

### Best Practices

**Query performance**:
- Always add LIMIT clause if not specified by user
- Use indexes (foreign keys are indexed automatically)
- Avoid SELECT * on large tables
- Use WHERE filters before JOINs when possible

**NULL handling**:
```sql
-- Safe division
CAST(numerator AS FLOAT) / NULLIF(denominator, 0)

-- Default values
COALESCE(column, default_value)
```

**JSONB queries**:
- Add `WHERE jsonb_column IS NOT NULL` before extracting
- Cast types explicitly: `(jsonb_field->>'key')::int`
- Use `?` operator to check key existence: `WHERE token_usage ? 'total_tokens'`

**Date filtering**:
```sql
-- Last 30 days
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'

-- Specific date range
WHERE created_at BETWEEN '2025-01-01' AND '2025-01-31'
```

---

## Troubleshooting

### Connection Issues

**Error**: "Database connection not found"

**Solutions**:
1. Verify `configs/db.yaml` exists in project root
2. Check environment variables: `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
3. Run from AutoRAG-Research project root directory

### Query Execution Errors

**Error**: "Forbidden keyword: INSERT"

**Solution**: Use SELECT-only queries. This skill is read-only.

---

**Error**: "Vector type error" or "bm25vector type error"

**Solution**: Exclude vector columns from SELECT:
- Remove: `embedding`, `embeddings`, `bm25_tokens`
- Script auto-retries after removing these columns

---

**Error**: "Query timeout"

**Solution**:
- Add WHERE filters to reduce result set
- Add LIMIT clause
- Increase timeout: `--timeout 30`

---

**Error**: "Syntax error near..."

**Solution**: Check SQL syntax:
- Missing comma in column list
- Mismatched parentheses
- Invalid JOIN conditions
- PostgreSQL-specific syntax (use `::int` for casting, not `CAST()`)

### Empty Results

**Result**: "No results found"

**Debugging**:
1. Check table has data: `SELECT COUNT(*) FROM table_name;`
2. Verify WHERE conditions aren't too restrictive
3. Check foreign key relationships (e.g., query_id exists in query table)
4. Use LEFT JOIN instead of INNER JOIN if optional relationships

---

## Examples by Complexity

### Simple (Single Table)

**Question**: "List all pipelines"

**SQL**:
```sql
SELECT id, name, pipeline_type, created_at
FROM pipeline
ORDER BY created_at DESC;
```

### Moderate (2-3 Tables)

**Question**: "Show BLEU scores by pipeline"

**SQL**:
```sql
SELECT p.name, s.metric_result
FROM summary s
JOIN pipeline p ON s.pipeline_id = p.id
JOIN metric m ON s.metric_id = m.id
WHERE m.name = 'bleu'
ORDER BY s.metric_result DESC;
```

### Advanced (Multi-Table + JSONB)

**Question**: "Compare token usage and BLEU scores across pipelines"

**SQL**:
```sql
SELECT
    p.name AS pipeline_name,
    s.metric_result AS bleu_score,
    AVG((exe.token_usage->>'total_tokens')::int) AS avg_tokens
FROM summary s
JOIN pipeline p ON s.pipeline_id = p.id
JOIN metric m ON s.metric_id = m.id
JOIN executor_result exe ON exe.pipeline_id = p.id
WHERE m.name = 'bleu'
  AND exe.token_usage IS NOT NULL
GROUP BY p.name, s.metric_result
ORDER BY bleu_score DESC, avg_tokens ASC;
```

### Expert (Window Functions + Pivot)

**Question**: "Rank pipelines by multiple metrics with relative performance"

**SQL**:
```sql
WITH metric_pivot AS (
    SELECT
        p.name AS pipeline_name,
        MAX(CASE WHEN m.name = 'bleu' THEN s.metric_result END) AS bleu,
        MAX(CASE WHEN m.name = 'rouge' THEN s.metric_result END) AS rouge,
        MAX(CASE WHEN m.name = 'precision' THEN s.metric_result END) AS precision
    FROM summary s
    JOIN pipeline p ON s.pipeline_id = p.id
    JOIN metric m ON s.metric_id = m.id
    GROUP BY p.name
)
SELECT
    pipeline_name,
    bleu,
    RANK() OVER (ORDER BY bleu DESC) AS bleu_rank,
    rouge,
    RANK() OVER (ORDER BY rouge DESC) AS rouge_rank,
    precision,
    RANK() OVER (ORDER BY precision DESC) AS precision_rank
FROM metric_pivot
ORDER BY bleu_rank, rouge_rank, precision_rank;
```

---

## Installation & Setup

### Project-Local Use (Automatic)

The skill is automatically available when placed in `.agents/skills/autorag-query/`.

All AI agents (Claude Code, Codex, Kiro, Antigravity, Cursor) can use it.

### Global Installation (Future)

Once pushed to GitHub:

```bash
# Install to ~/.agents/skills/ (universal)
npx skills add NomaDamas/AutoRAG-Research --skill autorag-query

# Install to agent-specific directory
npx skills add NomaDamas/AutoRAG-Research --skill autorag-query -a claude-code
```

### Requirements

**Database**:
- PostgreSQL 10+ with VectorChord/pgvector extensions
- Database connection via `configs/db.yaml` or environment variables

**Python packages** (already in AutoRAG-Research):
- `sqlalchemy` - SQL execution
- `psycopg` (psycopg3) - PostgreSQL driver
- `tabulate` - ASCII table formatting

---

## Summary

AutoRAG-Query enables AI agents to query AutoRAG-Research pipeline results using natural language:

1. **Parse** user's question
2. **Load** schema from `references/schema.sql`
3. **Generate** SQL following safety rules
4. **Execute** via `scripts/query_executor.py`
5. **Format** results as table/JSON/CSV
6. **Present** insights to user

**Key rules**:
- ✅ SELECT-only queries
- ⛔ Exclude vector columns (embedding, embeddings, bm25_tokens)
- 📏 Add LIMIT clause
- 🔗 Use JOINs for relationships
- 📦 Extract JSONB with `->>` and cast types

**Resources**:
- Schema: `references/schema.sql`
- Query templates: `references/common-queries.md`
- Executor: `scripts/query_executor.py`

For complex queries, consult `common-queries.md` for 20+ curated templates.
