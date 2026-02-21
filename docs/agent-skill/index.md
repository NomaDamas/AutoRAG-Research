# Agent Skill: autorag-query

AutoRAG-Research ships with an [agent skill](https://vercel.com/changelog/introducing-skills-the-open-agent-skills-ecosystem) that lets AI coding agents query pipeline results and metrics using natural language.

The skill follows the [Vercel skills standard](https://skills.sh) and works with Claude Code, Codex, Kiro, Cursor, and other compatible agents.

## Installation

The skill is bundled at `.agents/skills/autorag-query/` in the repository and is auto-detected by agents when you work inside the project.

To install globally (available across all projects):

```bash
npx skills add NomaDamas/AutoRAG-Research --skill autorag-query
```

## How It Works

When you ask a data question, the agent:

1. Reads the bundled database schema (`references/schema.sql`)
2. Generates a SELECT-only SQL query
3. Executes it via `scripts/query_executor.py`
4. Returns formatted results (table / JSON / CSV)

**Example:**

> **You**: "Which pipeline has the best BLEU score?"
>
> **Agent** reads the schema, generates SQL, runs it, and replies:
> "**hybrid_search_v2** achieved the highest BLEU score of **0.85**."

## What You Can Ask

- "Show me all pipelines and their types"
- "Which retrieval pipeline has the best recall?"
- "Compare token usage across generation pipelines"
- "What are the 5 worst-performing queries for BLEU?"
- "Show retrieval scores for query #42"

## Query Executor Script

The skill includes a standalone script you can also run directly.

### Basic Usage

```bash
uv run python .agents/skills/autorag-query/scripts/query_executor.py \
  --query "SELECT name, pipeline_type FROM pipeline LIMIT 5" \
  --config-path configs
```

### Parameterized Queries

Use `:param_name` placeholders with `--params` for safe value substitution:

```bash
uv run python .agents/skills/autorag-query/scripts/query_executor.py \
  --query "SELECT p.name, s.metric_result FROM summary s JOIN pipeline p ON s.pipeline_id = p.id JOIN metric m ON s.metric_id = m.id WHERE m.name = :metric_name ORDER BY s.metric_result DESC LIMIT 3" \
  --config-path configs \
  --params '{"metric_name": "rouge"}'
```

### Output Formats

```bash
# JSON output
uv run python .agents/skills/autorag-query/scripts/query_executor.py \
  --query "SELECT name, metric_type FROM metric" \
  --config-path configs \
  --format json

# CSV output
uv run python .agents/skills/autorag-query/scripts/query_executor.py \
  --query "SELECT name FROM pipeline" \
  --config-path configs \
  --format csv
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--query`, `-q` | SQL query (SELECT only, required) | - |
| `--format`, `-f` | Output format: `table`, `json`, `csv` | `table` |
| `--config-path`, `-c` | Path to `configs/` directory containing `db.yaml` | env vars fallback |
| `--params`, `-p` | JSON parameters for `:param` placeholders | - |
| `--timeout`, `-t` | Query timeout in seconds | `10` |
| `--limit`, `-l` | Max rows returned (0 = unlimited) | `10000` |
| `--database`, `-d` | Database name override | from config |

## Connection

The script auto-detects the database connection:

1. **Config file** (if `--config-path` is provided): Reads `db.yaml` from the specified directory
2. **Environment variables** (fallback): Uses `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`

## Safety

- Only `SELECT` statements are allowed (DDL/DML keywords are rejected)
- Dangerous PostgreSQL functions are blocked (`pg_read_file`, `pg_execute`, `COPY`, etc.)
- Results are capped at 10,000 rows by default (enforced via subquery wrapper)
- Query timeout defaults to 10 seconds
- Engine connections are disposed after each execution

## Query Templates

The skill bundles 20+ query templates in `references/common-queries.md`, organized by use case:

- **Pipeline comparison**: Top pipelines by metric, multi-metric pivot tables
- **Per-query analysis**: Score breakdowns, ground truth comparison, worst-performing queries
- **Retrieval results**: Retrieved chunks with scores, recall calculation
- **Token usage**: Per-pipeline totals, most expensive queries, usage over time
- **Execution performance**: Slowest queries, average execution time by pipeline
- **JSONB extraction**: `token_usage`, `config`, and `result_metadata` patterns

## Skill Directory Structure

```
.agents/skills/autorag-query/
├── SKILL.md                    # Skill definition (auto-detected by agents)
├── references/
│   ├── schema.sql              # Complete database schema
│   └── common-queries.md       # 20+ curated query templates
└── scripts/
    └── query_executor.py       # Safe SQL execution script
```
