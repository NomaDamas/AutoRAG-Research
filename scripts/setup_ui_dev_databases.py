"""Setup multiple databases for UI development testing.

Creates three separate databases with different dataset characteristics
for testing cross-database comparison features in the UI.

Databases created:
- dataset_alpha: Scientific domain, 15 queries, colbert_v2 performs best
- dataset_beta: Legal domain, 8 queries, bm25_baseline performs best
- dataset_gamma: Support domain, 25 queries, hybrid_fusion performs best

Usage:
    uv run python scripts/setup_ui_dev_databases.py
    # Or via Makefile:
    make ui-setup
"""

import os
from pathlib import Path

import psycopg
import psycopg.sql

from autorag_research.orm.util import create_database, drop_database, install_vector_extensions

# Database names for UI testing
DATABASES = ["dataset_alpha", "dataset_beta", "dataset_gamma"]

# Schema file path
SCHEMA_FILE = Path(__file__).parent.parent / "postgresql/db/init/001-schema.sql"

# Connection params from env (same pattern as conftest.py)
HOST = os.getenv("POSTGRES_HOST", "localhost")
PORT = int(os.getenv("POSTGRES_PORT", "5433"))
USER = os.getenv("POSTGRES_USER", "postgres")
PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

# Common pipelines for all datasets
PIPELINES = [
    (1, "bm25_baseline", '{"type": "bm25", "k1": 1.2, "b": 0.75}'),
    (2, "dense_retriever", '{"type": "dense", "model": "sentence-transformers/all-MiniLM-L6-v2"}'),
    (3, "hybrid_fusion", '{"type": "hybrid", "alpha": 0.5, "dense_weight": 0.6, "sparse_weight": 0.4}'),
    (4, "colbert_v2", '{"type": "colbert", "model": "colbert-ir/colbertv2.0", "max_doc_len": 180}'),
]

# Common metrics for all datasets
METRICS = [
    (1, "recall@5", "retrieval"),
    (2, "recall@10", "retrieval"),
    (3, "ndcg@10", "retrieval"),
    (4, "mrr", "retrieval"),
    (5, "bleu", "generation"),
    (6, "rouge_l", "generation"),
    (7, "bertscore", "generation"),
]


def _insert_common_data(conn: psycopg.Connection) -> None:
    """Insert pipelines and metrics common to all datasets."""
    with conn.cursor() as cur:
        # Insert pipelines
        for pid, name, config in PIPELINES:
            cur.execute(
                "INSERT INTO pipeline (id, name, config) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (pid, name, config),
            )

        # Insert metrics
        for mid, name, mtype in METRICS:
            cur.execute(
                "INSERT INTO metric (id, name, type) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (mid, name, mtype),
            )


def _insert_chunks(conn: psycopg.Connection, count: int) -> None:
    """Insert chunk data."""
    with conn.cursor() as cur:
        for i in range(1, count + 1):
            cur.execute(
                "INSERT INTO chunk (id, contents) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (i, f"Chunk content {i} for testing retrieval"),
            )


def _insert_queries(conn: psycopg.Connection, count: int, domain: str) -> None:
    """Insert query data for a specific domain."""
    with conn.cursor() as cur:
        for i in range(1, count + 1):
            cur.execute(
                "INSERT INTO query (id, contents, generation_gt) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (i, f"{domain} query {i}: sample question text", [f"Ground truth answer for query {i}"]),
            )


def _insert_summary_results(
    conn: psycopg.Connection,
    results: dict[tuple[int, int], float],
) -> None:
    """Insert summary results for pipeline-metric combinations."""
    with conn.cursor() as cur:
        for (pipeline_id, metric_id), score in results.items():
            cur.execute(
                """INSERT INTO summary (pipeline_id, metric_id, metric_result, token_usage, execution_time)
                   VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                (
                    pipeline_id,
                    metric_id,
                    score,
                    '{"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}',
                    1500,
                ),
            )


def _advance_sequences(conn: psycopg.Connection) -> None:
    """Advance all sequences to prevent ID conflicts on future inserts."""
    sequences = ["chunk", "query", "pipeline", "metric"]
    with conn.cursor() as cur:
        for table in sequences:
            # Use psycopg.sql for safe SQL composition
            query = psycopg.sql.SQL(
                "SELECT setval(pg_get_serial_sequence({table}, 'id'), "
                "(SELECT COALESCE(MAX(id), 1) FROM {table_ident}), true)"
            ).format(
                table=psycopg.sql.Literal(table),
                table_ident=psycopg.sql.Identifier(table),
            )
            cur.execute(query)


def seed_alpha(conn: psycopg.Connection) -> None:
    """Seed dataset_alpha: Scientific domain, colbert_v2 performs best.

    - 15 queries
    - 12 chunks
    - Best pipeline: colbert_v2 with recall@10 = 0.94
    """
    _insert_common_data(conn)
    _insert_chunks(conn, 12)
    _insert_queries(conn, 15, "Scientific")

    # Summary results - colbert_v2 (id=4) performs best
    results = {
        # bm25_baseline (1)
        (1, 1): 0.72,  # recall@5
        (1, 2): 0.78,  # recall@10
        (1, 3): 0.71,  # ndcg@10
        (1, 4): 0.65,  # mrr
        (1, 5): 0.42,  # bleu
        (1, 6): 0.48,  # rouge_l
        (1, 7): 0.81,  # bertscore
        # dense_retriever (2)
        (2, 1): 0.80,
        (2, 2): 0.86,
        (2, 3): 0.79,
        (2, 4): 0.74,
        (2, 5): 0.51,
        (2, 6): 0.56,
        (2, 7): 0.85,
        # hybrid_fusion (3)
        (3, 1): 0.84,
        (3, 2): 0.89,
        (3, 3): 0.83,
        (3, 4): 0.78,
        (3, 5): 0.55,
        (3, 6): 0.60,
        (3, 7): 0.87,
        # colbert_v2 (4) - BEST
        (4, 1): 0.91,
        (4, 2): 0.94,  # Best recall@10
        (4, 3): 0.90,
        (4, 4): 0.86,
        (4, 5): 0.62,
        (4, 6): 0.67,
        (4, 7): 0.91,
    }
    _insert_summary_results(conn, results)
    _advance_sequences(conn)


def seed_beta(conn: psycopg.Connection) -> None:
    """Seed dataset_beta: Legal domain, bm25_baseline performs best.

    - 8 queries
    - 18 chunks
    - Best pipeline: bm25_baseline with recall@10 = 0.89
    """
    _insert_common_data(conn)
    _insert_chunks(conn, 18)
    _insert_queries(conn, 8, "Legal")

    # Summary results - bm25_baseline (id=1) performs best (legal domain favors keyword matching)
    results = {
        # bm25_baseline (1) - BEST
        (1, 1): 0.85,
        (1, 2): 0.89,  # Best recall@10
        (1, 3): 0.84,
        (1, 4): 0.80,
        (1, 5): 0.58,
        (1, 6): 0.63,
        (1, 7): 0.86,
        # dense_retriever (2)
        (2, 1): 0.68,
        (2, 2): 0.74,
        (2, 3): 0.67,
        (2, 4): 0.62,
        (2, 5): 0.45,
        (2, 6): 0.50,
        (2, 7): 0.78,
        # hybrid_fusion (3)
        (3, 1): 0.79,
        (3, 2): 0.84,
        (3, 3): 0.78,
        (3, 4): 0.73,
        (3, 5): 0.52,
        (3, 6): 0.57,
        (3, 7): 0.82,
        # colbert_v2 (4)
        (4, 1): 0.75,
        (4, 2): 0.81,
        (4, 3): 0.74,
        (4, 4): 0.69,
        (4, 5): 0.49,
        (4, 6): 0.54,
        (4, 7): 0.80,
    }
    _insert_summary_results(conn, results)
    _advance_sequences(conn)


def seed_gamma(conn: psycopg.Connection) -> None:
    """Seed dataset_gamma: Support domain, hybrid_fusion performs best.

    - 25 queries
    - 20 chunks
    - Best pipeline: hybrid_fusion with recall@10 = 0.89
    """
    _insert_common_data(conn)
    _insert_chunks(conn, 20)
    _insert_queries(conn, 25, "Support")

    # Summary results - hybrid_fusion (id=3) performs best (support needs both keyword + semantic)
    results = {
        # bm25_baseline (1)
        (1, 1): 0.70,
        (1, 2): 0.76,
        (1, 3): 0.69,
        (1, 4): 0.64,
        (1, 5): 0.40,
        (1, 6): 0.45,
        (1, 7): 0.79,
        # dense_retriever (2)
        (2, 1): 0.78,
        (2, 2): 0.84,
        (2, 3): 0.77,
        (2, 4): 0.72,
        (2, 5): 0.50,
        (2, 6): 0.55,
        (2, 7): 0.84,
        # hybrid_fusion (3) - BEST
        (3, 1): 0.85,
        (3, 2): 0.89,  # Best recall@10
        (3, 3): 0.84,
        (3, 4): 0.80,
        (3, 5): 0.58,
        (3, 6): 0.63,
        (3, 7): 0.88,
        # colbert_v2 (4)
        (4, 1): 0.82,
        (4, 2): 0.87,
        (4, 3): 0.81,
        (4, 4): 0.76,
        (4, 5): 0.54,
        (4, 6): 0.59,
        (4, 7): 0.86,
    }
    _insert_summary_results(conn, results)
    _advance_sequences(conn)


def setup_database(db_name: str, schema_sql: str, seed_func) -> None:
    """Setup a single database with schema and seed data.

    Args:
        db_name: Name of the database to create.
        schema_sql: SQL string containing the schema definition.
        seed_func: Function to seed the database with test data.
    """
    print(f"  Setting up '{db_name}'...")

    # 1. Drop if exists (clean slate)
    dropped = drop_database(HOST, USER, PASSWORD, db_name, PORT, force=True)
    if dropped:
        print("    Dropped existing database")

    # 2. Create new database
    create_database(HOST, USER, PASSWORD, db_name, PORT)
    print("    Created database")

    # 3. Install vector extensions
    install_vector_extensions(HOST, USER, PASSWORD, db_name, PORT)
    print("    Installed vector extensions")

    # 4. Apply schema + seed data
    with psycopg.connect(host=HOST, port=PORT, user=USER, password=PASSWORD, dbname=db_name) as conn:
        with conn.cursor() as cur:
            # Use bytes for schema SQL from file (bypasses LiteralString requirement)
            cur.execute(schema_sql.encode())
        seed_func(conn)
        conn.commit()
    print("    Applied schema and seed data")
    print(f"  ✓ '{db_name}' ready")


def main() -> None:
    """Main entry point for setting up UI development databases."""
    print(f"🎨 Setting up UI development databases on {HOST}:{PORT}")
    print()

    # Load schema
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(SCHEMA_FILE)
    schema_sql = SCHEMA_FILE.read_text()

    # Setup each database
    db_configs = [
        ("dataset_alpha", seed_alpha),
        ("dataset_beta", seed_beta),
        ("dataset_gamma", seed_gamma),
    ]

    for db_name, seed_func in db_configs:
        setup_database(db_name, schema_sql, seed_func)
        print()

    print("✅ All UI development databases ready!")
    print()
    print("Available databases:")
    print("  - testdb (default, from docker init)")
    print("  - dataset_alpha (Scientific, 15 queries, colbert_v2 best)")
    print("  - dataset_beta (Legal, 8 queries, bm25_baseline best)")
    print("  - dataset_gamma (Support, 25 queries, hybrid_fusion best)")


if __name__ == "__main__":
    main()
