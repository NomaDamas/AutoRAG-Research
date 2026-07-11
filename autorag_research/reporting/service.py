import logging
from types import TracebackType
from typing import Literal

import duckdb
import pandas as pd

from autorag_research.orm.connection import DBConnection

logger = logging.getLogger("AutoRAG-Research")


class ReportingService:
    """DuckDB-based result query service for RAG pipeline evaluation results.

    This service enables querying evaluation results across single or multiple
    PostgreSQL databases (each representing a dataset) using DuckDB's PostgreSQL
    extension for efficient cross-database analytics.

    All queries use ``postgres_query()`` to send raw SQL to PostgreSQL, avoiding
    DuckDB schema introspection of tables with unsupported types (e.g. pgvector
    VECTOR, VectorChord bm25vector).
    """

    def __init__(self, config: DBConnection | None = None):
        """Initialize the reporting service.

        Args:
            config: Database connection configuration. If None, loads from environment.
        """
        self.config = config or DBConnection.from_env()
        self._conn = duckdb.connect()
        try:
            self._conn.install_extension("postgres")
            self._conn.load_extension("postgres")
        except Exception:
            self._conn.close()
            raise
        self._attached_dbs: set[str] = set()

    def attach_dataset(self, db_name: str) -> str:
        """Attach a PostgreSQL database to DuckDB.

        Args:
            db_name: Name of the PostgreSQL database.

        Returns:
            The database name (used as quoted identifier in queries).
        """
        if db_name not in self._attached_dbs:
            self.config.database = db_name
            # Use double-quoted identifier to allow any valid PostgreSQL database name
            # Use duckdb_url (without +psycopg driver) for DuckDB PostgreSQL extension compatibility
            self._conn.execute(f"ATTACH '{self.config.duckdb_url}' AS \"{db_name}\" (TYPE POSTGRES, READ_ONLY)")
            self._attached_dbs.add(db_name)
        return db_name

    # === Helpers ===

    @staticmethod
    def _escape_sql_value(value: str) -> str:
        """Escape a string value for safe embedding in SQL by doubling single quotes."""
        return value.replace("'", "''")

    @staticmethod
    def _metric_scores_cte() -> str:
        """Return canonical aggregate metric scores with legacy summary fallback.

        Evaluation results are persisted per query. Summary rows are retained for
        legacy databases, but only supply a pipeline/metric pair when no evaluated
        rows exist for that pair.
        """
        return """
            WITH metric_scores AS (
                SELECT
                    pipeline_id,
                    metric_id,
                    AVG(metric_result) AS metric_result,
                    NULL::DOUBLE PRECISION AS execution_time
                FROM evaluation_result
                GROUP BY pipeline_id, metric_id
                UNION ALL
                SELECT
                    s.pipeline_id,
                    s.metric_id,
                    s.metric_result,
                    s.execution_time
                FROM summary s
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM evaluation_result e
                    WHERE e.pipeline_id = s.pipeline_id AND e.metric_id = s.metric_id
                )
            )
        """

    def _pg_query(self, db_name: str, sql: str) -> pd.DataFrame:
        """Execute a raw SQL query against PostgreSQL via DuckDB's postgres_query().

        This bypasses DuckDB's schema introspection of ATTACH'ed databases,
        avoiding crashes on tables with unsupported types (VECTOR, bm25vector).

        Args:
            db_name: Name of the PostgreSQL database (must be attached).
            sql: Raw PostgreSQL SQL to execute.

        Returns:
            Query results as a pandas DataFrame.
        """
        self.attach_dataset(db_name)
        escaped_sql = sql.replace("'", "''")
        return self._conn.execute(f"SELECT * FROM postgres_query('{db_name}', '{escaped_sql}')").df()  # noqa: S608

    # === Single Dataset Queries ===

    def get_leaderboard(self, db_name: str, metric_name: str, limit: int = 10, ascending: bool = False) -> pd.DataFrame:
        """Get pipeline rankings for a specific metric.

        Args:
            db_name: Name of the PostgreSQL database.
            metric_name: Name of the metric to rank by.
            limit: Maximum number of results to return.
            ascending: If True, lower scores rank higher. Defaults to False.

        Returns:
            DataFrame with columns: rank, pipeline, score, time_ms
        """
        order = "ASC" if ascending else "DESC"
        escaped_metric = self._escape_sql_value(metric_name)
        sql = f"""
            {self._metric_scores_cte()}
            SELECT
                ROW_NUMBER() OVER (ORDER BY s.metric_result {order}) as rank,
                p.name as pipeline,
                s.metric_result as score,
                s.execution_time as time_ms
            FROM metric_scores s
            JOIN pipeline p ON s.pipeline_id = p.id
            JOIN metric m ON s.metric_id = m.id
            WHERE m.name = '{escaped_metric}'
            ORDER BY s.metric_result {order}
            LIMIT {int(limit)}
            """  # noqa: S608
        return self._pg_query(db_name, sql)

    def list_pipelines(self, db_name: str) -> list[str]:
        """List all pipelines in a database.

        Args:
            db_name: Name of the PostgreSQL database.

        Returns:
            List of pipeline names.
        """
        sql = f"""
            {self._metric_scores_cte()}
            SELECT DISTINCT p.name
            FROM metric_scores s
            JOIN pipeline p ON s.pipeline_id = p.id
            ORDER BY p.name
            """  # noqa: S608
        return self._pg_query(db_name, sql)["name"].tolist()

    def get_pipeline_type(self, db_name: str, pipeline_name: str) -> Literal["retrieval", "generation"] | None:
        """Get a pipeline's execution type from persisted results.

        Args:
            db_name: Name of the PostgreSQL database.
            pipeline_name: Name of the pipeline.

        Returns:
            Pipeline type ('retrieval' or 'generation'), or None if no results exist.
        """
        escaped_name = self._escape_sql_value(pipeline_name)
        sql = f"""
            {self._metric_scores_cte()}
            SELECT CASE
                WHEN EXISTS (
                    SELECT 1 FROM executor_result e WHERE e.pipeline_id = p.id
                ) OR BOOL_OR(m.type = 'generation')
                THEN 'generation'
                ELSE 'retrieval'
            END AS pipeline_type
            FROM metric_scores s
            JOIN pipeline p ON s.pipeline_id = p.id
            JOIN metric m ON s.metric_id = m.id
            WHERE p.name = '{escaped_name}'
            GROUP BY p.id
            LIMIT 1
            """  # noqa: S608
        result = self._pg_query(db_name, sql)
        if result.empty or result["pipeline_type"].iloc[0] is None:
            return None
        return result["pipeline_type"].iloc[0]

    def list_metrics(self, db_name: str) -> list[str]:
        """List all metrics in a database.

        Args:
            db_name: Name of the PostgreSQL database.

        Returns:
            List of metric names.
        """
        sql = f"""
            {self._metric_scores_cte()}
            SELECT DISTINCT m.name
            FROM metric_scores s
            JOIN metric m ON s.metric_id = m.id
            ORDER BY m.name
            """  # noqa: S608
        return self._pg_query(db_name, sql)["name"].tolist()

    # === Dataset Discovery ===

    def list_available_datasets(self) -> list[str]:
        """List PostgreSQL databases that contain AutoRAG schema.

        Uses pg_database catalog to find databases, then checks for 'summary' table
        to identify valid AutoRAG datasets.

        Returns:
            List of database names that contain AutoRAG schema.
        """
        # First, get all non-template databases from PostgreSQL
        # We need to connect to 'postgres' database to query pg_database
        all_dbs = self._pg_query(
            "postgres",
            """
                SELECT datname
                FROM pg_catalog.pg_database
                WHERE datistemplate = false
                  AND datname NOT IN ('postgres')
                ORDER BY datname
                """,
        )["datname"].tolist()

        # Check each database for AutoRAG-Research schema (presence of 'summary' table)
        valid_datasets = []
        for db_name in all_dbs:
            try:
                result = self._pg_query(
                    db_name,
                    """
                    SELECT COUNT(*) as cnt
                    FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'summary'
                    """,
                )
                if result["cnt"].iloc[0] > 0:
                    valid_datasets.append(db_name)
            except Exception:
                logger.warning(f"Skipping {db_name}")
                continue

        return valid_datasets

    def list_metrics_by_type(self, db_name: str, metric_type: Literal["retrieval", "generation"]) -> list[str]:
        """Filter metrics by type.

        Args:
            db_name: Name of the PostgreSQL database.
            metric_type: One of 'retrieval' or 'generation'.

        Returns:
            List of metric names matching the specified type.

        Raises:
            ValueError: If metric_type is not 'retrieval' or 'generation'.
        """
        if metric_type not in ("retrieval", "generation"):
            raise ValueError(f"Invalid metric_type: '{metric_type}'. Must be 'retrieval' or 'generation'.")  # noqa: TRY003

        sql = f"""
            {self._metric_scores_cte()}
            SELECT DISTINCT m.name
            FROM metric_scores s
            JOIN metric m ON s.metric_id = m.id
            WHERE m.type = '{metric_type}'
            ORDER BY m.name
            """  # noqa: S608
        return self._pg_query(db_name, sql)["name"].tolist()

    def get_dataset_stats(self, db_name: str) -> dict:
        """Return dataset statistics including query, chunk, and document counts.

        Args:
            db_name: Name of the PostgreSQL database.

        Returns:
            Dictionary with keys: query_count, chunk_count, document_count.
        """
        stats = {}

        for table, key in [("query", "query_count"), ("chunk", "chunk_count"), ("document", "document_count")]:
            try:
                result = self._pg_query(db_name, f"SELECT COUNT(*) as cnt FROM {table}")  # noqa: S608
                stats[key] = int(result["cnt"].iloc[0])
            except Exception:
                stats[key] = 0

        return stats

    # === Multi-Dataset Queries ===

    def compare_across_datasets(self, db_names: list[str], pipeline_name: str, metric_name: str) -> pd.DataFrame:
        """Compare a single pipeline's performance across multiple datasets.

        Args:
            db_names: List of database names (each representing a dataset).
            pipeline_name: Name of the pipeline to compare.
            metric_name: Name of the metric to use.

        Returns:
            DataFrame with columns: dataset, score, time_ms
        """
        if not db_names:
            return pd.DataFrame()

        escaped_pipeline = self._escape_sql_value(pipeline_name)
        escaped_metric = self._escape_sql_value(metric_name)

        results = []
        for db_name in db_names:
            sql = f"""
                {self._metric_scores_cte()}
                SELECT s.metric_result as score, s.execution_time as time_ms
                FROM metric_scores s
                JOIN pipeline p ON s.pipeline_id = p.id
                JOIN metric m ON s.metric_id = m.id
                WHERE p.name = '{escaped_pipeline}' AND m.name = '{escaped_metric}'
                """  # noqa: S608
            df = self._pg_query(db_name, sql)
            df["dataset"] = db_name
            results.append(df)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def get_all_metrics_leaderboard(
        self,
        db_name: str,
        metric_type: Literal["retrieval", "generation"],
        pipeline_names: list[str] | tuple[str, ...] | None = None,
        metric_names: list[str] | tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """Get leaderboard with selected pipelines and metrics as columns.

        Args:
            db_name: Name of the PostgreSQL database.
            metric_type: One of 'retrieval' or 'generation'.
            pipeline_names: Optional pipeline allowlist. ``None`` includes every pipeline.
            metric_names: Optional metric allowlist. ``None`` includes every metric of the requested type.

        Returns:
            DataFrame with columns: rank, pipeline, <metric1>, <metric2>, ..., Average
            Sorted by Average descending.
        """
        if metric_type not in ("retrieval", "generation"):
            raise ValueError(f"Invalid metric_type: '{metric_type}'. Must be 'retrieval' or 'generation'.")  # noqa: TRY003
        if pipeline_names is not None and not pipeline_names:
            return pd.DataFrame()
        if metric_names is not None and not metric_names:
            return pd.DataFrame()

        filters = [f"m.type = '{metric_type}'"]
        if pipeline_names is not None:
            escaped_pipelines = ", ".join(f"'{self._escape_sql_value(name)}'" for name in pipeline_names)
            filters.append(f"p.name IN ({escaped_pipelines})")
        if metric_names is not None:
            escaped_metrics = ", ".join(f"'{self._escape_sql_value(name)}'" for name in metric_names)
            filters.append(f"m.name IN ({escaped_metrics})")
        where_clause = " AND ".join(filters)

        sql = f"""
            {self._metric_scores_cte()}
            SELECT p.name as pipeline, m.name as metric, s.metric_result as score
            FROM metric_scores s
            JOIN pipeline p ON s.pipeline_id = p.id
            JOIN metric m ON s.metric_id = m.id
            WHERE {where_clause}
            """  # noqa: S608
        df = self._pg_query(db_name, sql)

        if df.empty:
            return pd.DataFrame()

        # Pivot: rows=pipeline, columns=metric, values=score
        pivoted = df.pivot(index="pipeline", columns="metric", values="score")
        pivoted["Average"] = pivoted.mean(axis=1)
        pivoted = pivoted.sort_values("Average", ascending=False).reset_index()
        pivoted.insert(0, "rank", list(range(1, len(pivoted) + 1)))  # ty: ignore[invalid-argument-type]

        # Round numeric columns for display
        numeric_cols = pivoted.select_dtypes(include=["float64"]).columns
        pivoted[numeric_cols] = pivoted[numeric_cols].round(3)

        return pivoted

    def compare_pipeline_all_metrics(self, db_names: list[str], pipeline_name: str) -> pd.DataFrame:
        """Compare pipeline across datasets with all metrics.

        Args:
            db_names: List of database names (each representing a dataset).
            pipeline_name: Name of the pipeline to compare.

        Returns:
            DataFrame with columns: dataset, <metric1>, <metric2>, ..., Average
        """
        if not db_names or not pipeline_name:
            return pd.DataFrame()

        # Get pipeline type from first dataset
        pipeline_type = self.get_pipeline_type(db_names[0], pipeline_name)
        if not pipeline_type:
            return pd.DataFrame()

        escaped_pipeline = self._escape_sql_value(pipeline_name)

        results = []
        for db_name in db_names:
            escaped_dataset = self._escape_sql_value(db_name)
            sql = f"""
                {self._metric_scores_cte()}
                SELECT '{escaped_dataset}' as dataset, m.name as metric, s.metric_result as score
                FROM metric_scores s
                JOIN pipeline p ON s.pipeline_id = p.id
                JOIN metric m ON s.metric_id = m.id
                WHERE p.name = '{escaped_pipeline}' AND m.type = '{pipeline_type}'
                """  # noqa: S608
            df = self._pg_query(db_name, sql)
            results.append(df)

        if not results:
            return pd.DataFrame()

        combined = pd.concat(results, ignore_index=True)
        if combined.empty:
            return pd.DataFrame()

        pivoted = combined.pivot(index="dataset", columns="metric", values="score")
        pivoted["Average"] = pivoted.mean(axis=1)
        pivoted = pivoted.reset_index()

        # Round numeric columns for display
        numeric_cols = pivoted.select_dtypes(include=["float64"]).columns
        pivoted[numeric_cols] = pivoted[numeric_cols].round(3)

        return pivoted

    def get_borda_count_leaderboard(
        self,
        db_names: list[str],
        metric_names: list[str],
        ascending_metrics: set[str] | None = None,
    ) -> pd.DataFrame:
        """Compute Borda Count rankings across datasets/metrics.

        The Borda Count is a fair ranking method that aggregates rankings across
        multiple criteria. This is similar to MTEB's approach for ranking models.

        Algorithm:
        1. For each (dataset, metric) pair, rank pipelines from 1 to N
        2. Sum ranks across all (dataset, metric) pairs
        3. Lower total rank = better overall performance

        Args:
            db_names: List of database names (each representing a dataset).
            metric_names: List of metric names to include in ranking.
            ascending_metrics: Set of metric names where lower is better (e.g., latency).
                              Default is None (all metrics: higher is better).

        Returns:
            DataFrame with columns: pipeline, total_rank, avg_rank, num_rankings
            Sorted by total_rank ascending (lower is better).
        """
        if not db_names or not metric_names:
            return pd.DataFrame()

        ascending_metrics = ascending_metrics or set()

        # Collect all rankings across (dataset, metric) pairs
        all_rankings: list[pd.DataFrame] = []

        for db_name in db_names:
            for metric_name in metric_names:
                # Determine sort order for this metric
                ascending = metric_name in ascending_metrics
                order = "ASC" if ascending else "DESC"
                escaped_metric = self._escape_sql_value(metric_name)

                sql = f"""
                    {self._metric_scores_cte()}
                    SELECT
                        p.name as pipeline,
                        RANK() OVER (ORDER BY s.metric_result {order}) as rank
                    FROM metric_scores s
                    JOIN pipeline p ON s.pipeline_id = p.id
                    JOIN metric m ON s.metric_id = m.id
                    WHERE m.name = '{escaped_metric}'
                    """  # noqa: S608
                df = self._pg_query(db_name, sql)

                if not df.empty:
                    df["dataset"] = db_name
                    df["metric"] = metric_name
                    all_rankings.append(df)

        if not all_rankings:
            return pd.DataFrame()

        # Combine all rankings
        combined = pd.concat(all_rankings, ignore_index=True)

        # Aggregate by pipeline: sum ranks, count rankings
        result = (
            combined.groupby("pipeline").agg(total_rank=("rank", "sum"), num_rankings=("rank", "count")).reset_index()
        )

        # Compute average rank
        result["avg_rank"] = result["total_rank"] / result["num_rankings"]

        # Sort by total_rank (lower is better)
        result = result.sort_values("total_rank").reset_index(drop=True)

        return result[["pipeline", "total_rank", "avg_rank", "num_rankings"]]

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()

    def __enter__(self) -> "ReportingService":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
