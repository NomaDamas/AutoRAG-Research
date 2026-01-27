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
        db = self.attach_dataset(db_name)
        order = "ASC" if ascending else "DESC"
        # S608: db is from attach_dataset (user-controlled but quoted), order is hardcoded
        return self._conn.execute(
            f'''
            SELECT
                ROW_NUMBER() OVER (ORDER BY s.metric_result {order}) as rank,
                p.name as pipeline,
                s.metric_result as score,
                s.execution_time as time_ms
            FROM "{db}".summary s
            JOIN "{db}".pipeline p ON s.pipeline_id = p.id
            JOIN "{db}".metric m ON s.metric_id = m.id
            WHERE m.name = $1
            ORDER BY s.metric_result {order}
            LIMIT $2
            ''',  # noqa: S608
            [metric_name, limit],
        ).df()

    def list_pipelines(self, db_name: str) -> list[str]:
        """List all pipelines in a database.

        Args:
            db_name: Name of the PostgreSQL database.

        Returns:
            List of pipeline names.
        """
        db = self.attach_dataset(db_name)
        return self._conn.execute(f'SELECT name FROM "{db}".pipeline').df()["name"].tolist()  # noqa: S608

    def get_pipeline_type(self, db_name: str, pipeline_name: str) -> Literal["retrieval", "generation"] | None:
        """Get the type of a pipeline from its config.

        Args:
            db_name: Name of the PostgreSQL database.
            pipeline_name: Name of the pipeline.

        Returns:
            Pipeline type ('retrieval' or 'generation'), or None if not found.
        """
        db = self.attach_dataset(db_name)
        # Query config JSONB field for pipeline_type
        # S608: db from attach_dataset (quoted)
        result = self._conn.execute(
            f"""
            SELECT config->>'pipeline_type' as pipeline_type
            FROM "{db}".pipeline
            WHERE name = $1
            """,  # noqa: S608
            [pipeline_name],
        ).df()
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
        db = self.attach_dataset(db_name)
        return self._conn.execute(f'SELECT name FROM "{db}".metric').df()["name"].tolist()  # noqa: S608

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
        pg = self.attach_dataset("postgres")

        # Query pg_database for all user databases (excluding templates)
        # S608: pg is from attach_dataset, quoted identifier
        all_dbs = (
            self._conn.execute(
                f'''
            SELECT datname
            FROM "{pg}".pg_catalog.pg_database
            WHERE datistemplate = false
              AND datname NOT IN ('postgres')
            ORDER BY datname
            '''  # noqa: S608
            )
            .df()["datname"]
            .tolist()
        )

        # Check each database for AutoRAG-Research schema (presence of 'summary' table)
        valid_datasets = []
        for db_name in all_dbs:
            try:
                db = self.attach_dataset(db_name)
                # Check if summary table exists by querying information_schema
                # S608: db is from attach_dataset, quoted identifier
                result = self._conn.execute(
                    f'''
                    SELECT COUNT(*) as cnt
                    FROM "{db}".information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'summary'
                    '''  # noqa: S608
                ).df()
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

        db = self.attach_dataset(db_name)
        # Embed validated metric_type directly to avoid DuckDB PostgreSQL extension
        # parameter pushdown issues with ATTACH'ed databases
        # S608: db from attach_dataset (quoted), metric_type validated above
        return (
            self._conn.execute(f"SELECT name FROM \"{db}\".metric WHERE type = '{metric_type}' ORDER BY name")  # noqa: S608
            .df()["name"]
            .tolist()
        )

    def get_dataset_stats(self, db_name: str) -> dict:
        """Return dataset statistics including query, chunk, and document counts.

        Args:
            db_name: Name of the PostgreSQL database.

        Returns:
            Dictionary with keys: query_count, chunk_count, document_count.
        """
        db = self.attach_dataset(db_name)

        # Query counts from each table
        stats = {}

        # S608: db from attach_dataset (quoted), table names are hardcoded
        for table, key in [("query", "query_count"), ("chunk", "chunk_count"), ("document", "document_count")]:
            try:
                result = self._conn.execute(f'SELECT COUNT(*) as cnt FROM "{db}".{table}').df()  # noqa: S608
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

        results = []
        for db_name in db_names:
            db = self.attach_dataset(db_name)
            # S608: db is from attach_dataset (quoted)
            df = self._conn.execute(
                f'''
                SELECT $1 as dataset, s.metric_result as score, s.execution_time as time_ms
                FROM "{db}".summary s
                JOIN "{db}".pipeline p ON s.pipeline_id = p.id
                JOIN "{db}".metric m ON s.metric_id = m.id
                WHERE p.name = $2 AND m.name = $3
                ''',  # noqa: S608
                [db_name, pipeline_name, metric_name],
            ).df()
            results.append(df)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def get_all_metrics_leaderboard(
        self, db_name: str, metric_type: Literal["retrieval", "generation"]
    ) -> pd.DataFrame:
        """Get leaderboard with all metrics as columns.

        Args:
            db_name: Name of the PostgreSQL database.
            metric_type: One of 'retrieval' or 'generation'.

        Returns:
            DataFrame with columns: rank, pipeline, <metric1>, <metric2>, ..., Average
            Sorted by Average descending.
        """
        if metric_type not in ("retrieval", "generation"):
            raise ValueError(f"Invalid metric_type: '{metric_type}'. Must be 'retrieval' or 'generation'.")  # noqa: TRY003

        db = self.attach_dataset(db_name)
        # S608: db from attach_dataset (quoted), metric_type validated above
        df = self._conn.execute(
            f"""
            SELECT p.name as pipeline, m.name as metric, s.metric_result as score
            FROM "{db}".summary s
            JOIN "{db}".pipeline p ON s.pipeline_id = p.id
            JOIN "{db}".metric m ON s.metric_id = m.id
            WHERE m.type = '{metric_type}'
            """,  # noqa: S608
        ).df()

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

        results = []
        for db_name in db_names:
            db = self.attach_dataset(db_name)
            # S608: db from attach_dataset (quoted), pipeline_type from get_pipeline_type (validated)
            df = self._conn.execute(
                f"""
                SELECT $1 as dataset, m.name as metric, s.metric_result as score
                FROM "{db}".summary s
                JOIN "{db}".pipeline p ON s.pipeline_id = p.id
                JOIN "{db}".metric m ON s.metric_id = m.id
                WHERE p.name = $2 AND m.type = '{pipeline_type}'
                """,  # noqa: S608
                [db_name, pipeline_name],
            ).df()
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
            db = self.attach_dataset(db_name)

            for metric_name in metric_names:
                # Determine sort order for this metric
                ascending = metric_name in ascending_metrics
                order = "ASC" if ascending else "DESC"

                # S608: db from attach_dataset (quoted), order is hardcoded
                df = self._conn.execute(
                    f'''
                    SELECT
                        p.name as pipeline,
                        RANK() OVER (ORDER BY s.metric_result {order}) as rank
                    FROM "{db}".summary s
                    JOIN "{db}".pipeline p ON s.pipeline_id = p.id
                    JOIN "{db}".metric m ON s.metric_id = m.id
                    WHERE m.name = $1
                    ''',  # noqa: S608
                    [metric_name],
                ).df()

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
