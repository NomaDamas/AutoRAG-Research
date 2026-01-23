import re
from types import TracebackType

import duckdb
import pandas as pd

from autorag_research.reporting.config import DatabaseConfig

# Valid SQL identifier pattern (letters, digits, underscores, starting with letter/underscore)
_VALID_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Valid aggregation functions for pandas
VALID_AGGREGATIONS = frozenset({"mean", "median", "min", "max", "sum", "std"})


class ReportingService:
    """DuckDB-based result query service for RAG pipeline evaluation results.

    This service enables querying evaluation results across single or multiple
    PostgreSQL databases (each representing a dataset) using DuckDB's PostgreSQL
    extension for efficient cross-database analytics.
    """

    def __init__(self, config: DatabaseConfig | None = None):
        """Initialize the reporting service.

        Args:
            config: Database connection configuration. If None, loads from environment.
        """
        self.config = config or DatabaseConfig.from_env()
        self._conn = duckdb.connect()
        try:
            self._conn.install_extension("postgres")
            self._conn.load_extension("postgres")
        except Exception:
            self._conn.close()
            raise
        self._attached_dbs: set[str] = set()

    def attach_dataset(self, db_name: str, alias: str | None = None) -> str:
        """Attach a PostgreSQL database to DuckDB.

        Args:
            db_name: Name of the PostgreSQL database.
            alias: Optional alias for the database. Defaults to db_name with hyphens replaced.

        Returns:
            The alias used for the attached database.

        Raises:
            ValueError: If the alias is not a valid SQL identifier.
        """
        alias = alias or db_name.replace("-", "_")
        if not _VALID_IDENTIFIER_PATTERN.match(alias):
            raise ValueError(
                f"Invalid alias: '{alias}'. Must be a valid SQL identifier "
                "(letters, digits, underscores, starting with letter or underscore)."
            )
        if alias not in self._attached_dbs:
            conn_str = self.config.get_duckdb_connection_string(db_name)
            self._conn.execute(f"ATTACH '{conn_str}' AS {alias} (TYPE POSTGRES, READ_ONLY)")
            self._attached_dbs.add(alias)
        return alias

    # === Single Dataset Queries ===

    def get_leaderboard(
        self, db_name: str, metric_name: str, limit: int = 10, ascending: bool = False
    ) -> pd.DataFrame:
        """Get pipeline rankings for a specific metric.

        Args:
            db_name: Name of the PostgreSQL database.
            metric_name: Name of the metric to rank by.
            limit: Maximum number of results to return.
            ascending: If True, lower scores rank higher. Defaults to False.

        Returns:
            DataFrame with columns: rank, pipeline, score, time_ms
        """
        alias = self.attach_dataset(db_name)
        order = "ASC" if ascending else "DESC"
        return self._conn.execute(
            f"""
            SELECT
                ROW_NUMBER() OVER (ORDER BY s.metric_result {order}) as rank,
                p.name as pipeline,
                s.metric_result as score,
                s.execution_time as time_ms
            FROM {alias}.summary s
            JOIN {alias}.pipeline p ON s.pipeline_id = p.id
            JOIN {alias}.metric m ON s.metric_id = m.id
            WHERE m.name = $1
            ORDER BY s.metric_result {order}
            LIMIT $2
            """,
            [metric_name, limit],
        ).df()

    def compare_pipelines(
        self, db_name: str, pipeline_names: list[str], metric_names: list[str]
    ) -> pd.DataFrame:
        """Compare multiple pipelines across multiple metrics (pivot table).

        Args:
            db_name: Name of the PostgreSQL database.
            pipeline_names: List of pipeline names to compare.
            metric_names: List of metric names to include.

        Returns:
            DataFrame with pipelines as rows and metrics as columns.
        """
        if not pipeline_names or not metric_names:
            return pd.DataFrame()

        alias = self.attach_dataset(db_name)

        # Create parameterized placeholders
        pipeline_placeholders = ", ".join(f"${i + 1}" for i in range(len(pipeline_names)))
        metric_placeholders = ", ".join(
            f"${i + 1 + len(pipeline_names)}" for i in range(len(metric_names))
        )

        df = self._conn.execute(
            f"""
            SELECT p.name as pipeline, m.name as metric, s.metric_result as score
            FROM {alias}.summary s
            JOIN {alias}.pipeline p ON s.pipeline_id = p.id
            JOIN {alias}.metric m ON s.metric_id = m.id
            WHERE p.name IN ({pipeline_placeholders}) AND m.name IN ({metric_placeholders})
            """,
            [*pipeline_names, *metric_names],
        ).df()

        if df.empty:
            return df

        return df.pivot(index="pipeline", columns="metric", values="score")

    def list_pipelines(self, db_name: str) -> list[str]:
        """List all pipelines in a database.

        Args:
            db_name: Name of the PostgreSQL database.

        Returns:
            List of pipeline names.
        """
        alias = self.attach_dataset(db_name)
        return self._conn.execute(f"SELECT name FROM {alias}.pipeline").df()["name"].tolist()

    def list_metrics(self, db_name: str) -> list[str]:
        """List all metrics in a database.

        Args:
            db_name: Name of the PostgreSQL database.

        Returns:
            List of metric names.
        """
        alias = self.attach_dataset(db_name)
        return self._conn.execute(f"SELECT name FROM {alias}.metric").df()["name"].tolist()

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
        postgres_alias = self.attach_dataset("postgres", alias="pg_catalog_db")

        # Query pg_database for all user databases (excluding templates)
        all_dbs = self._conn.execute(
            f"""
            SELECT datname
            FROM {postgres_alias}.pg_database
            WHERE datistemplate = false
              AND datname NOT IN ('postgres')
            ORDER BY datname
            """
        ).df()["datname"].tolist()

        # Check each database for AutoRAG schema (presence of 'summary' table)
        valid_datasets = []
        for db_name in all_dbs:
            try:
                alias = self.attach_dataset(db_name)
                # Check if summary table exists by querying information_schema
                result = self._conn.execute(
                    f"""
                    SELECT COUNT(*) as cnt
                    FROM {alias}.information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'summary'
                    """
                ).df()
                if result["cnt"].iloc[0] > 0:
                    valid_datasets.append(db_name)
            except Exception:
                # Skip databases we can't access or that don't have the schema
                continue

        return valid_datasets

    def list_metrics_by_type(self, db_name: str, metric_type: str) -> list[str]:
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
            raise ValueError(f"Invalid metric_type: '{metric_type}'. Must be 'retrieval' or 'generation'.")

        alias = self.attach_dataset(db_name)
        return (
            self._conn.execute(
                f"SELECT name FROM {alias}.metric WHERE type = $1 ORDER BY name",
                [metric_type],
            )
            .df()["name"]
            .tolist()
        )

    def get_pipeline_details(self, db_name: str, pipeline_name: str) -> dict:
        """Get pipeline config JSONB for metadata display/filtering.

        Args:
            db_name: Name of the PostgreSQL database.
            pipeline_name: Name of the pipeline.

        Returns:
            Dictionary containing pipeline id, name, and config.
            Returns empty dict if pipeline not found.
        """
        alias = self.attach_dataset(db_name)
        df = self._conn.execute(
            f"SELECT id, name, config FROM {alias}.pipeline WHERE name = $1",
            [pipeline_name],
        ).df()

        if df.empty:
            return {}

        row = df.iloc[0]
        return {
            "id": int(row["id"]),
            "name": row["name"],
            "config": row["config"],
        }

    def get_dataset_stats(self, db_name: str) -> dict:
        """Return dataset statistics including query, chunk, and document counts.

        Args:
            db_name: Name of the PostgreSQL database.

        Returns:
            Dictionary with keys: query_count, chunk_count, document_count.
        """
        alias = self.attach_dataset(db_name)

        # Query counts from each table
        stats = {}

        for table, key in [("query", "query_count"), ("chunk", "chunk_count"), ("document", "document_count")]:
            try:
                result = self._conn.execute(f"SELECT COUNT(*) as cnt FROM {alias}.{table}").df()
                stats[key] = int(result["cnt"].iloc[0])
            except Exception:
                stats[key] = 0

        return stats

    # === Multi-Dataset Queries ===

    def compare_across_datasets(
        self, db_names: list[str], pipeline_name: str, metric_name: str
    ) -> pd.DataFrame:
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
            alias = self.attach_dataset(db_name)
            df = self._conn.execute(
                f"""
                SELECT $1 as dataset, s.metric_result as score, s.execution_time as time_ms
                FROM {alias}.summary s
                JOIN {alias}.pipeline p ON s.pipeline_id = p.id
                JOIN {alias}.metric m ON s.metric_id = m.id
                WHERE p.name = $2 AND m.name = $3
                """,
                [db_name, pipeline_name, metric_name],
            ).df()
            results.append(df)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def generate_benchmark_report(
        self, db_names: list[str], pipeline_names: list[str], metric_names: list[str]
    ) -> pd.DataFrame:
        """Generate a comprehensive benchmark report (dataset × pipeline × metric).

        Args:
            db_names: List of database names (each representing a dataset).
            pipeline_names: List of pipeline names to include.
            metric_names: List of metric names to include.

        Returns:
            DataFrame with columns: dataset, pipeline, metric, score
        """
        if not db_names or not pipeline_names or not metric_names:
            return pd.DataFrame()

        results = []
        for db_name in db_names:
            alias = self.attach_dataset(db_name)

            # Create parameterized placeholders (starting at $2 because $1 is db_name)
            pipeline_placeholders = ", ".join(f"${i + 2}" for i in range(len(pipeline_names)))
            metric_placeholders = ", ".join(
                f"${i + 2 + len(pipeline_names)}" for i in range(len(metric_names))
            )

            df = self._conn.execute(
                f"""
                SELECT
                    $1 as dataset,
                    p.name as pipeline,
                    m.name as metric,
                    s.metric_result as score
                FROM {alias}.summary s
                JOIN {alias}.pipeline p ON s.pipeline_id = p.id
                JOIN {alias}.metric m ON s.metric_id = m.id
                WHERE p.name IN ({pipeline_placeholders}) AND m.name IN ({metric_placeholders})
                """,
                [db_name, *pipeline_names, *metric_names],
            ).df()
            results.append(df)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

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
            alias = self.attach_dataset(db_name)

            for metric_name in metric_names:
                # Determine sort order for this metric
                ascending = metric_name in ascending_metrics
                order = "ASC" if ascending else "DESC"

                df = self._conn.execute(
                    f"""
                    SELECT
                        p.name as pipeline,
                        RANK() OVER (ORDER BY s.metric_result {order}) as rank
                    FROM {alias}.summary s
                    JOIN {alias}.pipeline p ON s.pipeline_id = p.id
                    JOIN {alias}.metric m ON s.metric_id = m.id
                    WHERE m.name = $1
                    """,
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
            combined.groupby("pipeline")
            .agg(total_rank=("rank", "sum"), num_rankings=("rank", "count"))
            .reset_index()
        )

        # Compute average rank
        result["avg_rank"] = result["total_rank"] / result["num_rankings"]

        # Sort by total_rank (lower is better)
        result = result.sort_values("total_rank").reset_index(drop=True)

        return result[["pipeline", "total_rank", "avg_rank", "num_rankings"]]

    def get_aggregated_leaderboard(
        self, db_names: list[str], metric_name: str, aggregation: str = "mean"
    ) -> pd.DataFrame:
        """Get pipeline rankings aggregated across multiple datasets.

        Args:
            db_names: List of database names (each representing a dataset).
            metric_name: Name of the metric to rank by.
            aggregation: Aggregation function ('mean', 'median', 'min', 'max', 'sum', 'std').

        Returns:
            DataFrame with columns: pipeline, score (aggregated)

        Raises:
            ValueError: If aggregation is not a valid function name.
        """
        if aggregation not in VALID_AGGREGATIONS:
            raise ValueError(
                f"Invalid aggregation: '{aggregation}'. Must be one of {sorted(VALID_AGGREGATIONS)}."
            )

        if not db_names:
            return pd.DataFrame()

        results = []
        for db_name in db_names:
            alias = self.attach_dataset(db_name)
            df = self._conn.execute(
                f"""
                SELECT $1 as dataset, p.name as pipeline, s.metric_result as score
                FROM {alias}.summary s
                JOIN {alias}.pipeline p ON s.pipeline_id = p.id
                JOIN {alias}.metric m ON s.metric_id = m.id
                WHERE m.name = $2
                """,
                [db_name, metric_name],
            ).df()
            results.append(df)

        df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

        if df.empty:
            return df

        return (
            df.groupby("pipeline")["score"]
            .agg(aggregation)
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"score": f"score_{aggregation}"})
        )

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
