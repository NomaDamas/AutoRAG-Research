import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from autorag_research.reporting.config import DatabaseConfig
from autorag_research.reporting.service import ReportingService


class TestDatabaseConfig:
    """Tests for DatabaseConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.user == "postgres"
        assert config.password == "postgres"

    def test_custom_values(self):
        """Test that custom values override defaults."""
        config = DatabaseConfig(host="db.example.com", port=5433, user="admin", password="secret")
        assert config.host == "db.example.com"
        assert config.port == 5433
        assert config.user == "admin"
        assert config.password == "secret"

    def test_from_env(self):
        """Test creating config from environment variables."""
        with patch.dict(
            os.environ,
            {
                "POSTGRES_HOST": "env-host",
                "POSTGRES_PORT": "5434",
                "POSTGRES_USER": "env-user",
                "POSTGRES_PASSWORD": "env-pass",
            },
        ):
            config = DatabaseConfig.from_env()
            assert config.host == "env-host"
            assert config.port == 5434
            assert config.user == "env-user"
            assert config.password == "env-pass"

    def test_from_env_defaults(self):
        """Test that from_env uses defaults when env vars are not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = DatabaseConfig.from_env()
            assert config.host == "localhost"
            assert config.port == 5432
            assert config.user == "postgres"
            assert config.password == "postgres"

    def test_get_duckdb_connection_string(self):
        """Test generating DuckDB connection string."""
        config = DatabaseConfig(host="myhost", port=5433, user="myuser", password="mypass")
        conn_str = config.get_duckdb_connection_string("test_db")
        assert conn_str == "postgresql://myuser:mypass@myhost:5433/test_db"


class TestReportingService:
    """Tests for ReportingService class."""

    @pytest.fixture
    def mock_duckdb_connection(self):
        """Create a mock DuckDB connection."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.df.return_value = pd.DataFrame()
        return mock_conn

    @pytest.fixture
    def service_with_mock(self, mock_duckdb_connection):
        """Create ReportingService with mocked DuckDB."""
        with patch("autorag_research.reporting.service.duckdb") as mock_duckdb:
            mock_duckdb.connect.return_value = mock_duckdb_connection
            service = ReportingService(DatabaseConfig())
            yield service, mock_duckdb_connection

    def test_init_installs_postgres_extension(self, service_with_mock):
        """Test that init installs and loads postgres extension."""
        _, mock_conn = service_with_mock
        mock_conn.install_extension.assert_called_once_with("postgres")
        mock_conn.load_extension.assert_called_once_with("postgres")

    def test_attach_dataset(self, service_with_mock):
        """Test attaching a PostgreSQL database."""
        service, mock_conn = service_with_mock
        alias = service.attach_dataset("test-db")

        assert alias == "test_db"
        assert "test_db" in service._attached_dbs
        mock_conn.execute.assert_called()

    def test_attach_dataset_with_custom_alias(self, service_with_mock):
        """Test attaching with custom alias."""
        service, _ = service_with_mock
        alias = service.attach_dataset("test-db", alias="custom_alias")
        assert alias == "custom_alias"
        assert "custom_alias" in service._attached_dbs

    def test_attach_dataset_idempotent(self, service_with_mock):
        """Test that attaching same database twice doesn't duplicate."""
        service, mock_conn = service_with_mock
        initial_call_count = mock_conn.execute.call_count

        service.attach_dataset("test-db")
        first_call_count = mock_conn.execute.call_count

        service.attach_dataset("test-db")
        second_call_count = mock_conn.execute.call_count

        assert second_call_count == first_call_count

    def test_get_leaderboard(self, service_with_mock):
        """Test get_leaderboard generates correct query structure."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame(
            {"rank": [1, 2], "pipeline": ["bm25", "vector"], "score": [0.9, 0.8], "time_ms": [100, 200]}
        )

        df = service.get_leaderboard("test-db", "recall@10", limit=5)

        assert not df.empty
        assert list(df.columns) == ["rank", "pipeline", "score", "time_ms"]

    def test_get_leaderboard_ascending(self, service_with_mock):
        """Test get_leaderboard with ascending order."""
        service, mock_conn = service_with_mock
        service.get_leaderboard("test-db", "latency", ascending=True)

        # Check that ASC is in the executed query
        call_args = mock_conn.execute.call_args_list[-1]
        query = call_args[0][0]
        assert "ASC" in query

    def test_compare_pipelines(self, service_with_mock):
        """Test compare_pipelines generates pivot table."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame(
            {
                "pipeline": ["bm25", "bm25", "vector", "vector"],
                "metric": ["recall@10", "ndcg@10", "recall@10", "ndcg@10"],
                "score": [0.9, 0.85, 0.88, 0.87],
            }
        )

        df = service.compare_pipelines("test-db", ["bm25", "vector"], ["recall@10", "ndcg@10"])

        assert "ndcg@10" in df.columns
        assert "recall@10" in df.columns

    def test_compare_pipelines_empty_result(self, service_with_mock):
        """Test compare_pipelines with no results."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame()

        df = service.compare_pipelines("test-db", ["nonexistent"], ["metric"])
        assert df.empty

    def test_list_pipelines(self, service_with_mock):
        """Test list_pipelines returns pipeline names."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame({"name": ["bm25", "vector", "hybrid"]})

        pipelines = service.list_pipelines("test-db")
        assert pipelines == ["bm25", "vector", "hybrid"]

    def test_list_metrics(self, service_with_mock):
        """Test list_metrics returns metric names."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame({"name": ["recall@10", "ndcg@10", "mrr"]})

        metrics = service.list_metrics("test-db")
        assert metrics == ["recall@10", "ndcg@10", "mrr"]

    def test_compare_across_datasets(self, service_with_mock):
        """Test cross-dataset comparison."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame(
            {"dataset": ["db1", "db2"], "score": [0.9, 0.85], "time_ms": [100, 150]}
        )

        df = service.compare_across_datasets(["db1", "db2"], "bm25", "recall@10")

        assert "dataset" in df.columns
        assert "score" in df.columns

    def test_generate_benchmark_report(self, service_with_mock):
        """Test comprehensive benchmark report generation."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame(
            {
                "dataset": ["db1", "db1", "db2", "db2"],
                "pipeline": ["bm25", "vector", "bm25", "vector"],
                "metric": ["recall@10", "recall@10", "recall@10", "recall@10"],
                "score": [0.9, 0.88, 0.85, 0.87],
            }
        )

        df = service.generate_benchmark_report(["db1", "db2"], ["bm25", "vector"], ["recall@10"])

        assert "dataset" in df.columns
        assert "pipeline" in df.columns
        assert "metric" in df.columns
        assert "score" in df.columns

    def test_get_aggregated_leaderboard(self, service_with_mock):
        """Test aggregated leaderboard across datasets."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame(
            {
                "dataset": ["db1", "db1", "db2", "db2"],
                "pipeline": ["bm25", "vector", "bm25", "vector"],
                "score": [0.9, 0.88, 0.85, 0.87],
            }
        )

        df = service.get_aggregated_leaderboard(["db1", "db2"], "recall@10")

        assert "pipeline" in df.columns
        assert "score_mean" in df.columns

    def test_get_aggregated_leaderboard_empty(self, service_with_mock):
        """Test aggregated leaderboard with no results."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame()

        df = service.get_aggregated_leaderboard(["db1"], "metric")
        assert df.empty

    def test_context_manager(self, service_with_mock):
        """Test context manager protocol."""
        service, mock_conn = service_with_mock

        with service as svc:
            assert svc is service

        mock_conn.close.assert_called_once()

    def test_close(self, service_with_mock):
        """Test close method."""
        service, mock_conn = service_with_mock
        service.close()
        mock_conn.close.assert_called_once()

    def test_attach_dataset_invalid_alias_raises_error(self, service_with_mock):
        """Test that invalid alias raises ValueError."""
        service, _ = service_with_mock
        with pytest.raises(ValueError, match="Invalid alias"):
            service.attach_dataset("test-db", alias="invalid;alias")

    def test_attach_dataset_invalid_alias_with_quotes(self, service_with_mock):
        """Test that alias with quotes raises ValueError."""
        service, _ = service_with_mock
        with pytest.raises(ValueError, match="Invalid alias"):
            service.attach_dataset("test-db", alias="alias'injection")

    def test_attach_dataset_invalid_alias_starting_with_number(self, service_with_mock):
        """Test that alias starting with number raises ValueError."""
        service, _ = service_with_mock
        with pytest.raises(ValueError, match="Invalid alias"):
            service.attach_dataset("test-db", alias="123invalid")

    def test_get_aggregated_leaderboard_invalid_aggregation(self, service_with_mock):
        """Test that invalid aggregation raises ValueError."""
        service, _ = service_with_mock
        with pytest.raises(ValueError, match="Invalid aggregation"):
            service.get_aggregated_leaderboard(["db1"], "metric", aggregation="invalid")

    def test_get_aggregated_leaderboard_valid_aggregations(self, service_with_mock):
        """Test that all valid aggregations are accepted."""
        from autorag_research.reporting.service import VALID_AGGREGATIONS

        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame(
            {"dataset": ["db1"], "pipeline": ["bm25"], "score": [0.9]}
        )

        for agg in VALID_AGGREGATIONS:
            # Should not raise
            service.get_aggregated_leaderboard(["db1"], "metric", aggregation=agg)

    def test_compare_pipelines_empty_inputs(self, service_with_mock):
        """Test compare_pipelines with empty inputs returns empty DataFrame."""
        service, _ = service_with_mock

        assert service.compare_pipelines("db", [], ["metric"]).empty
        assert service.compare_pipelines("db", ["pipeline"], []).empty

    def test_compare_across_datasets_empty_db_names(self, service_with_mock):
        """Test compare_across_datasets with empty db_names returns empty DataFrame."""
        service, _ = service_with_mock
        assert service.compare_across_datasets([], "pipeline", "metric").empty

    def test_generate_benchmark_report_empty_inputs(self, service_with_mock):
        """Test generate_benchmark_report with empty inputs returns empty DataFrame."""
        service, _ = service_with_mock

        assert service.generate_benchmark_report([], ["p"], ["m"]).empty
        assert service.generate_benchmark_report(["db"], [], ["m"]).empty
        assert service.generate_benchmark_report(["db"], ["p"], []).empty

    def test_get_aggregated_leaderboard_empty_db_names(self, service_with_mock):
        """Test get_aggregated_leaderboard with empty db_names returns empty DataFrame."""
        service, _ = service_with_mock
        assert service.get_aggregated_leaderboard([], "metric").empty

    def test_init_closes_connection_on_extension_error(self):
        """Test that connection is closed if extension loading fails."""
        with patch("autorag_research.reporting.service.duckdb") as mock_duckdb:
            mock_conn = MagicMock()
            mock_duckdb.connect.return_value = mock_conn
            mock_conn.install_extension.side_effect = RuntimeError("Extension error")

            with pytest.raises(RuntimeError, match="Extension error"):
                ReportingService(DatabaseConfig())

            mock_conn.close.assert_called_once()

    # === Tests for New Leaderboard Methods ===

    def test_list_available_datasets(self, service_with_mock):
        """Test listing available datasets with AutoRAG schema."""
        service, mock_conn = service_with_mock

        # Mock pg_database query
        def execute_side_effect(query, params=None):
            result = MagicMock()
            if "pg_database" in query:
                result.df.return_value = pd.DataFrame({"datname": ["dataset1", "dataset2", "no_schema_db"]})
            elif "information_schema.tables" in query:
                # dataset1 and dataset2 have summary table, no_schema_db doesn't
                if "dataset1" in str(service._attached_dbs) or "dataset2" in str(service._attached_dbs):
                    result.df.return_value = pd.DataFrame({"cnt": [1]})
                else:
                    result.df.return_value = pd.DataFrame({"cnt": [0]})
            return result

        mock_conn.execute.side_effect = execute_side_effect

        # We need to manually track which db is being checked
        # Simplify: just return all have summary table
        mock_conn.execute.side_effect = None
        mock_conn.execute.return_value.df.return_value = pd.DataFrame({"datname": ["dataset1", "dataset2"]})

        # Call the method (note: actual implementation is more complex)
        # For unit test, we verify it attempts correct queries
        service.list_available_datasets()

        # Verify pg_database was queried
        calls = [str(call) for call in mock_conn.execute.call_args_list]
        assert any("pg_database" in str(call) or "ATTACH" in str(call) for call in calls)

    def test_list_metrics_by_type_retrieval(self, service_with_mock):
        """Test filtering metrics by retrieval type."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame({"name": ["recall@10", "ndcg@10", "mrr"]})

        metrics = service.list_metrics_by_type("test-db", "retrieval")

        assert metrics == ["recall@10", "ndcg@10", "mrr"]
        # Verify type filter was used
        call_args = mock_conn.execute.call_args_list[-1]
        query = call_args[0][0]
        params = call_args[0][1] if len(call_args[0]) > 1 else []
        assert "type" in query
        assert "retrieval" in params

    def test_list_metrics_by_type_generation(self, service_with_mock):
        """Test filtering metrics by generation type."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame({"name": ["bleu", "rouge", "bertscore"]})

        metrics = service.list_metrics_by_type("test-db", "generation")

        assert metrics == ["bleu", "rouge", "bertscore"]
        call_args = mock_conn.execute.call_args_list[-1]
        params = call_args[0][1] if len(call_args[0]) > 1 else []
        assert "generation" in params

    def test_list_metrics_by_type_invalid_raises_error(self, service_with_mock):
        """Test that invalid metric type raises ValueError."""
        service, _ = service_with_mock
        with pytest.raises(ValueError, match="Invalid metric_type"):
            service.list_metrics_by_type("test-db", "invalid")

    def test_get_pipeline_details(self, service_with_mock):
        """Test getting pipeline details with config."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame(
            {"id": [1], "name": ["bm25"], "config": [{"k": 10, "b": 0.75}]}
        )

        details = service.get_pipeline_details("test-db", "bm25")

        assert details["id"] == 1
        assert details["name"] == "bm25"
        assert details["config"] == {"k": 10, "b": 0.75}

    def test_get_pipeline_details_not_found(self, service_with_mock):
        """Test getting pipeline details when pipeline doesn't exist."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame()

        details = service.get_pipeline_details("test-db", "nonexistent")

        assert details == {}

    def test_get_dataset_stats(self, service_with_mock):
        """Test getting dataset statistics."""
        service, mock_conn = service_with_mock

        # Mock different counts for different tables
        call_count = [0]

        def execute_side_effect(query, params=None):
            result = MagicMock()
            if "ATTACH" in query:
                return result
            # Return different counts based on which table
            counts = [100, 1000, 50]  # query, chunk, document
            result.df.return_value = pd.DataFrame({"cnt": [counts[call_count[0] % 3]]})
            call_count[0] += 1
            return result

        mock_conn.execute.side_effect = execute_side_effect

        stats = service.get_dataset_stats("test-db")

        assert "query_count" in stats
        assert "chunk_count" in stats
        assert "document_count" in stats

    def test_get_borda_count_leaderboard(self, service_with_mock):
        """Test Borda Count leaderboard computation."""
        service, mock_conn = service_with_mock

        # Mock ranking results for two datasets and two metrics
        def execute_side_effect(query, params=None):
            result = MagicMock()
            if "ATTACH" in query:
                return result
            # Return rankings
            result.df.return_value = pd.DataFrame(
                {"pipeline": ["bm25", "vector", "hybrid"], "rank": [1, 2, 3]}
            )
            return result

        mock_conn.execute.side_effect = execute_side_effect

        df = service.get_borda_count_leaderboard(["db1", "db2"], ["recall@10", "ndcg@10"])

        assert "pipeline" in df.columns
        assert "total_rank" in df.columns
        assert "avg_rank" in df.columns
        assert "num_rankings" in df.columns

    def test_get_borda_count_leaderboard_with_ascending_metrics(self, service_with_mock):
        """Test Borda Count with ascending metrics (lower is better)."""
        service, mock_conn = service_with_mock

        def execute_side_effect(query, params=None):
            result = MagicMock()
            if "ATTACH" in query:
                return result
            # Check if ASC is used for latency metric
            if "ASC" in query:
                result.df.return_value = pd.DataFrame(
                    {"pipeline": ["vector", "bm25"], "rank": [1, 2]}
                )
            else:
                result.df.return_value = pd.DataFrame(
                    {"pipeline": ["bm25", "vector"], "rank": [1, 2]}
                )
            return result

        mock_conn.execute.side_effect = execute_side_effect

        df = service.get_borda_count_leaderboard(
            ["db1"], ["recall@10", "latency"], ascending_metrics={"latency"}
        )

        assert not df.empty

    def test_get_borda_count_leaderboard_empty_inputs(self, service_with_mock):
        """Test Borda Count with empty inputs returns empty DataFrame."""
        service, _ = service_with_mock

        assert service.get_borda_count_leaderboard([], ["metric"]).empty
        assert service.get_borda_count_leaderboard(["db"], []).empty

    def test_get_borda_count_leaderboard_no_results(self, service_with_mock):
        """Test Borda Count when no results from queries."""
        service, mock_conn = service_with_mock

        def execute_side_effect(query, params=None):
            result = MagicMock()
            if "ATTACH" in query:
                return result
            result.df.return_value = pd.DataFrame()
            return result

        mock_conn.execute.side_effect = execute_side_effect

        df = service.get_borda_count_leaderboard(["db1"], ["metric"])
        assert df.empty
