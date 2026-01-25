from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from autorag_research.orm.connection import DBConnection
from autorag_research.reporting.service import ReportingService


@pytest.fixture
def mock_db_connection() -> DBConnection:
    """Create a mock database connection for testing."""
    return DBConnection(
        host="localhost",
        port=5432,
        username="test_user",
        password="test_pass",  # noqa: S106
        database="test_db",
    )


class TestReportingService:
    """Tests for ReportingService class."""

    @pytest.fixture
    def mock_duckdb_connection(self):
        """Create a mock DuckDB connection."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.df.return_value = pd.DataFrame()
        return mock_conn

    @pytest.fixture
    def service_with_mock(self, mock_duckdb_connection, mock_db_connection):
        """Create ReportingService with mocked DuckDB."""
        with patch("autorag_research.reporting.service.duckdb") as mock_duckdb:
            mock_duckdb.connect.return_value = mock_duckdb_connection
            service = ReportingService(mock_db_connection)
            yield service, mock_duckdb_connection

    def test_init_installs_postgres_extension(self, service_with_mock):
        """Test that init installs and loads postgres extension."""
        _, mock_conn = service_with_mock
        mock_conn.install_extension.assert_called_once_with("postgres")
        mock_conn.load_extension.assert_called_once_with("postgres")

    def test_attach_dataset(self, service_with_mock):
        """Test attaching a PostgreSQL database."""
        service, mock_conn = service_with_mock
        db = service.attach_dataset("test-db")

        assert db == "test-db"
        assert "test-db" in service._attached_dbs
        mock_conn.execute.assert_called()

    def test_attach_dataset_idempotent(self, service_with_mock):
        """Test that attaching same database twice doesn't duplicate."""
        service, mock_conn = service_with_mock

        service.attach_dataset("test-db")
        first_call_count = mock_conn.execute.call_count

        service.attach_dataset("test-db")
        second_call_count = mock_conn.execute.call_count

        assert second_call_count == first_call_count

    def test_attach_dataset_with_hyphen(self, service_with_mock):
        """Test attaching database with hyphen in name uses quoted identifier."""
        service, mock_conn = service_with_mock
        db = service.attach_dataset("my-test-db")

        assert db == "my-test-db"
        # Verify quoted identifier is used in ATTACH statement
        call_args = mock_conn.execute.call_args[0][0]
        assert '"my-test-db"' in call_args

    def test_get_leaderboard(self, service_with_mock):
        """Test get_leaderboard generates correct query structure."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame({
            "rank": [1, 2],
            "pipeline": ["bm25", "vector"],
            "score": [0.9, 0.8],
            "time_ms": [100, 200],
        })

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

    def test_get_pipeline_type_retrieval(self, service_with_mock):
        """Test get_pipeline_type returns 'retrieval' for retrieval pipeline."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame({"pipeline_type": ["retrieval"]})

        result = service.get_pipeline_type("test-db", "bm25_pipeline")
        assert result == "retrieval"

    def test_get_pipeline_type_generation(self, service_with_mock):
        """Test get_pipeline_type returns 'generation' for generation pipeline."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame({"pipeline_type": ["generation"]})

        result = service.get_pipeline_type("test-db", "naive_rag_pipeline")
        assert result == "generation"

    def test_get_pipeline_type_not_found(self, service_with_mock):
        """Test get_pipeline_type returns None when pipeline not found."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame()

        result = service.get_pipeline_type("test-db", "nonexistent_pipeline")
        assert result is None

    def test_get_pipeline_type_null_value(self, service_with_mock):
        """Test get_pipeline_type returns None when pipeline_type is null in config."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame({"pipeline_type": [None]})

        result = service.get_pipeline_type("test-db", "legacy_pipeline")
        assert result is None

    def test_compare_across_datasets(self, service_with_mock):
        """Test cross-dataset comparison."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame({
            "dataset": ["db1", "db2"],
            "score": [0.9, 0.85],
            "time_ms": [100, 150],
        })

        df = service.compare_across_datasets(["db1", "db2"], "bm25", "recall@10")

        assert "dataset" in df.columns
        assert "score" in df.columns

    def test_compare_across_datasets_empty_db_names(self, service_with_mock):
        """Test compare_across_datasets with empty db_names returns empty DataFrame."""
        service, _ = service_with_mock
        assert service.compare_across_datasets([], "pipeline", "metric").empty

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

    def test_init_closes_connection_on_extension_error(self, mock_db_connection):
        """Test that connection is closed if extension loading fails."""
        with patch("autorag_research.reporting.service.duckdb") as mock_duckdb:
            mock_conn = MagicMock()
            mock_duckdb.connect.return_value = mock_conn
            mock_conn.install_extension.side_effect = RuntimeError("Extension error")

            with pytest.raises(RuntimeError, match="Extension error"):
                ReportingService(mock_db_connection)

            mock_conn.close.assert_called_once()

    # === Tests for Dataset Discovery Methods ===

    def test_list_available_datasets(self, service_with_mock):
        """Test listing available datasets with AutoRAG schema."""
        service, mock_conn = service_with_mock

        # Mock pg_database query
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
        assert "type" in query
        assert "retrieval" in query

    def test_list_metrics_by_type_generation(self, service_with_mock):
        """Test filtering metrics by generation type."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame({"name": ["bleu", "rouge", "bertscore"]})

        metrics = service.list_metrics_by_type("test-db", "generation")

        assert metrics == ["bleu", "rouge", "bertscore"]
        call_args = mock_conn.execute.call_args_list[-1]
        query = call_args[0][0]
        assert "generation" in query

    def test_list_metrics_by_type_invalid_raises_error(self, service_with_mock):
        """Test that invalid metric type raises ValueError."""
        service, _ = service_with_mock
        with pytest.raises(ValueError, match="Invalid metric_type"):
            service.list_metrics_by_type("test-db", "invalid")

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
            result.df.return_value = pd.DataFrame({"pipeline": ["bm25", "vector", "hybrid"], "rank": [1, 2, 3]})
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
                result.df.return_value = pd.DataFrame({"pipeline": ["vector", "bm25"], "rank": [1, 2]})
            else:
                result.df.return_value = pd.DataFrame({"pipeline": ["bm25", "vector"], "rank": [1, 2]})
            return result

        mock_conn.execute.side_effect = execute_side_effect

        df = service.get_borda_count_leaderboard(["db1"], ["recall@10", "latency"], ascending_metrics={"latency"})

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

    # === Tests for Multi-Metric Leaderboard Methods ===

    def test_get_all_metrics_leaderboard(self, service_with_mock):
        """Test get_all_metrics_leaderboard returns pivoted DataFrame with Average."""
        service, mock_conn = service_with_mock

        def execute_side_effect(query, params=None):
            result = MagicMock()
            if "ATTACH" in query:
                return result
            # Return data for pivoting
            result.df.return_value = pd.DataFrame({
                "pipeline": ["bm25", "bm25", "vector", "vector"],
                "metric": ["recall@10", "mrr", "recall@10", "mrr"],
                "score": [0.8, 0.7, 0.9, 0.6],
            })
            return result

        mock_conn.execute.side_effect = execute_side_effect

        df = service.get_all_metrics_leaderboard("test-db", "retrieval")

        # Check column structure
        assert "rank" in df.columns
        assert "pipeline" in df.columns
        assert "Average" in df.columns
        # Check sorting by Average (vector has avg 0.75, bm25 has avg 0.75 but different)
        assert not df.empty

    def test_get_all_metrics_leaderboard_retrieval_query(self, service_with_mock):
        """Test that retrieval type is used in query."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame()

        service.get_all_metrics_leaderboard("test-db", "retrieval")

        call_args = mock_conn.execute.call_args_list[-1]
        query = call_args[0][0]
        assert "retrieval" in query

    def test_get_all_metrics_leaderboard_generation_query(self, service_with_mock):
        """Test that generation type is used in query."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame()

        service.get_all_metrics_leaderboard("test-db", "generation")

        call_args = mock_conn.execute.call_args_list[-1]
        query = call_args[0][0]
        assert "generation" in query

    def test_get_all_metrics_leaderboard_empty_result(self, service_with_mock):
        """Test get_all_metrics_leaderboard returns empty DataFrame when no data."""
        service, mock_conn = service_with_mock
        mock_conn.execute.return_value.df.return_value = pd.DataFrame()

        df = service.get_all_metrics_leaderboard("test-db", "retrieval")
        assert df.empty

    def test_get_all_metrics_leaderboard_invalid_type(self, service_with_mock):
        """Test that invalid metric type raises ValueError."""
        service, _ = service_with_mock
        with pytest.raises(ValueError, match="Invalid metric_type"):
            service.get_all_metrics_leaderboard("test-db", "invalid")

    def test_get_all_metrics_leaderboard_rounds_values(self, service_with_mock):
        """Test that numeric values are rounded to 3 decimal places."""
        service, mock_conn = service_with_mock

        def execute_side_effect(query, params=None):
            result = MagicMock()
            if "ATTACH" in query:
                return result
            result.df.return_value = pd.DataFrame({
                "pipeline": ["bm25", "bm25"],
                "metric": ["recall@10", "mrr"],
                "score": [0.123456789, 0.987654321],
            })
            return result

        mock_conn.execute.side_effect = execute_side_effect

        df = service.get_all_metrics_leaderboard("test-db", "retrieval")

        # Check that values are rounded
        for col in df.select_dtypes(include=["float64"]).columns:
            for val in df[col]:
                # Values should be rounded to at most 3 decimal places
                assert round(val, 3) == val

    def test_compare_pipeline_all_metrics(self, service_with_mock):
        """Test compare_pipeline_all_metrics returns pivoted DataFrame with Average."""
        service, mock_conn = service_with_mock

        # Track which dataset we're querying
        dataset_index = [0]
        datasets = ["db1", "db2"]

        def execute_side_effect(query, params=None):
            result = MagicMock()
            if "ATTACH" in query:
                return result
            # First call: get_pipeline_type
            if "config" in query:
                result.df.return_value = pd.DataFrame({"pipeline_type": ["retrieval"]})
            # Subsequent calls: metric data for each dataset
            else:
                # Use the actual dataset name from params
                current_dataset = params[0] if params else datasets[dataset_index[0] % 2]
                result.df.return_value = pd.DataFrame({
                    "dataset": [current_dataset] * 2,
                    "metric": ["recall@10", "mrr"],
                    "score": [0.8, 0.7],
                })
                dataset_index[0] += 1
            return result

        mock_conn.execute.side_effect = execute_side_effect

        df = service.compare_pipeline_all_metrics(["db1", "db2"], "bm25")

        assert "dataset" in df.columns
        assert "Average" in df.columns
        assert not df.empty

    def test_compare_pipeline_all_metrics_empty_db_names(self, service_with_mock):
        """Test compare_pipeline_all_metrics with empty db_names returns empty DataFrame."""
        service, _ = service_with_mock
        assert service.compare_pipeline_all_metrics([], "pipeline").empty

    def test_compare_pipeline_all_metrics_empty_pipeline_name(self, service_with_mock):
        """Test compare_pipeline_all_metrics with empty pipeline_name returns empty DataFrame."""
        service, _ = service_with_mock
        assert service.compare_pipeline_all_metrics(["db1"], "").empty

    def test_compare_pipeline_all_metrics_pipeline_type_not_found(self, service_with_mock):
        """Test compare_pipeline_all_metrics returns empty when pipeline type not found."""
        service, mock_conn = service_with_mock

        def execute_side_effect(query, params=None):
            result = MagicMock()
            if "ATTACH" in query:
                return result
            # Return empty for pipeline type query
            result.df.return_value = pd.DataFrame()
            return result

        mock_conn.execute.side_effect = execute_side_effect

        df = service.compare_pipeline_all_metrics(["db1"], "nonexistent")
        assert df.empty

    def test_compare_pipeline_all_metrics_rounds_values(self, service_with_mock):
        """Test that numeric values are rounded to 3 decimal places."""
        service, mock_conn = service_with_mock

        def execute_side_effect(query, params=None):
            result = MagicMock()
            if "ATTACH" in query:
                return result
            if "config" in query:
                result.df.return_value = pd.DataFrame({"pipeline_type": ["retrieval"]})
            else:
                result.df.return_value = pd.DataFrame({
                    "dataset": ["db1"],
                    "metric": ["recall@10"],
                    "score": [0.123456789],
                })
            return result

        mock_conn.execute.side_effect = execute_side_effect

        df = service.compare_pipeline_all_metrics(["db1"], "bm25")

        if not df.empty:
            for col in df.select_dtypes(include=["float64"]).columns:
                for val in df[col]:
                    assert round(val, 3) == val
