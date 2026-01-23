"""Tests for the Gradio leaderboard UI module."""

from unittest.mock import MagicMock, patch

import gradio as gr
import pandas as pd
import pytest

from autorag_research.reporting.ui import (
    create_leaderboard_app,
    fetch_all_metrics,
    fetch_borda_ranking,
    fetch_cross_dataset,
    fetch_dataset_stats,
    fetch_datasets,
    fetch_leaderboard,
    fetch_metrics,
    fetch_pipelines,
    get_service,
    on_dataset_change,
    on_datasets_select_for_metrics,
    on_datasets_select_for_pipelines,
    on_metric_type_change,
    on_refresh_leaderboard,
    reset_service,
)


class TestServiceManagement:
    """Tests for service singleton management."""

    def test_get_service_creates_singleton(self):
        """Test that get_service creates a singleton instance."""
        with patch("autorag_research.reporting.ui.ReportingService") as mock_cls:
            mock_service = MagicMock()
            mock_cls.return_value = mock_service

            # Reset before test
            reset_service()

            service1 = get_service()
            service2 = get_service()

            assert service1 is service2
            mock_cls.assert_called_once()

            # Cleanup
            reset_service()

    def test_reset_service_closes_and_clears(self):
        """Test that reset_service closes the service and clears singleton."""
        with patch("autorag_research.reporting.ui.ReportingService") as mock_cls:
            mock_service = MagicMock()
            mock_cls.return_value = mock_service

            reset_service()
            get_service()
            reset_service()

            mock_service.close.assert_called_once()


class TestDataFetchers:
    """Tests for data fetching functions."""

    @pytest.fixture(autouse=True)
    def setup_mock_service(self):
        """Setup mock service for each test."""
        reset_service()
        with patch("autorag_research.reporting.ui.get_service") as mock_get:
            self.mock_service = MagicMock()
            mock_get.return_value = self.mock_service
            yield
        reset_service()

    def test_fetch_datasets_returns_list(self):
        """Test fetch_datasets returns available datasets."""
        self.mock_service.list_available_datasets.return_value = ["db1", "db2"]
        result = fetch_datasets()
        assert result == ["db1", "db2"]

    def test_fetch_datasets_handles_error(self):
        """Test fetch_datasets returns empty list on error."""
        self.mock_service.list_available_datasets.side_effect = Exception("Connection failed")
        result = fetch_datasets()
        assert result == []

    def test_fetch_metrics_returns_filtered_list(self):
        """Test fetch_metrics returns metrics filtered by type."""
        self.mock_service.list_metrics_by_type.return_value = ["recall@5", "ndcg@10"]
        result = fetch_metrics("test_db", "retrieval")
        assert result == ["recall@5", "ndcg@10"]
        self.mock_service.list_metrics_by_type.assert_called_once_with("test_db", "retrieval")

    def test_fetch_metrics_empty_db_returns_empty(self):
        """Test fetch_metrics returns empty list when db_name is empty."""
        result = fetch_metrics("", "retrieval")
        assert result == []

    def test_fetch_metrics_handles_error(self):
        """Test fetch_metrics returns empty list on error."""
        self.mock_service.list_metrics_by_type.side_effect = Exception("Error")
        result = fetch_metrics("test_db", "retrieval")
        assert result == []

    def test_fetch_pipelines_returns_list(self):
        """Test fetch_pipelines returns pipeline names."""
        self.mock_service.list_pipelines.return_value = ["pipeline1", "pipeline2"]
        result = fetch_pipelines("test_db")
        assert result == ["pipeline1", "pipeline2"]

    def test_fetch_pipelines_empty_db_returns_empty(self):
        """Test fetch_pipelines returns empty list when db_name is empty."""
        result = fetch_pipelines("")
        assert result == []

    def test_fetch_all_metrics_returns_list(self):
        """Test fetch_all_metrics returns all metrics."""
        self.mock_service.list_metrics.return_value = ["metric1", "metric2"]
        result = fetch_all_metrics("test_db")
        assert result == ["metric1", "metric2"]

    def test_fetch_all_metrics_empty_db_returns_empty(self):
        """Test fetch_all_metrics returns empty list when db_name is empty."""
        result = fetch_all_metrics("")
        assert result == []

    def test_fetch_leaderboard_returns_dataframe(self):
        """Test fetch_leaderboard returns DataFrame."""
        expected_df = pd.DataFrame({
            "rank": [1, 2],
            "pipeline": ["p1", "p2"],
            "score": [0.85, 0.80],
            "time_ms": [100, 120],
        })
        self.mock_service.get_leaderboard.return_value = expected_df
        result = fetch_leaderboard("test_db", "recall@5")
        pd.testing.assert_frame_equal(result, expected_df)

    def test_fetch_leaderboard_empty_inputs_returns_empty(self):
        """Test fetch_leaderboard returns empty DataFrame for empty inputs."""
        result = fetch_leaderboard("", "recall@5")
        assert result.empty
        result = fetch_leaderboard("test_db", "")
        assert result.empty

    def test_fetch_leaderboard_handles_error(self):
        """Test fetch_leaderboard returns empty DataFrame on error."""
        self.mock_service.get_leaderboard.side_effect = Exception("Error")
        result = fetch_leaderboard("test_db", "recall@5")
        assert result.empty

    def test_fetch_dataset_stats_returns_formatted_string(self):
        """Test fetch_dataset_stats returns formatted stats string."""
        self.mock_service.get_dataset_stats.return_value = {
            "query_count": 1000,
            "chunk_count": 50000,
            "document_count": 500,
        }
        result = fetch_dataset_stats("test_db")
        assert "1000 queries" in result
        assert "50000 chunks" in result
        assert "500 documents" in result

    def test_fetch_dataset_stats_empty_db_returns_empty(self):
        """Test fetch_dataset_stats returns empty string for empty db."""
        result = fetch_dataset_stats("")
        assert result == ""

    def test_fetch_cross_dataset_returns_dataframe(self):
        """Test fetch_cross_dataset returns comparison DataFrame."""
        expected_df = pd.DataFrame({
            "dataset": ["db1", "db2"],
            "score": [0.85, 0.80],
            "time_ms": [100, 120],
        })
        self.mock_service.compare_across_datasets.return_value = expected_df
        result = fetch_cross_dataset(["db1", "db2"], "pipeline1", "recall@5")
        pd.testing.assert_frame_equal(result, expected_df)

    def test_fetch_cross_dataset_empty_inputs_returns_empty(self):
        """Test fetch_cross_dataset returns empty DataFrame for empty inputs."""
        result = fetch_cross_dataset([], "pipeline1", "recall@5")
        assert result.empty
        result = fetch_cross_dataset(["db1"], "", "recall@5")
        assert result.empty
        result = fetch_cross_dataset(["db1"], "pipeline1", "")
        assert result.empty

    def test_fetch_borda_ranking_returns_dataframe(self):
        """Test fetch_borda_ranking returns Borda ranking DataFrame."""
        expected_df = pd.DataFrame({
            "pipeline": ["p1", "p2"],
            "total_rank": [12, 18],
            "avg_rank": [1.5, 2.25],
            "num_rankings": [8, 8],
        })
        self.mock_service.get_borda_count_leaderboard.return_value = expected_df
        result = fetch_borda_ranking(["db1", "db2"], ["recall@5", "ndcg@10"])
        pd.testing.assert_frame_equal(result, expected_df)

    def test_fetch_borda_ranking_empty_inputs_returns_empty(self):
        """Test fetch_borda_ranking returns empty DataFrame for empty inputs."""
        result = fetch_borda_ranking([], ["recall@5"])
        assert result.empty
        result = fetch_borda_ranking(["db1"], [])
        assert result.empty


class TestUIUpdateHandlers:
    """Tests for UI update handler functions."""

    @pytest.fixture(autouse=True)
    def setup_mock_service(self):
        """Setup mock service for each test."""
        reset_service()
        with patch("autorag_research.reporting.ui.get_service") as mock_get:
            self.mock_service = MagicMock()
            mock_get.return_value = self.mock_service
            yield
        reset_service()

    def test_on_dataset_change_updates_metrics_and_stats(self):
        """Test on_dataset_change returns updated metrics and stats."""
        self.mock_service.list_metrics_by_type.return_value = ["recall@5", "ndcg@10"]
        self.mock_service.get_dataset_stats.return_value = {
            "query_count": 100,
            "chunk_count": 1000,
            "document_count": 50,
        }

        metric_update, stats_update = on_dataset_change("test_db", "retrieval")

        assert metric_update["choices"] == ["recall@5", "ndcg@10"]
        assert metric_update["value"] == "recall@5"
        assert "100 queries" in stats_update["value"]

    def test_on_metric_type_change_updates_metrics(self):
        """Test on_metric_type_change returns updated metrics."""
        self.mock_service.list_metrics_by_type.return_value = ["bleu", "rouge"]

        result = on_metric_type_change("test_db", "generation")

        assert result["choices"] == ["bleu", "rouge"]
        assert result["value"] == "bleu"

    def test_on_refresh_leaderboard_returns_data_and_stats(self):
        """Test on_refresh_leaderboard returns leaderboard and stats."""
        expected_df = pd.DataFrame({"rank": [1], "pipeline": ["p1"], "score": [0.9], "time_ms": [50]})
        self.mock_service.get_leaderboard.return_value = expected_df
        self.mock_service.get_dataset_stats.return_value = {
            "query_count": 100,
            "chunk_count": 1000,
            "document_count": 50,
        }

        df, stats_update = on_refresh_leaderboard("test_db", "recall@5")

        pd.testing.assert_frame_equal(df, expected_df)
        assert "100 queries" in stats_update["value"]

    def test_on_datasets_select_for_pipelines_updates_dropdown(self):
        """Test on_datasets_select_for_pipelines returns updated pipeline dropdown."""
        self.mock_service.list_pipelines.return_value = ["pipeline1", "pipeline2"]

        result = on_datasets_select_for_pipelines(["db1", "db2"])

        assert result["choices"] == ["pipeline1", "pipeline2"]
        assert result["value"] == "pipeline1"

    def test_on_datasets_select_for_pipelines_empty_returns_empty(self):
        """Test on_datasets_select_for_pipelines returns empty for no datasets."""
        result = on_datasets_select_for_pipelines([])

        assert result["choices"] == []
        assert result["value"] is None

    def test_on_datasets_select_for_metrics_updates_checkbox(self):
        """Test on_datasets_select_for_metrics returns updated metrics checkbox."""
        self.mock_service.list_metrics.return_value = ["metric1", "metric2"]

        result = on_datasets_select_for_metrics(["db1"])

        assert result["choices"] == ["metric1", "metric2"]
        assert result["value"] == []

    def test_on_datasets_select_for_metrics_empty_returns_empty(self):
        """Test on_datasets_select_for_metrics returns empty for no datasets."""
        result = on_datasets_select_for_metrics([])

        assert result["choices"] == []
        assert result["value"] == []


class TestAppCreation:
    """Tests for app creation."""

    def test_create_leaderboard_app_returns_blocks(self):
        """Test that create_leaderboard_app returns a Gradio Blocks instance."""
        with patch("autorag_research.reporting.ui.get_service") as mock_get:
            mock_service = MagicMock()
            mock_service.list_available_datasets.return_value = []
            mock_get.return_value = mock_service

            app = create_leaderboard_app()

            assert isinstance(app, gr.Blocks)

    def test_create_leaderboard_app_with_datasets(self):
        """Test app creation with available datasets."""
        with patch("autorag_research.reporting.ui.get_service") as mock_get:
            mock_service = MagicMock()
            mock_service.list_available_datasets.return_value = ["test_db1", "test_db2"]
            mock_service.list_metrics_by_type.return_value = ["recall@5"]
            mock_service.get_dataset_stats.return_value = {
                "query_count": 100,
                "chunk_count": 1000,
                "document_count": 50,
            }
            mock_get.return_value = mock_service

            app = create_leaderboard_app()

            assert isinstance(app, gr.Blocks)
            assert app.title == "AutoRAG Leaderboard"
