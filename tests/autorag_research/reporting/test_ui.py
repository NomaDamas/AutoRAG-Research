"""Tests for the Gradio leaderboard UI module."""

from unittest.mock import MagicMock, patch

import gradio as gr
import pandas as pd
import pytest

from autorag_research.reporting.ui import (
    create_leaderboard_app,
    format_dataset_stats,
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


class TestFormatDatasetStats:
    """Tests for format_dataset_stats function."""

    @pytest.fixture(autouse=True)
    def setup_mock_service(self):
        """Setup mock service for each test."""
        reset_service()
        with patch("autorag_research.reporting.ui.get_service") as mock_get:
            self.mock_service = MagicMock()
            mock_get.return_value = self.mock_service
            yield
        reset_service()

    def test_format_dataset_stats_returns_formatted_string(self):
        """Test format_dataset_stats returns formatted stats string."""
        self.mock_service.get_dataset_stats.return_value = {
            "query_count": 1000,
            "chunk_count": 50000,
            "document_count": 500,
        }
        result = format_dataset_stats("test_db")
        assert "1000 queries" in result
        assert "50000 chunks" in result
        assert "500 documents" in result

    def test_format_dataset_stats_empty_db_returns_empty(self):
        """Test format_dataset_stats returns empty string for empty db."""
        result = format_dataset_stats("")
        assert result == ""


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

    def test_on_dataset_change_returns_leaderboard_and_stats(self):
        """Test on_dataset_change returns leaderboard DataFrame and stats."""
        expected_df = pd.DataFrame({
            "rank": [1, 2],
            "pipeline": ["bm25", "vector"],
            "recall@5": [0.85, 0.80],
            "ndcg@10": [0.75, 0.70],
            "Average": [0.80, 0.75],
        })
        self.mock_service.get_all_metrics_leaderboard.return_value = expected_df
        self.mock_service.get_dataset_stats.return_value = {
            "query_count": 100,
            "chunk_count": 1000,
            "document_count": 50,
        }

        df, stats_update = on_dataset_change("test_db", "retrieval")

        pd.testing.assert_frame_equal(df, expected_df)
        assert "100 queries" in stats_update["value"]

    def test_on_dataset_change_empty_db_returns_empty(self):
        """Test on_dataset_change returns empty for empty db_name."""
        df, stats_update = on_dataset_change("", "retrieval")
        assert df.empty
        assert stats_update["value"] == ""

    def test_on_metric_type_change_returns_leaderboard(self):
        """Test on_metric_type_change returns leaderboard DataFrame."""
        expected_df = pd.DataFrame({
            "rank": [1],
            "pipeline": ["naive_rag"],
            "bleu": [0.45],
            "rouge": [0.55],
            "Average": [0.50],
        })
        self.mock_service.get_all_metrics_leaderboard.return_value = expected_df

        result = on_metric_type_change("test_db", "generation")

        pd.testing.assert_frame_equal(result, expected_df)

    def test_on_metric_type_change_empty_db_returns_empty(self):
        """Test on_metric_type_change returns empty for empty db_name."""
        result = on_metric_type_change("", "retrieval")
        assert result.empty

    def test_on_refresh_leaderboard_returns_data_and_stats(self):
        """Test on_refresh_leaderboard returns leaderboard and stats."""
        expected_df = pd.DataFrame({
            "rank": [1],
            "pipeline": ["p1"],
            "recall@5": [0.9],
            "Average": [0.9],
        })
        self.mock_service.get_all_metrics_leaderboard.return_value = expected_df
        self.mock_service.get_dataset_stats.return_value = {
            "query_count": 100,
            "chunk_count": 1000,
            "document_count": 50,
        }

        df, stats_update = on_refresh_leaderboard("test_db", "retrieval")

        pd.testing.assert_frame_equal(df, expected_df)
        assert "100 queries" in stats_update["value"]

    def test_on_refresh_leaderboard_empty_db_returns_empty(self):
        """Test on_refresh_leaderboard returns empty for empty db_name."""
        df, stats_update = on_refresh_leaderboard("", "retrieval")
        assert df.empty
        assert stats_update["value"] == ""

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
            mock_service.get_all_metrics_leaderboard.return_value = pd.DataFrame({
                "rank": [1],
                "pipeline": ["bm25"],
                "recall@5": [0.85],
                "Average": [0.85],
            })
            mock_service.get_dataset_stats.return_value = {
                "query_count": 100,
                "chunk_count": 1000,
                "document_count": 50,
            }
            mock_get.return_value = mock_service

            app = create_leaderboard_app()

            assert isinstance(app, gr.Blocks)
            assert app.title == "AutoRAG-Research Leaderboard"
