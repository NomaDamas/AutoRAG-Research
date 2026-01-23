"""Tests for CLI configuration modules."""

from autorag_research.cli.configs.datasets import AVAILABLE_DATASETS
from autorag_research.cli.configs.db import DatabaseConfig
from autorag_research.cli.configs.metrics import AVAILABLE_METRICS
from autorag_research.cli.configs.pipelines import AVAILABLE_PIPELINES


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.user == "postgres"
        assert config.database == "autorag_research"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DatabaseConfig(
            host="db.example.com",
            port=5433,
            user="custom_user",
            database="custom_db",
        )
        assert config.host == "db.example.com"
        assert config.port == 5433
        assert config.user == "custom_user"
        assert config.database == "custom_db"


class TestAvailableResources:
    """Tests for available resource dictionaries."""

    def test_available_datasets_not_empty(self):
        """Test that available datasets dictionary is populated."""
        assert len(AVAILABLE_DATASETS) > 0
        # Now uses ingestor names instead of individual datasets
        assert "beir" in AVAILABLE_DATASETS
        assert "mrtydi" in AVAILABLE_DATASETS

    def test_available_pipelines_not_empty(self):
        """Test that available pipelines dictionary is populated."""
        assert len(AVAILABLE_PIPELINES) > 0
        assert "bm25_baseline" in AVAILABLE_PIPELINES

    def test_available_metrics_not_empty(self):
        """Test that available metrics dictionary is populated."""
        assert len(AVAILABLE_METRICS) > 0
        assert "recall" in AVAILABLE_METRICS
        assert "ndcg" in AVAILABLE_METRICS
