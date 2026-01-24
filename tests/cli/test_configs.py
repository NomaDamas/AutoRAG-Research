"""Tests for CLI configuration modules."""

from autorag_research.cli.configs.db import DatabaseConfig
from autorag_research.cli.utils import discover_metrics, discover_pipelines
from autorag_research.data.registry import discover_ingestors


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

    def test_discover_ingestors_not_empty(self):
        """Test that ingestors are discovered via decorator registration."""
        ingestors = discover_ingestors()
        assert len(ingestors) > 0
        # Check expected ingestors are registered
        assert "beir" in ingestors
        assert "mrtydi" in ingestors
        assert "ragbench" in ingestors
        assert "mteb" in ingestors
        assert "bright" in ingestors
        assert "vidore" in ingestors
        assert "vidorev2" in ingestors

    def test_ingestor_has_params(self):
        """Test that ingestors have auto-extracted parameters."""
        ingestors = discover_ingestors()

        # BEIR should have dataset_name param with Literal choices
        beir = ingestors["beir"]
        assert beir.description == "BEIR benchmark datasets for information retrieval"
        assert len(beir.params) > 0
        dataset_param = beir.params[0]
        assert dataset_param.name == "dataset_name"
        assert dataset_param.choices is not None
        assert "msmarco" in dataset_param.choices
        assert "scifact" in dataset_param.choices

        # MrTyDi should have language param with default
        mrtydi = ingestors["mrtydi"]
        language_param = mrtydi.params[0]
        assert language_param.name == "language"
        assert language_param.default == "english"
        assert not language_param.required

    def test_available_pipelines_not_empty(self):
        """Test that available pipelines are discovered from YAML configs."""
        pipelines = discover_pipelines()
        assert len(pipelines) > 0
        assert "bm25" in pipelines

    def test_available_metrics_not_empty(self):
        """Test that available metrics are discovered from YAML configs."""
        metrics = discover_metrics()
        assert len(metrics) > 0
        assert "recall" in metrics
        assert "ndcg" in metrics
