"""Shared fixtures for CLI tests."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

import autorag_research.cli as cli


@pytest.fixture
def cli_runner() -> CliRunner:
    """Return a Typer CliRunner for testing commands."""
    return CliRunner()


@pytest.fixture
def mock_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set cli.CONFIG_PATH to a temp directory and return it."""
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path)
    return tmp_path


@pytest.fixture
def reset_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset cli.CONFIG_PATH to None before test."""
    monkeypatch.setattr(cli, "CONFIG_PATH", None)


@pytest.fixture
def mock_db_config() -> dict[str, str | int]:
    """Return mock database configuration values."""
    return {
        "host": "localhost",
        "port": 5432,
        "user": "test_user",
        "password": "test_pass",
        "database": "test_db",
    }


@pytest.fixture
def mock_ingestor_meta() -> Mock:
    """Return a mock IngestorMeta for testing."""
    from autorag_research.data.registry import IngestorMeta, ParamMeta

    return IngestorMeta(
        name="test_ingestor",
        ingestor_class=Mock,
        description="Test ingestor for unit tests",
        params=[
            ParamMeta(
                name="dataset_name",
                cli_option="dataset-name",
                param_type=str,
                choices=["option_a", "option_b", "option_c"],
                required=True,
                default=None,
                help="Dataset name to ingest",
                is_list=False,
            ),
            ParamMeta(
                name="batch_size",
                cli_option="batch-size",
                param_type=int,
                choices=None,
                required=False,
                default=100,
                help="Batch size for ingestion",
                is_list=False,
            ),
        ],
    )


@pytest.fixture
def sample_config_dir(tmp_path: Path) -> Path:
    """Create a sample config directory structure for testing."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Create pipeline configs
    pipelines_dir = config_dir / "pipelines"
    pipelines_dir.mkdir()
    (pipelines_dir / "bm25.yaml").write_text("description: BM25 retrieval pipeline\n_target_: some.module.BM25")
    (pipelines_dir / "dense.yaml").write_text("_target_: some.module.DenseRetrieval")

    # Create metric configs
    metrics_dir = config_dir / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "ndcg.yaml").write_text("description: NDCG metric\n_target_: some.module.NDCG")

    # Create embedding configs (note: directory is "embedding" not "embeddings")
    embedding_dir = config_dir / "embedding"
    embedding_dir.mkdir()
    (embedding_dir / "openai-small.yaml").write_text("description: OpenAI small embedding\n_target_: some.Embedding")

    # Create db.yaml
    (config_dir / "db.yaml").write_text(
        """host: localhost
port: 5432
user: autorag
password: secret
database: autorag_db
"""
    )

    return config_dir
