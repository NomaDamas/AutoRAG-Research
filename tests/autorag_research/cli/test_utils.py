"""Tests for autorag_research.cli.utils module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import autorag_research.cli as cli
from autorag_research.cli.utils import (
    discover_configs,
    discover_embedding_configs,
    discover_metrics,
    discover_pipelines,
    get_config_dir,
    health_check_embedding,
    load_embedding_model,
    setup_logging,
)


class TestDiscoverConfigs:
    """Tests for discover_configs function."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Returns empty dict for empty directory."""
        result = discover_configs(tmp_path)

        assert result == {}

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Returns empty dict for non-existent directory."""
        nonexistent = tmp_path / "does_not_exist"

        with pytest.raises(FileNotFoundError):
            discover_configs(nonexistent)

    def test_yaml_with_description(self, tmp_path: Path) -> None:
        """Uses description field from YAML file."""
        (tmp_path / "test.yaml").write_text("description: Test pipeline config")

        result = discover_configs(tmp_path)

        assert "test" in result
        assert result["test"] == "Test pipeline config"

    def test_yaml_falls_back_to_target(self, tmp_path: Path) -> None:
        """Falls back to _target_ if no description field."""
        (tmp_path / "dense.yaml").write_text("_target_: some.module.DenseRetrieval\nk: 10")

        result = discover_configs(tmp_path)

        assert "dense" in result
        assert "some.module.DenseRetrieval" in result["dense"]

    def test_yaml_without_description_or_target(self, tmp_path: Path) -> None:
        """Uses filename as fallback if no description or target."""
        (tmp_path / "simple.yaml").write_text("key: value")

        result = discover_configs(tmp_path)

        assert "simple" in result

    def test_returns_sorted_by_name(self, tmp_path: Path) -> None:
        """Results are sorted alphabetically by name."""
        (tmp_path / "zebra.yaml").write_text("description: Z config")
        (tmp_path / "alpha.yaml").write_text("description: A config")
        (tmp_path / "middle.yaml").write_text("description: M config")

        result = discover_configs(tmp_path)

        keys = list(result.keys())
        assert keys == sorted(keys)

    def test_handles_invalid_yaml(self, tmp_path: Path) -> None:
        """Handles malformed YAML gracefully."""
        (tmp_path / "invalid.yaml").write_text("invalid: yaml: content: [")

        result = discover_configs(tmp_path)

        assert "invalid" in result
        assert "Error" in result["invalid"] or "error" in result["invalid"].lower() or result["invalid"] != ""


class TestDiscoverPipelines:
    """Tests for discover_pipelines function using real configs."""

    def test_discover_pipelines_finds_real_configs(self, real_config_path: Path) -> None:
        """discover_pipelines finds bm25 and basic_rag in real configs/pipelines/."""
        result = discover_pipelines()

        assert "generation" in result
        assert "retrieval" in result
        assert "bm25" in result["retrieval"]
        assert "basic_rag" in result["generation"]


class TestDiscoverMetrics:
    """Tests for discover_metrics function using real configs."""

    def test_discover_metrics_finds_real_configs(self, real_config_path: Path) -> None:
        """discover_metrics finds ndcg and recall in real configs/metrics/."""
        result = discover_metrics()

        assert "generation" in result
        assert "retrieval" in result
        assert "ndcg" in result["retrieval"]
        assert "rouge" in result["generation"]


class TestDiscoverEmbeddingConfigs:
    """Tests for discover_embedding_configs function using real configs."""

    def test_discover_embedding_configs_finds_real_configs(self, real_config_path: Path) -> None:
        """discover_embedding_configs finds openai configs in real configs/embedding/."""
        result = discover_embedding_configs()

        assert "openai-small" in result
        assert "openai-large" in result
        assert "openai-like" in result


class TestGetConfigDir:
    """Tests for get_config_dir function."""

    def test_returns_cli_config_path_when_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns cli.CONFIG_PATH when it's set."""
        monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path)

        result = get_config_dir()

        assert result == tmp_path

    def test_falls_back_to_cwd_configs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falls back to cwd/configs when CONFIG_PATH is None."""
        monkeypatch.setattr(cli, "CONFIG_PATH", None)

        result = get_config_dir()

        assert result == Path.cwd() / "configs"


class TestLoadEmbeddingModel:
    """Tests for load_embedding_model function."""

    def test_raises_file_not_found_for_missing_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises FileNotFoundError when config doesn't exist."""
        monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path)

        with pytest.raises(FileNotFoundError):
            load_embedding_model("nonexistent")

    @patch("hydra.utils.instantiate")
    def test_raises_type_error_for_wrong_type(self, mock_instantiate: MagicMock, real_config_path: Path) -> None:
        """Raises TypeError when instantiated object is not BaseEmbedding."""
        mock_instantiate.return_value = "not an embedding"

        with pytest.raises(TypeError, match="BaseEmbedding"):
            load_embedding_model("openai-small")

    def test_returns_embedding_instance(self, real_config_path: Path) -> None:
        """Returns BaseEmbedding instance when config is valid."""
        from llama_index.core.base.embeddings.base import BaseEmbedding

        result = load_embedding_model("mock")

        assert isinstance(result, BaseEmbedding)


class TestHealthCheckEmbedding:
    """Tests for health_check_embedding function.

    Uses mock embedding model to avoid real API calls.
    """

    def test_returns_dimension_on_success(self) -> None:
        """Returns embedding dimension on success."""
        from llama_index.core.embeddings.mock_embed_model import MockEmbedding

        mock_embedding = MockEmbedding(384)

        result = health_check_embedding(mock_embedding)

        assert result == 384

    def test_raises_on_embedding_failure(self) -> None:
        """Raises EmbeddingNotSetError when embedding fails."""
        from autorag_research.exceptions import EmbeddingNotSetError

        mock_model = MagicMock()
        mock_model.get_text_embedding.side_effect = Exception("API Error")

        with pytest.raises(EmbeddingNotSetError):
            health_check_embedding(mock_model)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_verbose_sets_debug_level(self) -> None:
        """verbose=True sets DEBUG level on root logger."""
        import logging

        # Reset root logger handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        setup_logging(verbose=True)

        # basicConfig sets root logger level
        assert root_logger.level == logging.DEBUG

    def test_default_sets_info_level(self) -> None:
        """verbose=False sets INFO level on root logger."""
        import logging

        # Reset root logger handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        setup_logging(verbose=False)

        assert root_logger.level == logging.INFO
