"""Tests for ConfigResolver class and related helpers."""

import pytest
from omegaconf import OmegaConf

from autorag_research.cli.config_resolver import ConfigResolver


class TestConfigResolver:
    """Tests for ConfigResolver."""

    def test_resolve_configs_single_pipeline(self, tmp_path):
        """Test resolving a single pipeline config."""
        # Setup: Create config directory structure
        (tmp_path / "pipelines" / "retrieval").mkdir(parents=True)
        (tmp_path / "pipelines" / "retrieval" / "bm25.yaml").write_text(
            "_target_: autorag_research.pipelines.retrieval.bm25.BM25PipelineConfig\nname: bm25\ntop_k: 10\n"
        )

        resolver = ConfigResolver(config_dir=tmp_path)
        config_dict = OmegaConf.create({"retrieval": ["bm25"]})

        result = resolver.resolve_configs("pipelines", config_dict)

        assert len(result) == 1
        assert result[0].name == "bm25"
        assert result[0].top_k == 10

    def test_resolve_configs_multiple_pipelines(self, tmp_path):
        """Test resolving multiple pipeline configs from different subdirs."""
        # Setup
        (tmp_path / "pipelines" / "retrieval").mkdir(parents=True)
        (tmp_path / "pipelines" / "generation").mkdir(parents=True)
        (tmp_path / "pipelines" / "retrieval" / "bm25.yaml").write_text("name: bm25\ntop_k: 10\n")
        (tmp_path / "pipelines" / "generation" / "basic_rag.yaml").write_text(
            "name: basic_rag\nllm_model: gpt-4o-mini\n"
        )

        resolver = ConfigResolver(config_dir=tmp_path)
        config_dict = OmegaConf.create({
            "retrieval": ["bm25"],
            "generation": ["basic_rag"],
        })

        result = resolver.resolve_configs("pipelines", config_dict)

        assert len(result) == 2
        names = {cfg.name for cfg in result}
        assert names == {"bm25", "basic_rag"}

    def test_resolve_configs_multiple_in_same_subdir(self, tmp_path):
        """Test resolving multiple configs from the same subdirectory."""
        (tmp_path / "metrics" / "retrieval").mkdir(parents=True)
        (tmp_path / "metrics" / "retrieval" / "recall.yaml").write_text("name: recall\nk: 10\n")
        (tmp_path / "metrics" / "retrieval" / "ndcg.yaml").write_text("name: ndcg\nk: 10\n")

        resolver = ConfigResolver(config_dir=tmp_path)
        config_dict = OmegaConf.create({"retrieval": ["recall", "ndcg"]})

        result = resolver.resolve_configs("metrics", config_dict)

        assert len(result) == 2
        names = {cfg.name for cfg in result}
        assert names == {"recall", "ndcg"}

    def test_resolve_configs_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing config."""
        (tmp_path / "pipelines" / "retrieval").mkdir(parents=True)

        resolver = ConfigResolver(config_dir=tmp_path)
        config_dict = OmegaConf.create({"retrieval": ["nonexistent"]})

        with pytest.raises(FileNotFoundError) as exc_info:
            resolver.resolve_configs("pipelines", config_dict)

        assert "nonexistent.yaml" in str(exc_info.value)

    def test_resolve_configs_single_string_value(self, tmp_path):
        """Test resolving when value is a single string instead of list."""
        (tmp_path / "pipelines" / "retrieval").mkdir(parents=True)
        (tmp_path / "pipelines" / "retrieval" / "bm25.yaml").write_text("name: bm25\n")

        resolver = ConfigResolver(config_dir=tmp_path)
        # Single string instead of list
        config_dict = OmegaConf.create({"retrieval": "bm25"})

        result = resolver.resolve_configs("pipelines", config_dict)

        assert len(result) == 1
        assert result[0].name == "bm25"

    def test_resolve_pipelines_convenience_method(self, tmp_path):
        """Test resolve_pipelines convenience method."""
        (tmp_path / "pipelines" / "retrieval").mkdir(parents=True)
        (tmp_path / "pipelines" / "retrieval" / "bm25.yaml").write_text("name: bm25\n")

        resolver = ConfigResolver(config_dir=tmp_path)
        config_dict = OmegaConf.create({"retrieval": ["bm25"]})

        result = resolver.resolve_pipelines(config_dict)

        assert len(result) == 1
        assert result[0].name == "bm25"

    def test_resolve_metrics_convenience_method(self, tmp_path):
        """Test resolve_metrics convenience method."""
        (tmp_path / "metrics" / "retrieval").mkdir(parents=True)
        (tmp_path / "metrics" / "retrieval" / "recall.yaml").write_text("name: recall\n")

        resolver = ConfigResolver(config_dir=tmp_path)
        config_dict = OmegaConf.create({"retrieval": ["recall"]})

        result = resolver.resolve_metrics(config_dict)

        assert len(result) == 1
        assert result[0].name == "recall"
