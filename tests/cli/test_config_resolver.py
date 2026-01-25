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
            "_target_: autorag_research.pipelines.retrieval.bm25.BM25PipelineConfig\n"
            "name: bm25\n"
            "top_k: 10\n"
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
        (tmp_path / "pipelines" / "retrieval" / "bm25.yaml").write_text(
            "name: bm25\ntop_k: 10\n"
        )
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
        (tmp_path / "metrics" / "retrieval" / "recall.yaml").write_text(
            "name: recall\nk: 10\n"
        )
        (tmp_path / "metrics" / "retrieval" / "ndcg.yaml").write_text(
            "name: ndcg\nk: 10\n"
        )

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
        (tmp_path / "pipelines" / "retrieval" / "bm25.yaml").write_text(
            "name: bm25\n"
        )

        resolver = ConfigResolver(config_dir=tmp_path)
        # Single string instead of list
        config_dict = OmegaConf.create({"retrieval": "bm25"})

        result = resolver.resolve_configs("pipelines", config_dict)

        assert len(result) == 1
        assert result[0].name == "bm25"

    def test_resolve_pipelines_convenience_method(self, tmp_path):
        """Test resolve_pipelines convenience method."""
        (tmp_path / "pipelines" / "retrieval").mkdir(parents=True)
        (tmp_path / "pipelines" / "retrieval" / "bm25.yaml").write_text(
            "name: bm25\n"
        )

        resolver = ConfigResolver(config_dir=tmp_path)
        config_dict = OmegaConf.create({"retrieval": ["bm25"]})

        result = resolver.resolve_pipelines(config_dict)

        assert len(result) == 1
        assert result[0].name == "bm25"

    def test_resolve_metrics_convenience_method(self, tmp_path):
        """Test resolve_metrics convenience method."""
        (tmp_path / "metrics" / "retrieval").mkdir(parents=True)
        (tmp_path / "metrics" / "retrieval" / "recall.yaml").write_text(
            "name: recall\n"
        )

        resolver = ConfigResolver(config_dir=tmp_path)
        config_dict = OmegaConf.create({"retrieval": ["recall"]})

        result = resolver.resolve_metrics(config_dict)

        assert len(result) == 1
        assert result[0].name == "recall"

    def test_load_db_config(self, tmp_path):
        """Test loading database config."""
        (tmp_path / "db").mkdir(parents=True)
        (tmp_path / "db" / "default.yaml").write_text(
            "host: localhost\n"
            "port: 5432\n"
            "database: test_db\n"
        )

        resolver = ConfigResolver(config_dir=tmp_path)

        result = resolver.load_db_config("default")

        assert result.host == "localhost"
        assert result.port == 5432
        assert result.database == "test_db"

    def test_load_db_config_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing db config."""
        (tmp_path / "db").mkdir(parents=True)

        resolver = ConfigResolver(config_dir=tmp_path)

        with pytest.raises(FileNotFoundError) as exc_info:
            resolver.load_db_config("nonexistent")

        assert "nonexistent.yaml" in str(exc_info.value)

    def test_load_db_config_list_raises_type_error(self, tmp_path):
        """Test that TypeError is raised when db config is a list."""
        (tmp_path / "db").mkdir(parents=True)
        (tmp_path / "db" / "bad.yaml").write_text("- item1\n- item2\n")

        resolver = ConfigResolver(config_dir=tmp_path)

        with pytest.raises(TypeError) as exc_info:
            resolver.load_db_config("bad")

        assert "must be a mapping" in str(exc_info.value)


class TestIsDictSyntax:
    """Tests for _is_dict_syntax helper."""

    def test_new_dict_syntax_with_list(self):
        """Test detection of new dict syntax with lists."""
        from autorag_research.cli.commands.run import _is_dict_syntax

        cfg = OmegaConf.create({
            "pipelines": {
                "retrieval": ["bm25"],
                "generation": ["basic_rag"],
            }
        })

        assert _is_dict_syntax(cfg) is True

    def test_new_dict_syntax_with_single_string(self):
        """Test detection when values are strings (not lists)."""
        from autorag_research.cli.commands.run import _is_dict_syntax

        cfg = OmegaConf.create({
            "pipelines": {
                "retrieval": "bm25",
            }
        })

        # Single strings don't match the list pattern, but the dict doesn't have _target_
        assert _is_dict_syntax(cfg) is True

    def test_legacy_syntax_with_target(self):
        """Test detection of legacy Hydra syntax with _target_."""
        from autorag_research.cli.commands.run import _is_dict_syntax

        cfg = OmegaConf.create({
            "pipelines": [
                {
                    "_target_": "autorag_research.pipelines.retrieval.bm25.BM25PipelineConfig",
                    "name": "bm25",
                }
            ]
        })

        assert _is_dict_syntax(cfg) is False

    def test_empty_pipelines(self):
        """Test with empty pipelines dict."""
        from autorag_research.cli.commands.run import _is_dict_syntax

        cfg = OmegaConf.create({"pipelines": {}})

        # Empty dict has no values to check
        assert _is_dict_syntax(cfg) is False

    def test_no_pipelines_key(self):
        """Test with missing pipelines key."""
        from autorag_research.cli.commands.run import _is_dict_syntax

        cfg = OmegaConf.create({"db_name": "test"})

        assert _is_dict_syntax(cfg) is False
