"""Tests for plugin CLI commands."""

from unittest.mock import patch

from typer.testing import CliRunner

from autorag_research.cli.app import app
from autorag_research.plugin_registry import SyncResult

runner = CliRunner()


class TestPluginSync:
    """Tests for `autorag-research plugin sync` command."""

    def test_no_plugins_found(self, tmp_path):
        """Shows 'no plugins found' when nothing is installed."""
        with (
            patch("autorag_research.cli.utils.get_config_dir", return_value=tmp_path / "configs"),
            patch("autorag_research.plugin_registry.sync_plugin_configs", return_value=[]),
            patch("autorag_research.plugin_registry._discover_plugin_configs_uncached", return_value=[]),
        ):
            result = runner.invoke(app, ["plugin", "sync"])

        assert result.exit_code == 0
        assert "No plugins found" in result.output

    def test_sync_copies_files(self, tmp_path):
        """Shows copied files in output."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        mock_results = [
            SyncResult(
                config_name="es_search",
                plugin_name="elasticsearch",
                destination=config_dir / "pipelines" / "retrieval" / "es_search.yaml",
                copied=True,
                reason="copied",
            ),
        ]

        with (
            patch("autorag_research.cli.utils.get_config_dir", return_value=config_dir),
            patch("autorag_research.plugin_registry._discover_plugin_configs_uncached", return_value=mock_results),
            patch("autorag_research.plugin_registry.sync_plugin_configs", return_value=mock_results),
        ):
            result = runner.invoke(app, ["plugin", "sync"])

        assert result.exit_code == 0
        assert "Copied 1" in result.output
        assert "es_search" in result.output

    def test_sync_shows_skipped(self, tmp_path):
        """Shows skipped files in output."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        mock_results = [
            SyncResult(
                config_name="es_search",
                plugin_name="elasticsearch",
                destination=config_dir / "pipelines" / "retrieval" / "es_search.yaml",
                copied=False,
                reason="already exists",
            ),
        ]

        with (
            patch("autorag_research.cli.utils.get_config_dir", return_value=config_dir),
            patch("autorag_research.plugin_registry.sync_plugin_configs", return_value=mock_results),
        ):
            result = runner.invoke(app, ["plugin", "sync"])

        assert result.exit_code == 0
        assert "Skipped 1" in result.output


class TestPluginCreate:
    """Tests for `autorag-research plugin create` command."""

    def test_create_retrieval(self, tmp_path, monkeypatch):
        """Scaffolds correct retrieval plugin structure."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["plugin", "create", "my_search", "--type", "retrieval"])

        assert result.exit_code == 0
        plugin_dir = tmp_path / "my_search_plugin"
        assert plugin_dir.exists()
        assert (plugin_dir / "pyproject.toml").exists()
        assert (plugin_dir / "src" / "my_search_plugin" / "__init__.py").exists()
        assert (plugin_dir / "src" / "my_search_plugin" / "pipeline.py").exists()
        assert (plugin_dir / "src" / "my_search_plugin" / "retrieval" / "my_search.yaml").exists()
        assert (plugin_dir / "tests" / "test_my_search.py").exists()

        # Check pyproject.toml has entry_points
        toml_content = (plugin_dir / "pyproject.toml").read_text()
        assert "autorag_research.pipelines" in toml_content
        assert "my_search" in toml_content

        # Check pipeline.py has correct base class
        pipeline_content = (plugin_dir / "src" / "my_search_plugin" / "pipeline.py").read_text()
        assert "BaseRetrievalPipelineConfig" in pipeline_content
        assert "BaseRetrievalPipeline" in pipeline_content
        assert "MySearchPipelineConfig" in pipeline_content

    def test_create_generation(self, tmp_path, monkeypatch):
        """Scaffolds correct generation plugin structure."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["plugin", "create", "my_gen", "--type", "generation"])

        assert result.exit_code == 0
        plugin_dir = tmp_path / "my_gen_plugin"
        assert plugin_dir.exists()

        pipeline_content = (plugin_dir / "src" / "my_gen_plugin" / "pipeline.py").read_text()
        assert "BaseGenerationPipelineConfig" in pipeline_content
        assert "BaseGenerationPipeline" in pipeline_content
        assert "MyGenPipelineConfig" in pipeline_content

        # Check YAML has generation-specific fields
        yaml_content = (plugin_dir / "src" / "my_gen_plugin" / "generation" / "my_gen.yaml").read_text()
        assert "retrieval_pipeline_name" in yaml_content
        assert "llm" in yaml_content

    def test_create_metric_retrieval(self, tmp_path, monkeypatch):
        """Scaffolds correct retrieval metric plugin structure."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["plugin", "create", "my_metric", "--type", "metric_retrieval"])

        assert result.exit_code == 0
        plugin_dir = tmp_path / "my_metric_plugin"
        assert plugin_dir.exists()

        # Check metric.py instead of pipeline.py
        metric_content = (plugin_dir / "src" / "my_metric_plugin" / "metric.py").read_text()
        assert "BaseRetrievalMetricConfig" in metric_content
        assert "MyMetricMetricConfig" in metric_content

        # Check YAML is in retrieval subdirectory
        assert (plugin_dir / "src" / "my_metric_plugin" / "retrieval" / "my_metric.yaml").exists()

        # Check pyproject.toml has metrics entry point
        toml_content = (plugin_dir / "pyproject.toml").read_text()
        assert "autorag_research.metrics" in toml_content

    def test_create_metric_generation(self, tmp_path, monkeypatch):
        """Scaffolds correct generation metric plugin structure."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["plugin", "create", "custom_score", "--type", "metric_generation"])

        assert result.exit_code == 0
        plugin_dir = tmp_path / "custom_score_plugin"

        metric_content = (plugin_dir / "src" / "custom_score_plugin" / "metric.py").read_text()
        assert "BaseGenerationMetricConfig" in metric_content

        # Check YAML is in generation subdirectory
        assert (plugin_dir / "src" / "custom_score_plugin" / "generation" / "custom_score.yaml").exists()

    def test_create_invalid_type(self, tmp_path, monkeypatch):
        """Exits with error for invalid --type."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["plugin", "create", "bad", "--type", "invalid"])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_create_missing_type(self, tmp_path, monkeypatch):
        """Exits with error when --type is not provided."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["plugin", "create", "foo"])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "--type" in result.output

    def test_create_existing_dir(self, tmp_path, monkeypatch):
        """Exits with error if plugin directory already exists."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "my_search_plugin").mkdir()

        result = runner.invoke(app, ["plugin", "create", "my_search", "--type", "retrieval"])

        assert result.exit_code == 1
        assert "already exists" in result.output
