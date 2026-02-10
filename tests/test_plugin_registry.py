"""Tests for plugin_registry module."""

import types
from unittest.mock import MagicMock, patch

import pytest

from autorag_research.plugin_registry import (
    PluginConfigInfo,
    discover_plugin_configs,
    sync_plugin_configs,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear lru_cache before each test."""
    discover_plugin_configs.cache_clear()
    yield
    discover_plugin_configs.cache_clear()


def _mock_uncached_for_category(target_category, infos):
    """Return a side_effect function that returns infos only for the matching category."""

    def side_effect(group, category):
        if category == target_category:
            return infos
        return []

    return side_effect


class TestDiscoverPluginConfigs:
    """Tests for discover_plugin_configs()."""

    def test_no_plugins_returns_empty(self):
        """Returns empty list when no plugins are installed."""
        with patch("autorag_research.plugin_registry.entry_points", return_value=[]):
            result = discover_plugin_configs("autorag_research.pipelines", "pipelines")
        assert result == []

    def test_with_mock_flat_layout(self, tmp_path):
        """Discovers YAML from flat plugin layout (module/config.yaml)."""
        yaml_file = tmp_path / "es_search.yaml"
        yaml_file.write_text('_target_: my_plugin.EsPipelineConfig\ndescription: "ES search"\nname: es_search\n')

        mock_module = types.ModuleType("fake_plugin")
        mock_ep = MagicMock()
        mock_ep.name = "elasticsearch"
        mock_ep.load.return_value = mock_module

        mock_traversable = MagicMock()
        mock_yaml = MagicMock()
        mock_yaml.name = "es_search.yaml"
        mock_yaml.is_dir.return_value = False
        mock_yaml.read_text.return_value = yaml_file.read_text()
        mock_yaml.__str__ = lambda self: str(yaml_file)
        mock_traversable.iterdir.return_value = [mock_yaml]

        with (
            patch("autorag_research.plugin_registry.entry_points", return_value=[mock_ep]),
            patch("autorag_research.plugin_registry.files", return_value=mock_traversable),
        ):
            result = discover_plugin_configs("autorag_research.pipelines", "pipelines")

        assert len(result) == 1
        assert result[0].plugin_name == "elasticsearch"
        assert result[0].config_name == "es_search"
        assert result[0].description == "ES search"
        assert result[0].category == "pipelines"
        assert result[0].subcategory is None

    def test_nested_layout(self, tmp_path):
        """Discovers YAMLs in nested subdirectories (module/retrieval/config.yaml)."""
        retrieval_dir = tmp_path / "retrieval"
        retrieval_dir.mkdir()
        yaml_file = retrieval_dir / "es_search.yaml"
        yaml_file.write_text('description: "Nested ES"\n')

        mock_module = types.ModuleType("fake_plugin")
        mock_ep = MagicMock()
        mock_ep.name = "elasticsearch"
        mock_ep.load.return_value = mock_module

        mock_yaml = MagicMock()
        mock_yaml.name = "es_search.yaml"
        mock_yaml.read_text.return_value = yaml_file.read_text()
        mock_yaml.__str__ = lambda self: str(yaml_file)

        mock_subdir = MagicMock()
        mock_subdir.name = "retrieval"
        mock_subdir.is_dir.return_value = True
        mock_subdir.iterdir.return_value = [mock_yaml]

        mock_traversable = MagicMock()
        mock_traversable.iterdir.return_value = [mock_subdir]

        with (
            patch("autorag_research.plugin_registry.entry_points", return_value=[mock_ep]),
            patch("autorag_research.plugin_registry.files", return_value=mock_traversable),
        ):
            result = discover_plugin_configs("autorag_research.pipelines", "pipelines")

        assert len(result) == 1
        assert result[0].subcategory == "retrieval"
        assert result[0].config_name == "es_search"

    def test_broken_plugin_does_not_break_others(self, tmp_path):
        """One broken plugin doesn't prevent others from loading."""
        yaml_file = tmp_path / "good.yaml"
        yaml_file.write_text('description: "Good plugin"\n')

        broken_ep = MagicMock()
        broken_ep.name = "broken"
        broken_ep.load.side_effect = ImportError("broken!")

        good_module = types.ModuleType("good_plugin")
        good_ep = MagicMock()
        good_ep.name = "good"
        good_ep.load.return_value = good_module

        mock_yaml = MagicMock()
        mock_yaml.name = "good.yaml"
        mock_yaml.is_dir.return_value = False
        mock_yaml.read_text.return_value = yaml_file.read_text()
        mock_yaml.__str__ = lambda self: str(yaml_file)

        mock_traversable = MagicMock()
        mock_traversable.iterdir.return_value = [mock_yaml]

        with (
            patch("autorag_research.plugin_registry.entry_points", return_value=[broken_ep, good_ep]),
            patch("autorag_research.plugin_registry.files", return_value=mock_traversable),
        ):
            result = discover_plugin_configs("autorag_research.pipelines", "pipelines")

        assert len(result) == 1
        assert result[0].plugin_name == "good"


class TestSyncPluginConfigs:
    """Tests for sync_plugin_configs()."""

    def test_copies_to_correct_dirs(self, tmp_path):
        """YAML files are copied to configs/pipelines/{subcategory}/."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        yaml_source = tmp_path / "source_es.yaml"
        yaml_source.write_text('description: "ES search"\nname: es_search\n')

        info = PluginConfigInfo(
            plugin_name="elasticsearch",
            config_name="es_search",
            description="ES search",
            source_path=yaml_source,
            subcategory="retrieval",
            category="pipelines",
        )

        with patch(
            "autorag_research.plugin_registry._discover_plugin_configs_uncached",
            side_effect=_mock_uncached_for_category("pipelines", [info]),
        ):
            results = sync_plugin_configs(config_dir)

        copied = [r for r in results if r.copied]
        assert len(copied) == 1
        assert copied[0].config_name == "es_search"

        dest = config_dir / "pipelines" / "retrieval" / "es_search.yaml"
        assert dest.exists()
        assert "ES search" in dest.read_text()

    def test_skips_existing_files(self, tmp_path):
        """Existing files are not overwritten."""
        config_dir = tmp_path / "configs"
        (config_dir / "pipelines" / "retrieval").mkdir(parents=True)
        existing = config_dir / "pipelines" / "retrieval" / "es_search.yaml"
        existing.write_text("user customized content\n")

        yaml_source = tmp_path / "source_es.yaml"
        yaml_source.write_text('description: "Plugin version"\n')

        info = PluginConfigInfo(
            plugin_name="elasticsearch",
            config_name="es_search",
            description="Plugin version",
            source_path=yaml_source,
            subcategory="retrieval",
            category="pipelines",
        )

        with patch(
            "autorag_research.plugin_registry._discover_plugin_configs_uncached",
            side_effect=_mock_uncached_for_category("pipelines", [info]),
        ):
            results = sync_plugin_configs(config_dir)

        skipped = [r for r in results if not r.copied]
        assert len(skipped) == 1
        assert skipped[0].reason == "already exists"

        # Original content preserved
        assert existing.read_text() == "user customized content\n"

    def test_creates_subdirectories(self, tmp_path):
        """Creates missing subdirectories during sync."""
        config_dir = tmp_path / "configs"

        yaml_source = tmp_path / "source_metric.yaml"
        yaml_source.write_text('description: "Custom metric"\n')

        info = PluginConfigInfo(
            plugin_name="my_metrics",
            config_name="custom_score",
            description="Custom metric",
            source_path=yaml_source,
            subcategory="generation",
            category="metrics",
        )

        with patch(
            "autorag_research.plugin_registry._discover_plugin_configs_uncached",
            side_effect=_mock_uncached_for_category("metrics", [info]),
        ):
            results = sync_plugin_configs(config_dir)

        assert len(results) == 1
        assert results[0].copied
        dest = config_dir / "metrics" / "generation" / "custom_score.yaml"
        assert dest.exists()

    def test_no_subcategory(self, tmp_path):
        """Configs without subcategory go directly in category dir."""
        config_dir = tmp_path / "configs"

        yaml_source = tmp_path / "source.yaml"
        yaml_source.write_text('description: "Flat config"\n')

        info = PluginConfigInfo(
            plugin_name="flat_plugin",
            config_name="flat_config",
            description="Flat config",
            source_path=yaml_source,
            subcategory=None,
            category="pipelines",
        )

        with patch(
            "autorag_research.plugin_registry._discover_plugin_configs_uncached",
            side_effect=_mock_uncached_for_category("pipelines", [info]),
        ):
            results = sync_plugin_configs(config_dir)

        assert results[0].copied
        dest = config_dir / "pipelines" / "flat_config.yaml"
        assert dest.exists()

    def test_no_plugins_returns_empty(self, tmp_path):
        """Returns empty list when no plugins are installed."""
        config_dir = tmp_path / "configs"

        with patch("autorag_research.plugin_registry._discover_plugin_configs_uncached", return_value=[]):
            results = sync_plugin_configs(config_dir)

        assert results == []
