"""Tests for ConfigPathManager singleton."""

from pathlib import Path

import pytest

from autorag_research.cli.config_path import ConfigPathManager


class TestConfigPathManager:
    """Tests for the ConfigPathManager singleton."""

    def setup_method(self):
        """Reset singleton before each test."""
        ConfigPathManager.reset()

    def teardown_method(self):
        """Reset singleton after each test."""
        ConfigPathManager.reset()

    def test_initialize_with_default(self):
        """Test initialization with default path."""
        ConfigPathManager.initialize()
        assert ConfigPathManager.get_config_dir() == (Path.cwd() / "configs").resolve()

    def test_initialize_with_custom_path(self, tmp_path):
        """Test initialization with custom path."""
        custom_path = tmp_path / "custom_configs"
        ConfigPathManager.initialize(custom_path)
        assert ConfigPathManager.get_config_dir() == custom_path.resolve()

    def test_initialize_with_string_path(self, tmp_path):
        """Test initialization with string path."""
        custom_path = str(tmp_path / "string_configs")
        ConfigPathManager.initialize(custom_path)
        assert ConfigPathManager.get_config_dir() == Path(custom_path).resolve()

    def test_double_initialize_returns_existing(self):
        """Test that double initialization returns existing instance."""
        first = ConfigPathManager.initialize()
        second = ConfigPathManager.initialize()  # Should not raise, returns existing
        assert first is second

    def test_get_before_initialize_raises_error(self):
        """Test that get before initialize raises RuntimeError."""
        with pytest.raises(RuntimeError, match="not initialized"):
            ConfigPathManager.get_config_dir()

    def test_get_instance_before_initialize_raises_error(self):
        """Test that get_instance before initialize raises RuntimeError."""
        with pytest.raises(RuntimeError, match="not initialized"):
            ConfigPathManager.get_instance()

    def test_is_initialized_false_initially(self):
        """Test is_initialized returns False before initialization."""
        assert not ConfigPathManager.is_initialized()

    def test_is_initialized_true_after_init(self):
        """Test is_initialized returns True after initialization."""
        ConfigPathManager.initialize()
        assert ConfigPathManager.is_initialized()

    def test_reset(self):
        """Test reset method clears the singleton."""
        ConfigPathManager.initialize()
        assert ConfigPathManager.is_initialized()
        ConfigPathManager.reset()
        assert not ConfigPathManager.is_initialized()

    def test_path_is_resolved(self, tmp_path):
        """Test that the path is always resolved to absolute."""
        # Use a relative-looking path
        ConfigPathManager.initialize(tmp_path / "relative" / ".." / "actual")
        result = ConfigPathManager.get_config_dir()
        assert result.is_absolute()
        assert ".." not in str(result)
