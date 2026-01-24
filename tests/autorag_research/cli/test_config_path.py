"""Tests for CONFIG_PATH in cli module."""

from pathlib import Path

import autorag_research.cli as cli


class TestConfigPath:
    """Tests for the CONFIG_PATH global variable."""

    def setup_method(self):
        """Reset config path before each test."""
        cli.CONFIG_PATH = None

    def teardown_method(self):
        """Reset config path after each test."""
        cli.CONFIG_PATH = None

    def test_default_is_none(self):
        """Test that CONFIG_PATH is None by default."""
        assert cli.CONFIG_PATH is None

    def test_set_config_path(self, tmp_path):
        """Test setting CONFIG_PATH."""
        cli.CONFIG_PATH = tmp_path / "configs"
        assert tmp_path / "configs" == cli.CONFIG_PATH

    def test_or_fallback_when_none(self):
        """Test fallback behavior with or operator."""
        fallback = Path.cwd() / "configs"
        result = cli.CONFIG_PATH or fallback
        assert result == fallback

    def test_or_fallback_when_set(self, tmp_path):
        """Test that set value takes precedence over fallback."""
        cli.CONFIG_PATH = tmp_path / "custom"
        fallback = Path.cwd() / "configs"
        result = cli.CONFIG_PATH or fallback
        assert result == tmp_path / "custom"
