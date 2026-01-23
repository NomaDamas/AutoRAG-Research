"""Tests for CLI utility functions."""

from pathlib import Path

from omegaconf import DictConfig

from autorag_research.cli.utils import (
    APP_NAME,
    get_config_dir,
    get_db_url,
    get_user_data_dir,
)


class TestUtilityFunctions:
    """Tests for CLI utility functions."""

    def test_get_user_data_dir(self):
        """Test that user data directory path is returned."""
        data_dir = get_user_data_dir()
        assert isinstance(data_dir, Path)
        # Should contain the app name somewhere in the path
        assert APP_NAME in str(data_dir) or "autorag" in str(data_dir).lower()

    def test_get_config_dir(self):
        """Test that config directory is relative to cwd."""
        config_dir = get_config_dir()
        assert isinstance(config_dir, Path)
        assert config_dir.name == "configs"

    def test_get_db_url(self):
        """Test database URL generation."""
        cfg = DictConfig({
            "db": {
                "host": "localhost",
                "port": 5432,
                "user": "testuser",
                "password": "testpass",
                "database": "testdb",
            }
        })
        url = get_db_url(cfg)
        assert "postgresql+psycopg://" in url
        assert "testuser" in url
        assert "testpass" in url
        assert "localhost" in url
        assert "5432" in url
        assert "testdb" in url
