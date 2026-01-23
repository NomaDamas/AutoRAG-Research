"""Tests for CLI main module."""

import subprocess
import sys

import pytest


class TestCLIMain:
    """Tests for the main CLI entry point."""

    def test_help_command(self):
        """Test that --help shows usage information."""
        result = subprocess.run(
            [sys.executable, "-m", "autorag_research.cli.main", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "AutoRAG-Research CLI" in result.stdout
        assert "Commands:" in result.stdout

    def test_unknown_command_shows_usage(self):
        """Test that unknown command shows usage."""
        result = subprocess.run(
            [sys.executable, "-m", "autorag_research.cli.main", "unknown_command"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Unknown command" in result.stdout

    def test_no_args_shows_usage(self):
        """Test that no arguments shows usage."""
        result = subprocess.run(
            [sys.executable, "-m", "autorag_research.cli.main"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "AutoRAG-Research CLI" in result.stdout


class TestListCommand:
    """Tests for the list command."""

    def test_list_datasets(self):
        """Test listing available datasets."""
        result = subprocess.run(
            [sys.executable, "-m", "autorag_research.cli.main", "list", "resource=datasets"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Available Datasets:" in result.stdout
        # Now lists ingestor names instead of individual datasets
        assert "beir" in result.stdout
        assert "mrtydi" in result.stdout

    def test_list_pipelines(self):
        """Test listing available pipelines."""
        result = subprocess.run(
            [sys.executable, "-m", "autorag_research.cli.main", "list", "resource=pipelines"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Available Pipelines:" in result.stdout
        assert "bm25_baseline" in result.stdout

    def test_list_metrics(self):
        """Test listing available metrics."""
        result = subprocess.run(
            [sys.executable, "-m", "autorag_research.cli.main", "list", "resource=metrics"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Available Metrics:" in result.stdout
        assert "recall" in result.stdout
        assert "ndcg" in result.stdout


@pytest.mark.skip("This test requires pulling files from main branch; which means it needs to be merged first.")
class TestInitConfigCommand:
    """Tests for the init-config command."""

    def test_init_config_creates_files(self, tmp_path):
        """Test that init-config creates configuration files."""
        result = subprocess.run(
            [sys.executable, "-m", "autorag_research.cli.main", "init-config"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0
        assert "Initializing configuration files" in result.stdout

        # Check that configs directory was created
        configs_dir = tmp_path / "configs"
        assert configs_dir.exists()

        # Check that db/default.yaml was created
        assert (configs_dir / "db" / "default.yaml").exists()

        # Check subdirectories (datasets handled via CLI, not YAML)
        assert (configs_dir / "pipelines").exists()
        assert (configs_dir / "metrics").exists()

    def test_init_config_skips_existing_files(self, tmp_path):
        """Test that init-config skips existing files."""
        # Create configs directory and a file
        configs_dir = tmp_path / "configs" / "db"
        configs_dir.mkdir(parents=True)
        db_yaml = configs_dir / "default.yaml"
        db_yaml.write_text("existing: content")

        result = subprocess.run(
            [sys.executable, "-m", "autorag_research.cli.main", "init-config"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0
        assert "[skip] db/default.yaml" in result.stdout

        # Verify file wasn't overwritten
        assert db_yaml.read_text() == "existing: content"
