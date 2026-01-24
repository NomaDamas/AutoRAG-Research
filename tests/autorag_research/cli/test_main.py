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
        # Typer uses rich formatting with "Commands" header
        assert "Commands" in result.stdout

    def test_unknown_command_shows_usage(self):
        """Test that unknown command shows error message."""
        result = subprocess.run(
            [sys.executable, "-m", "autorag_research.cli.main", "unknown_command"],
            capture_output=True,
            text=True,
        )
        # Typer returns exit code 2 for unknown commands
        assert result.returncode == 2
        assert "No such command" in result.stderr

    def test_no_args_shows_usage(self):
        """Test that no arguments shows usage."""
        result = subprocess.run(
            [sys.executable, "-m", "autorag_research.cli.main"],
            capture_output=True,
            text=True,
        )
        # Typer returns exit code 2 when no_args_is_help=True
        assert result.returncode == 2
        assert "AutoRAG-Research CLI" in result.stdout


class TestListCommand:
    """Tests for the list command."""

    def test_list_datasets(self):
        """Test listing available datasets."""
        result = subprocess.run(
            [sys.executable, "-m", "autorag_research.cli.main", "list", "datasets"],
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
            [sys.executable, "-m", "autorag_research.cli.main", "list", "pipelines"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Available Pipelines:" in result.stdout
        assert "bm25" in result.stdout

    def test_list_metrics(self):
        """Test listing available metrics."""
        result = subprocess.run(
            [sys.executable, "-m", "autorag_research.cli.main", "list", "metrics"],
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

        # Check that db.yaml was created
        assert (configs_dir / "db.yaml").exists()

        # Check subdirectories (datasets handled via CLI, not YAML)
        assert (configs_dir / "pipelines").exists()
        assert (configs_dir / "metrics").exists()

    def test_init_config_skips_existing_files(self, tmp_path):
        """Test that init-config skips existing files."""
        # Create configs directory and a file
        db_yaml = tmp_path / "configs" / "db.yaml"
        db_yaml.write_text("existing: content")

        result = subprocess.run(
            [sys.executable, "-m", "autorag_research.cli.main", "init-config"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0
        assert "[skip] db.yaml" in result.stdout

        # Verify file wasn't overwritten
        assert db_yaml.read_text() == "existing: content"


class TestRunCommand:
    """Tests for the run command."""

    def test_run_help(self):
        """Test that run --help shows options with kebab-case."""
        result = subprocess.run(
            [sys.executable, "-m", "autorag_research.cli.main", "run", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--db-name" in result.stdout
        assert "--config-name" in result.stdout
        assert "--max-retries" in result.stdout


class TestIngestCommand:
    """Tests for the ingest command."""

    def test_ingest_help(self):
        """Test that ingest --help shows available ingestors."""
        result = subprocess.run(
            [sys.executable, "-m", "autorag_research.cli.main", "ingest", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "beir" in result.stdout
        assert "mrtydi" in result.stdout

    def test_ingest_beir_help(self):
        """Test that ingest beir --help shows options."""
        result = subprocess.run(
            [sys.executable, "-m", "autorag_research.cli.main", "ingest", "beir", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--dataset" in result.stdout
        assert "--db-name" in result.stdout
        assert "--db-host" in result.stdout
