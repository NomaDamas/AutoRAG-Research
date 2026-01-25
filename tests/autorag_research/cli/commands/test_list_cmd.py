"""Tests for autorag_research.cli.commands.list_cmd module."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import autorag_research.cli as cli
from autorag_research.cli.app import app
from autorag_research.cli.commands.list_cmd import (
    print_databases,
    print_ingestors,
    print_metrics,
    print_pipelines,
)


class TestListResourcesCommand:
    """Tests for the list_resources CLI command."""

    def test_list_ingestors_calls_print_ingestors(self, cli_runner: CliRunner) -> None:
        """'list ingestors' routes to print_ingestors."""
        with patch("autorag_research.cli.commands.list_cmd.print_ingestors") as mock:
            cli_runner.invoke(app, ["list", "ingestors"])
            mock.assert_called_once()

    def test_list_pipelines_calls_print_pipelines(self, cli_runner: CliRunner) -> None:
        """'list pipelines' routes to print_pipelines."""
        with patch("autorag_research.cli.commands.list_cmd.print_pipelines") as mock:
            cli_runner.invoke(app, ["list", "pipelines"])
            mock.assert_called_once()

    def test_list_metrics_calls_print_metrics(self, cli_runner: CliRunner) -> None:
        """'list metrics' routes to print_metrics."""
        with patch("autorag_research.cli.commands.list_cmd.print_metrics") as mock:
            cli_runner.invoke(app, ["list", "metrics"])
            mock.assert_called_once()

    def test_list_help_shows_options(self, cli_runner: CliRunner) -> None:
        """'list --help' shows available resource types."""
        result = cli_runner.invoke(app, ["list", "--help"])

        assert result.exit_code == 0
        assert "pipelines" in result.stdout
        assert "metrics" in result.stdout
        assert "ingestors" in result.stdout
        assert "databases" in result.stdout


def test_print_ingestors(capsys: pytest.CaptureFixture) -> None:
    """Displays real ingestor names from registry."""
    print_ingestors()

    captured = capsys.readouterr()
    assert "beir" in captured.out
    assert "ragbench" in captured.out
    assert "BEIR" in captured.out and "benchmark" in captured.out
    assert "--dataset-name" in captured.out
    assert "required" in captured.out.lower()


class TestPrintPipelines:
    """Tests for print_pipelines function using real configs."""

    def test_displays_real_pipelines(self, real_config_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Displays pipelines from real configs/pipelines/."""
        print_pipelines()

        captured = capsys.readouterr()
        assert "bm25" in captured.out
        assert "basic_rag" in captured.out


class TestPrintMetrics:
    """Tests for print_metrics function.

    Note: Real configs have metrics in subdirectories, so discover_metrics returns empty.
    We use tmp_path to test with valid metric configs.
    """

    def test_displays_metrics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Displays available metrics from temp config."""
        # Create metrics directory with top-level YAML files
        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()
        (metrics_dir / "ndcg.yaml").write_text("description: NDCG@k metric")
        (metrics_dir / "recall.yaml").write_text("description: Recall@k metric")
        monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path)

        print_metrics()

        captured = capsys.readouterr()
        assert "ndcg" in captured.out
        assert "recall" in captured.out


class TestPrintDatabases:
    """Tests for print_databases function using real database.

    Note: print_databases excludes system schemas including 'public',
    so a fresh test database shows 'No user schemas found'.
    """

    def test_displays_output_from_real_db(self, capsys: pytest.CaptureFixture) -> None:
        """Displays output from real test database connection."""
        cli.CONFIG_PATH = Path(__file__).parent.parent.parent.parent.parent / "configs"

        print_databases()

        captured = capsys.readouterr()
        # Should show server info (connection worked)
        assert "Server" in captured.out
        assert "testdb" in captured.out
