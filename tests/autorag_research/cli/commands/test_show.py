"""Tests for autorag_research.cli.commands.show module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

import autorag_research.cli as cli
from autorag_research.cli.app import app
from autorag_research.cli.commands.show import (
    print_databases,
    print_ingestors,
    print_metrics,
    print_pipelines,
)


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
    """Tests for print_metrics function."""

    def test_displays_metrics(self, capsys: pytest.CaptureFixture) -> None:
        """Displays available metrics from temp config."""
        cli.CONFIG_PATH = Path(__file__).parent.parent.parent.parent.parent / "configs"

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


class TestPrintDatasets:
    """Tests for print_datasets function."""

    def test_shows_ingestors_with_hf_repos(self, capsys: pytest.CaptureFixture) -> None:
        """Shows ingestors that have HuggingFace repos when no ingestor specified."""
        from autorag_research.cli.commands.show import print_datasets

        print_datasets()

        captured = capsys.readouterr()
        assert "beir" in captured.out
        assert "HuggingFace" in captured.out or "datasets" in captured.out.lower()


class TestShowDatasetsCommand:
    """Tests for 'show datasets' command."""

    def test_show_datasets_help_shows_options(self, cli_runner: CliRunner) -> None:
        """'show datasets --help' shows available options."""
        result = cli_runner.invoke(app, ["show", "datasets", "--help"])

        assert result.exit_code == 0
        assert "datasets" in result.stdout.lower() or "NAME" in result.stdout

    @patch("autorag_research.data.hf_storage.list_available_dumps")
    def test_show_datasets_with_ingestor(self, mock_list: MagicMock, cli_runner: CliRunner) -> None:
        """'show datasets <ingestor>' shows available dumps."""
        mock_list.return_value = ["scifact_openai-small", "nfcorpus_bge-small"]

        result = cli_runner.invoke(app, ["show", "datasets", "beir"])

        assert result.exit_code == 0
        assert "scifact_openai-small" in result.stdout
        assert "nfcorpus_bge-small" in result.stdout
        mock_list.assert_called_once_with("beir")

    @patch("autorag_research.data.hf_storage.list_available_dumps")
    def test_show_datasets_empty_repo(self, mock_list: MagicMock, cli_runner: CliRunner) -> None:
        """'show datasets' with empty repo shows message."""
        mock_list.return_value = []

        result = cli_runner.invoke(app, ["show", "datasets", "beir"])

        assert result.exit_code == 0
        assert "No dump files found" in result.stdout

    @patch("autorag_research.data.hf_storage.list_available_dumps")
    def test_show_datasets_unknown_ingestor(self, mock_list: MagicMock, cli_runner: CliRunner) -> None:
        """'show datasets' with unknown ingestor shows error."""
        mock_list.side_effect = KeyError("Unknown ingestor or no HF repo configured: 'unknown'")

        result = cli_runner.invoke(app, ["show", "datasets", "unknown"])

        assert result.exit_code == 1
        assert "Unknown ingestor" in result.output or "unknown" in result.output.lower()
