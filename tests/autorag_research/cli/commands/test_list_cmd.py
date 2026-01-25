"""Tests for autorag_research.cli.commands.list_cmd module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

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

    def test_list_datasets_calls_print_ingestors(self, cli_runner: CliRunner) -> None:
        """'list datasets' routes to print_ingestors."""
        with patch("autorag_research.cli.commands.list_cmd.print_ingestors") as mock:
            cli_runner.invoke(app, ["list", "datasets"])
            mock.assert_called_once()

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

    def test_list_databases_calls_handler(self, cli_runner: CliRunner) -> None:
        """'list databases' routes to database handler."""
        with patch("autorag_research.cli.commands.list_cmd._print_databases_with_config") as mock:
            cli_runner.invoke(app, ["list", "databases"])
            mock.assert_called_once()

    def test_list_help_shows_options(self, cli_runner: CliRunner) -> None:
        """'list --help' shows available resource types."""
        result = cli_runner.invoke(app, ["list", "--help"])

        assert result.exit_code == 0
        assert "datasets" in result.stdout
        assert "pipelines" in result.stdout
        assert "metrics" in result.stdout


class TestPrintIngestors:
    """Tests for print_ingestors function using real ingestor registry."""

    def test_displays_real_ingestor_names(self, capsys: pytest.CaptureFixture) -> None:
        """Displays real ingestor names from registry."""
        print_ingestors()

        captured = capsys.readouterr()
        assert "beir" in captured.out
        assert "ragbench" in captured.out

    def test_displays_descriptions(self, capsys: pytest.CaptureFixture) -> None:
        """Displays ingestor descriptions."""
        print_ingestors()

        captured = capsys.readouterr()
        # Real beir ingestor has description
        assert "BEIR" in captured.out or "benchmark" in captured.out.lower()

    def test_displays_parameters(self, capsys: pytest.CaptureFixture) -> None:
        """Displays parameter options from real ingestors."""
        print_ingestors()

        captured = capsys.readouterr()
        # beir has dataset-name parameter
        assert "dataset-name" in captured.out or "--dataset-name" in captured.out

    def test_displays_required_indicator(self, capsys: pytest.CaptureFixture) -> None:
        """Shows required indicator for required parameters."""
        print_ingestors()

        captured = capsys.readouterr()
        assert "required" in captured.out.lower()


class TestPrintPipelines:
    """Tests for print_pipelines function using real configs."""

    def test_displays_real_pipelines(self, real_config_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Displays pipelines from real configs/pipelines/."""
        print_pipelines()

        captured = capsys.readouterr()
        assert "bm25" in captured.out
        assert "basic_rag" in captured.out

    def test_empty_pipelines_shows_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Shows message when no pipelines found."""
        # Create empty pipelines directory
        pipelines_dir = tmp_path / "pipelines"
        pipelines_dir.mkdir()
        monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path)

        print_pipelines()

        captured = capsys.readouterr()
        # Should show some indication that no pipelines were found
        assert "No" in captured.out or "not found" in captured.out.lower() or captured.out.strip() == ""


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

    def test_empty_metrics_shows_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Shows message when no metrics found."""
        # Create empty metrics directory
        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()
        monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path)

        print_metrics()

        captured = capsys.readouterr()
        assert "No" in captured.out or "not found" in captured.out.lower() or captured.out.strip() == ""


class TestPrintDatabases:
    """Tests for print_databases function using real database.

    Note: print_databases excludes system schemas including 'public',
    so a fresh test database shows 'No user schemas found'.
    """

    def test_displays_output_from_real_db(
        self, test_db_params: dict[str, str | int], capsys: pytest.CaptureFixture
    ) -> None:
        """Displays output from real test database connection."""
        print_databases(
            host=test_db_params["host"],
            port=test_db_params["port"],
            user=test_db_params["user"],
            password=test_db_params["password"],
            database=test_db_params["database"],
        )

        captured = capsys.readouterr()
        # Should show database info (connection worked)
        assert "Database" in captured.out
        assert test_db_params["database"] in captured.out

    @patch("autorag_research.cli.utils.list_schemas_with_connection")
    def test_empty_schemas_shows_message(self, mock_list_schemas: MagicMock, capsys: pytest.CaptureFixture) -> None:
        """Shows message when no schemas found (excluding system schemas)."""
        mock_list_schemas.return_value = []

        print_databases("localhost", 5432, "postgres", "pass", "testdb")

        captured = capsys.readouterr()
        assert "No" in captured.out or "schemas" in captured.out.lower()

    @patch("autorag_research.cli.utils.list_schemas_with_connection")
    def test_connection_error_exits(self, mock_list_schemas: MagicMock) -> None:
        """Exits with code 1 on database connection error."""
        mock_list_schemas.side_effect = Exception("Connection refused")

        with pytest.raises(SystemExit) as exc_info:
            print_databases("localhost", 5432, "postgres", "pass", "testdb")

        assert exc_info.value.code == 1
