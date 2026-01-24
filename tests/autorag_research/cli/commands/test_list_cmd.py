"""Tests for autorag_research.cli.commands.list_cmd module."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from autorag_research.cli.app import app
from autorag_research.cli.commands.list_cmd import (
    print_databases,
    print_ingestors,
    print_metrics,
    print_pipelines,
)
from autorag_research.data.registry import IngestorMeta, ParamMeta


@pytest.fixture
def cli_runner() -> CliRunner:
    """Return a Typer CliRunner for testing commands."""
    return CliRunner()


@pytest.fixture
def mock_ingestor_registry() -> dict[str, IngestorMeta]:
    """Create mock ingestor registry for testing."""
    return {
        "beir": IngestorMeta(
            name="beir",
            ingestor_class=MagicMock,
            description="BEIR benchmark datasets",
            params=[
                ParamMeta(
                    name="dataset_name",
                    cli_option="dataset-name",
                    param_type=str,
                    choices=["scifact", "nfcorpus", "fiqa", "hotpotqa"],
                    required=True,
                    default=None,
                    help="Dataset to ingest",
                    is_list=False,
                ),
            ],
        ),
        "mteb": IngestorMeta(
            name="mteb",
            ingestor_class=MagicMock,
            description="MTEB retrieval tasks",
            params=[
                ParamMeta(
                    name="task_name",
                    cli_option="task-name",
                    param_type=str,
                    choices=None,
                    required=True,
                    default=None,
                    help="Task name",
                    is_list=False,
                ),
                ParamMeta(
                    name="batch_size",
                    cli_option="batch-size",
                    param_type=int,
                    choices=None,
                    required=False,
                    default=100,
                    help="Batch size",
                    is_list=False,
                ),
            ],
        ),
    }


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
    """Tests for print_ingestors function."""

    @patch("autorag_research.cli.commands.list_cmd.discover_ingestors")
    def test_displays_ingestor_names(
        self, mock_discover: MagicMock, mock_ingestor_registry: dict[str, IngestorMeta], capsys: pytest.CaptureFixture
    ) -> None:
        """Displays ingestor names."""
        mock_discover.return_value = mock_ingestor_registry

        print_ingestors()

        captured = capsys.readouterr()
        assert "beir" in captured.out
        assert "mteb" in captured.out

    @patch("autorag_research.cli.commands.list_cmd.discover_ingestors")
    def test_displays_descriptions(
        self, mock_discover: MagicMock, mock_ingestor_registry: dict[str, IngestorMeta], capsys: pytest.CaptureFixture
    ) -> None:
        """Displays ingestor descriptions."""
        mock_discover.return_value = mock_ingestor_registry

        print_ingestors()

        captured = capsys.readouterr()
        assert "BEIR benchmark datasets" in captured.out
        assert "MTEB retrieval tasks" in captured.out

    @patch("autorag_research.cli.commands.list_cmd.discover_ingestors")
    def test_displays_parameters(
        self, mock_discover: MagicMock, mock_ingestor_registry: dict[str, IngestorMeta], capsys: pytest.CaptureFixture
    ) -> None:
        """Displays parameter options."""
        mock_discover.return_value = mock_ingestor_registry

        print_ingestors()

        captured = capsys.readouterr()
        assert "--dataset-name" in captured.out or "dataset-name" in captured.out
        assert "--task-name" in captured.out or "task-name" in captured.out

    @patch("autorag_research.cli.commands.list_cmd.discover_ingestors")
    def test_displays_required_indicator(
        self, mock_discover: MagicMock, mock_ingestor_registry: dict[str, IngestorMeta], capsys: pytest.CaptureFixture
    ) -> None:
        """Shows required indicator for required parameters."""
        mock_discover.return_value = mock_ingestor_registry

        print_ingestors()

        captured = capsys.readouterr()
        assert "required" in captured.out.lower()

    @patch("autorag_research.cli.commands.list_cmd.discover_ingestors")
    def test_displays_default_values(
        self, mock_discover: MagicMock, mock_ingestor_registry: dict[str, IngestorMeta], capsys: pytest.CaptureFixture
    ) -> None:
        """Shows default values for optional parameters."""
        mock_discover.return_value = mock_ingestor_registry

        print_ingestors()

        captured = capsys.readouterr()
        assert "100" in captured.out or "default" in captured.out.lower()

    @patch("autorag_research.cli.commands.list_cmd.discover_ingestors")
    def test_displays_choices_preview(
        self, mock_discover: MagicMock, mock_ingestor_registry: dict[str, IngestorMeta], capsys: pytest.CaptureFixture
    ) -> None:
        """Shows choices preview for parameters with choices."""
        mock_discover.return_value = mock_ingestor_registry

        print_ingestors()

        captured = capsys.readouterr()
        # Should show some choices from beir dataset
        assert "scifact" in captured.out or "nfcorpus" in captured.out


class TestPrintPipelines:
    """Tests for print_pipelines function."""

    @patch("autorag_research.cli.commands.list_cmd.discover_pipelines")
    def test_displays_pipelines(self, mock_discover: MagicMock, capsys: pytest.CaptureFixture) -> None:
        """Displays available pipelines."""
        mock_discover.return_value = {"bm25": "BM25 retrieval", "dense": "Dense vector retrieval"}

        print_pipelines()

        captured = capsys.readouterr()
        assert "bm25" in captured.out
        assert "dense" in captured.out

    @patch("autorag_research.cli.commands.list_cmd.discover_pipelines")
    def test_displays_descriptions(self, mock_discover: MagicMock, capsys: pytest.CaptureFixture) -> None:
        """Displays pipeline descriptions."""
        mock_discover.return_value = {"bm25": "BM25 retrieval pipeline"}

        print_pipelines()

        captured = capsys.readouterr()
        assert "BM25 retrieval" in captured.out

    @patch("autorag_research.cli.commands.list_cmd.discover_pipelines")
    def test_empty_pipelines_shows_message(self, mock_discover: MagicMock, capsys: pytest.CaptureFixture) -> None:
        """Shows message when no pipelines found."""
        mock_discover.return_value = {}

        print_pipelines()

        captured = capsys.readouterr()
        # Should show some indication that no pipelines were found
        assert "No" in captured.out or "not found" in captured.out.lower() or captured.out.strip() == ""


class TestPrintMetrics:
    """Tests for print_metrics function."""

    @patch("autorag_research.cli.commands.list_cmd.discover_metrics")
    def test_displays_metrics(self, mock_discover: MagicMock, capsys: pytest.CaptureFixture) -> None:
        """Displays available metrics."""
        mock_discover.return_value = {"ndcg": "NDCG@k metric", "recall": "Recall@k metric"}

        print_metrics()

        captured = capsys.readouterr()
        assert "ndcg" in captured.out
        assert "recall" in captured.out

    @patch("autorag_research.cli.commands.list_cmd.discover_metrics")
    def test_empty_metrics_shows_message(self, mock_discover: MagicMock, capsys: pytest.CaptureFixture) -> None:
        """Shows message when no metrics found."""
        mock_discover.return_value = {}

        print_metrics()

        captured = capsys.readouterr()
        assert "No" in captured.out or "not found" in captured.out.lower() or captured.out.strip() == ""


class TestPrintDatabases:
    """Tests for print_databases function."""

    @patch("autorag_research.cli.utils.list_schemas_with_connection")
    def test_displays_schemas(self, mock_list_schemas: MagicMock, capsys: pytest.CaptureFixture) -> None:
        """Displays database schemas."""
        mock_list_schemas.return_value = ["beir_scifact", "mteb_nfcorpus"]

        print_databases("localhost", 5432, "postgres", "pass", "testdb")

        captured = capsys.readouterr()
        assert "beir_scifact" in captured.out
        assert "mteb_nfcorpus" in captured.out

    @patch("autorag_research.cli.utils.list_schemas_with_connection")
    def test_empty_schemas_shows_message(self, mock_list_schemas: MagicMock, capsys: pytest.CaptureFixture) -> None:
        """Shows message when no schemas found."""
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
