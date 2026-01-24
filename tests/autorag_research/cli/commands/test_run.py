"""Tests for autorag_research.cli.commands.run module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf
from typer.testing import CliRunner

import autorag_research.cli as cli
from autorag_research.cli.app import app
from autorag_research.cli.commands.run import (
    build_executor_config,
    create_session_factory,
    print_results,
)


@pytest.fixture
def cli_runner() -> CliRunner:
    """Return a Typer CliRunner for testing commands."""
    return CliRunner()


@pytest.fixture
def mock_cfg() -> MagicMock:
    """Create mock DictConfig for testing."""
    return OmegaConf.create({
        "db": {
            "host": "localhost",
            "port": 5432,
            "user": "testuser",
            "password": "testpass",
            "database": "testdb",
        },
        "db_name": "test_schema",
        "pipelines": [
            {"_target_": "some.module.Pipeline1", "name": "pipeline1"},
            {"_target_": "some.module.Pipeline2", "name": "pipeline2"},
        ],
        "metrics": [{"_target_": "some.module.Metric1", "name": "metric1"}],
        "max_retries": 3,
        "eval_batch_size": 32,
    })


@pytest.fixture
def mock_result() -> MagicMock:
    """Create mock executor result for testing."""
    result = MagicMock()
    result.pipeline_results = [
        MagicMock(pipeline_name="pipeline1", success=True, error=None),
        MagicMock(pipeline_name="pipeline2", success=True, error=None),
    ]
    result.metric_results = [
        MagicMock(metric_name="metric1", score=0.85),
    ]
    return result


class TestRunCommand:
    """Tests for the run CLI command."""

    def test_run_help_shows_options(self, cli_runner: CliRunner) -> None:
        """'run --help' shows available options."""
        result = cli_runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0
        assert "--db-name" in result.stdout
        assert "--config-name" in result.stdout
        assert "--max-retries" in result.stdout

    @patch("autorag_research.cli.commands.run._run_experiment")
    @patch("autorag_research.cli.commands.run.compose")
    @patch("autorag_research.cli.commands.run.initialize_config_dir")
    @patch("autorag_research.cli.commands.run.GlobalHydra")
    def test_run_with_config_path(
        self,
        mock_global_hydra: MagicMock,
        mock_init_config: MagicMock,
        mock_compose: MagicMock,
        mock_run_experiment: MagicMock,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """run command uses cli.CONFIG_PATH."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        monkeypatch.setattr(cli, "CONFIG_PATH", config_dir)

        mock_compose.return_value = OmegaConf.create({"db_name": "test"})
        mock_init_config.return_value.__enter__ = MagicMock()
        mock_init_config.return_value.__exit__ = MagicMock()

        cli_runner.invoke(app, ["run", "--db-name", "test_db"])

        mock_run_experiment.assert_called_once()

    def test_run_without_config_path_exits(self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        """run command exits if CONFIG_PATH is None."""
        monkeypatch.setattr(cli, "CONFIG_PATH", None)

        result = cli_runner.invoke(app, ["run", "--db-name", "test"])

        assert result.exit_code == 1


class TestBuildExecutorConfig:
    """Tests for build_executor_config function."""

    @patch("autorag_research.cli.commands.run.instantiate")
    def test_instantiates_pipelines(self, mock_instantiate: MagicMock, mock_cfg: MagicMock) -> None:
        """Instantiates each pipeline config."""
        mock_instantiate.return_value = MagicMock()

        build_executor_config(mock_cfg)

        # Should be called for each pipeline and metric
        assert mock_instantiate.call_count >= 2  # At least 2 pipelines

    @patch("autorag_research.cli.commands.run.instantiate")
    def test_instantiates_metrics(self, mock_instantiate: MagicMock, mock_cfg: MagicMock) -> None:
        """Instantiates each metric config."""
        mock_instantiate.return_value = MagicMock()

        build_executor_config(mock_cfg)

        # Should be called for metrics too
        assert mock_instantiate.call_count >= 3  # 2 pipelines + 1 metric

    @patch("autorag_research.cli.commands.run.instantiate")
    def test_returns_executor_config(self, mock_instantiate: MagicMock, mock_cfg: MagicMock) -> None:
        """Returns an ExecutorConfig object."""
        from autorag_research.config import ExecutorConfig

        mock_pipeline = MagicMock()
        mock_metric = MagicMock()
        mock_instantiate.side_effect = [mock_pipeline, mock_pipeline, mock_metric]

        result = build_executor_config(mock_cfg)

        assert isinstance(result, ExecutorConfig)

    @patch("autorag_research.cli.commands.run.instantiate")
    def test_uses_config_max_retries(self, mock_instantiate: MagicMock, mock_cfg: MagicMock) -> None:
        """Uses max_retries from config."""
        mock_instantiate.return_value = MagicMock()

        result = build_executor_config(mock_cfg)

        assert result.max_retries == 3

    @patch("autorag_research.cli.commands.run.instantiate")
    def test_uses_config_eval_batch_size(self, mock_instantiate: MagicMock, mock_cfg: MagicMock) -> None:
        """Uses eval_batch_size from config."""
        mock_instantiate.return_value = MagicMock()

        result = build_executor_config(mock_cfg)

        assert result.eval_batch_size == 32


class TestCreateSessionFactory:
    """Tests for create_session_factory function."""

    @patch("sqlalchemy.orm.sessionmaker")
    @patch("sqlalchemy.create_engine")
    def test_creates_engine_with_correct_url(
        self, mock_create_engine: MagicMock, mock_sessionmaker: MagicMock, mock_cfg: MagicMock
    ) -> None:
        """Creates engine with correct connection URL."""
        create_session_factory(mock_cfg)

        # Check that create_engine was called with correct URL format
        call_args = mock_create_engine.call_args
        url = call_args[0][0]
        assert "postgresql+psycopg" in url
        assert "testuser" in url
        assert "localhost" in url
        assert "5432" in str(url)
        assert "testdb" in url

    @patch("sqlalchemy.orm.sessionmaker")
    @patch("sqlalchemy.create_engine")
    def test_returns_sessionmaker(
        self, mock_create_engine: MagicMock, mock_sessionmaker: MagicMock, mock_cfg: MagicMock
    ) -> None:
        """Returns sessionmaker bound to engine."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        create_session_factory(mock_cfg)

        mock_sessionmaker.assert_called_once_with(bind=mock_engine)

    @patch("sqlalchemy.orm.sessionmaker")
    @patch("sqlalchemy.create_engine")
    def test_handles_env_var_password(
        self, mock_create_engine: MagicMock, mock_sessionmaker: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Handles ${env:...} password pattern by falling back to PGPASSWORD."""
        monkeypatch.setenv("PGPASSWORD", "env_password")

        cfg = OmegaConf.create({
            "db": {
                "host": "localhost",
                "port": 5432,
                "user": "testuser",
                "password": "${oc.env:PGPASSWORD}",
                "database": "testdb",
            }
        })

        create_session_factory(cfg)

        # Password should be resolved from env
        call_args = mock_create_engine.call_args
        url = call_args[0][0]
        assert "env_password" in url


class TestPrintResults:
    """Tests for print_results function."""

    def test_prints_successful_pipelines(self, mock_result: MagicMock, capsys: pytest.CaptureFixture) -> None:
        """Prints success indicator for successful pipelines."""
        print_results(mock_result)

        captured = capsys.readouterr()
        assert "pipeline1" in captured.out
        assert "pipeline2" in captured.out

    def test_prints_failed_pipelines(self, capsys: pytest.CaptureFixture) -> None:
        """Prints error for failed pipelines."""
        result = MagicMock()
        result.pipeline_results = [
            MagicMock(pipeline_name="failed_pipeline", success=False, error="Connection timeout"),
        ]
        result.metric_results = []

        print_results(result)

        captured = capsys.readouterr()
        assert "failed_pipeline" in captured.out

    def test_prints_metric_scores(self, mock_result: MagicMock, capsys: pytest.CaptureFixture) -> None:
        """Prints metric names and scores."""
        print_results(mock_result)

        captured = capsys.readouterr()
        assert "metric1" in captured.out
        # Score should be present (0.85 or formatted version)
        assert "0.85" in captured.out or "85" in captured.out

    def test_prints_summary(self, mock_result: MagicMock, capsys: pytest.CaptureFixture) -> None:
        """Prints summary with completion count."""
        print_results(mock_result)

        captured = capsys.readouterr()
        # Should show some form of "2/2" or "2 of 2" completion
        assert "2" in captured.out
