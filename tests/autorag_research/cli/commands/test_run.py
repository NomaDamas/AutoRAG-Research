"""Tests for autorag_research.cli.commands.run module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig, OmegaConf
from sqlalchemy.orm import Session
from typer.testing import CliRunner

import autorag_research.cli as cli
from autorag_research.cli.app import app
from autorag_research.cli.commands.run import (
    _load_experiment_config,
    build_executor_config,
    create_session_factory,
    print_results,
)
from autorag_research.config import ExecutorConfig

# Register mock hydra resolver for tests (handles ${hydra:runtime.cwd} etc.)
if not OmegaConf.has_resolver("hydra"):
    OmegaConf.register_new_resolver("hydra", lambda key: "./test_indices" if key == "runtime.cwd" else "")


@pytest.fixture
def real_db_cfg(test_db_params: dict[str, str | int]) -> DictConfig:
    """Create OmegaConf config using real test database parameters."""
    return OmegaConf.create({
        "db": {
            "host": test_db_params["host"],
            "port": test_db_params["port"],
            "user": test_db_params["user"],
            "password": test_db_params["password"],
            "database": test_db_params["database"],
        },
        "db_name": "public",
    })


@pytest.fixture
def mock_result() -> MagicMock:
    """Create mock executor result for testing print_results."""
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
    def test_run_with_real_config(
        self,
        mock_run_experiment: MagicMock,
        cli_runner: CliRunner,
        real_config_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """run command loads real experiment.yaml and calls _run_experiment."""
        monkeypatch.setattr(cli, "CONFIG_PATH", real_config_path)

        cli_runner.invoke(app, ["run", "--db-name", "test_db"])

        # Should call _run_experiment (we mock it to avoid actual execution)
        mock_run_experiment.assert_called_once()
        # The config passed should have our db_name override
        call_args = mock_run_experiment.call_args[0][0]
        assert isinstance(call_args, DictConfig)

    def test_run_without_config_path_exits(self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        """run command exits if CONFIG_PATH is None."""
        monkeypatch.setattr(cli, "CONFIG_PATH", None)

        result = cli_runner.invoke(app, ["run", "--db-name", "test"])

        assert result.exit_code == 1

    def test_run_with_missing_config_file_exits(
        self, cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run command exits if config file doesn't exist."""
        monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path)

        result = cli_runner.invoke(app, ["run", "--config-name", "nonexistent"])

        assert result.exit_code == 1
        # Error message may be in stdout or output
        output = result.stdout + (result.output if hasattr(result, "output") else "")
        assert "not found" in output.lower()


class TestLoadExperimentConfig:
    """Tests for _load_experiment_config function using real config files."""

    def test_loads_real_experiment_config(self, real_config_path: Path) -> None:
        """Loads and resolves real experiment.yaml."""
        cfg = _load_experiment_config(
            config_path=real_config_path,
            config_name="experiment",
            db_name=None,
            max_retries=None,
            eval_batch_size=None,
        )

        assert isinstance(cfg, DictConfig)
        assert "pipelines" in cfg
        assert "metrics" in cfg
        assert "db" in cfg

    def test_db_name_override(self, real_config_path: Path) -> None:
        """CLI db_name overrides config value."""
        cfg = _load_experiment_config(
            config_path=real_config_path,
            config_name="experiment",
            db_name="override_db",
            max_retries=None,
            eval_batch_size=None,
        )

        assert cfg.db_name == "override_db"

    def test_max_retries_override(self, real_config_path: Path) -> None:
        """CLI max_retries overrides config value."""
        cfg = _load_experiment_config(
            config_path=real_config_path,
            config_name="experiment",
            db_name=None,
            max_retries=10,
            eval_batch_size=None,
        )

        assert cfg.max_retries == 10

    def test_eval_batch_size_override(self, real_config_path: Path) -> None:
        """CLI eval_batch_size overrides config value."""
        cfg = _load_experiment_config(
            config_path=real_config_path,
            config_name="experiment",
            db_name=None,
            max_retries=None,
            eval_batch_size=50,
        )

        assert cfg.eval_batch_size == 50

    def test_resolves_pipeline_configs(self, real_config_path: Path) -> None:
        """Resolves pipeline names to full configs with _target_."""
        cfg = _load_experiment_config(
            config_path=real_config_path,
            config_name="experiment",
            db_name=None,
            max_retries=None,
            eval_batch_size=None,
        )

        # Pipelines should be resolved to list-like structure with _target_
        assert len(cfg.pipelines) > 0
        # Each pipeline config should have _target_
        for pipeline_cfg in cfg.pipelines:
            assert "_target_" in pipeline_cfg

    def test_resolves_metric_configs(self, real_config_path: Path) -> None:
        """Resolves metric names to full configs with _target_."""
        cfg = _load_experiment_config(
            config_path=real_config_path,
            config_name="experiment",
            db_name=None,
            max_retries=None,
            eval_batch_size=None,
        )

        # Metrics should be resolved to list-like structure with _target_
        assert len(cfg.metrics) > 0
        # Each metric config should have _target_
        for metric_cfg in cfg.metrics:
            assert "_target_" in metric_cfg


class TestBuildExecutorConfig:
    """Tests for build_executor_config function.

    Note: Uses mock instantiate because YAML configs may have extra fields
    (like 'description') that dataclasses don't accept.
    """

    @pytest.fixture
    def sample_cfg(self) -> DictConfig:
        """Create sample DictConfig for testing."""
        return OmegaConf.create({
            "db": {"host": "localhost", "port": 5432, "user": "test", "password": "test"},
            "db_name": "test_schema",
            "pipelines": [
                {"_target_": "some.module.Pipeline1", "name": "pipeline1"},
                {"_target_": "some.module.Pipeline2", "name": "pipeline2"},
            ],
            "metrics": [{"_target_": "some.module.Metric1", "name": "metric1"}],
            "max_retries": 3,
            "eval_batch_size": 32,
        })

    @patch("autorag_research.cli.commands.run.instantiate")
    def test_builds_executor_config(self, mock_instantiate: MagicMock, sample_cfg: DictConfig) -> None:
        """Builds ExecutorConfig by instantiating each pipeline and metric."""
        mock_instantiate.return_value = MagicMock()

        result = build_executor_config(sample_cfg)

        assert isinstance(result, ExecutorConfig)
        # 2 pipelines + 1 metric = 3 instantiate calls
        assert mock_instantiate.call_count == 3

    @patch("autorag_research.cli.commands.run.instantiate")
    def test_preserves_max_retries(self, mock_instantiate: MagicMock, sample_cfg: DictConfig) -> None:
        """Preserves max_retries from config."""
        mock_instantiate.return_value = MagicMock()

        result = build_executor_config(sample_cfg)

        assert result.max_retries == 3

    @patch("autorag_research.cli.commands.run.instantiate")
    def test_preserves_eval_batch_size(self, mock_instantiate: MagicMock, sample_cfg: DictConfig) -> None:
        """Preserves eval_batch_size from config."""
        mock_instantiate.return_value = MagicMock()

        result = build_executor_config(sample_cfg)

        assert result.eval_batch_size == 32


class TestCreateSessionFactory:
    """Tests for create_session_factory function using real database.

    Note: create_session_factory returns (sessionmaker, db_url) tuple.
    """

    def test_creates_working_session_factory(self, real_db_cfg: DictConfig) -> None:
        """Creates a session factory that can connect to real test database."""
        factory, _ = create_session_factory(real_db_cfg)

        assert callable(factory)

        session = factory()
        try:
            assert isinstance(session, Session)
            from sqlalchemy import text

            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1
        finally:
            session.close()

    def test_returns_sessionmaker_and_db_url(self, real_db_cfg: DictConfig) -> None:
        """Returns a tuple of (sessionmaker, db_url)."""
        result = create_session_factory(real_db_cfg)

        assert isinstance(result, tuple)
        assert len(result) == 2
        factory, db_url = result
        assert callable(factory)
        assert "postgresql" in db_url

    def test_handles_env_var_password(
        self, test_db_params: dict[str, str | int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Handles ${oc.env:...} password pattern by resolving from environment."""
        monkeypatch.setenv("PGPASSWORD", str(test_db_params["password"]))

        cfg = OmegaConf.create({
            "db": {
                "host": test_db_params["host"],
                "port": test_db_params["port"],
                "user": test_db_params["user"],
                "password": "${oc.env:PGPASSWORD}",
                "database": test_db_params["database"],
            }
        })

        factory, _ = create_session_factory(cfg)
        session = factory()
        try:
            from sqlalchemy import text

            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1
        finally:
            session.close()


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
        assert "0.85" in captured.out or "85" in captured.out

    def test_prints_summary(self, mock_result: MagicMock, capsys: pytest.CaptureFixture) -> None:
        """Prints summary with completion count."""
        print_results(mock_result)

        captured = capsys.readouterr()
        assert "2" in captured.out
