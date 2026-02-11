"""Tests for CLI run command."""

from unittest.mock import MagicMock, patch

from autorag_research.cli.app import app
from autorag_research.cli.commands.run import build_executor_config
from autorag_research.config import ExecutorConfig


class TestBuildExecutorConfig:
    """Test cases for build_executor_config."""

    def test_build_executor_config_returns_executor_config(self):
        """build_executor_config returns an ExecutorConfig with instantiated objects."""
        fake_pipeline_cfg = MagicMock(name="pipeline_cfg")
        fake_metric_cfg = MagicMock(name="metric_cfg")
        instantiated_pipeline = MagicMock(name="instantiated_pipeline")
        instantiated_metric = MagicMock(name="instantiated_metric")

        def mock_instantiate(cfg):
            if cfg is fake_pipeline_cfg:
                return instantiated_pipeline
            if cfg is fake_metric_cfg:
                return instantiated_metric
            raise ValueError(cfg)

        with patch("autorag_research.cli.commands.run.instantiate", side_effect=mock_instantiate):
            result = build_executor_config(
                pipelines=[fake_pipeline_cfg],
                metrics=[fake_metric_cfg],
                max_retries=5,
                eval_batch_size=50,
            )

        assert isinstance(result, ExecutorConfig)
        assert result.pipelines == [instantiated_pipeline]
        assert result.metrics == [instantiated_metric]
        assert result.max_retries == 5
        assert result.eval_batch_size == 50

    def test_build_executor_config_does_not_mutate_input_lists(self):
        """build_executor_config must not modify the original pipeline/metric lists."""
        pipeline_cfgs = [MagicMock(name="p1"), MagicMock(name="p2")]
        metric_cfgs = [MagicMock(name="m1")]
        original_pipeline_len = len(pipeline_cfgs)
        original_metric_len = len(metric_cfgs)

        with patch("autorag_research.cli.commands.run.instantiate", return_value=MagicMock()):
            build_executor_config(pipelines=pipeline_cfgs, metrics=metric_cfgs)

        assert len(pipeline_cfgs) == original_pipeline_len
        assert len(metric_cfgs) == original_metric_len

    def test_build_executor_config_with_empty_lists(self):
        """build_executor_config handles empty pipeline and metric lists."""
        with patch("autorag_research.cli.commands.run.instantiate") as mock_inst:
            result = build_executor_config(pipelines=[], metrics=[])

        assert result.pipelines == []
        assert result.metrics == []
        mock_inst.assert_not_called()


class TestRunCommand:
    """Test cases for run_command CLI."""

    def test_run_command_loads_config_and_calls_executor(self, cli_runner, real_config_path):
        """run command loads config and calls Executor.run."""
        mock_result = MagicMock()
        mock_result.pipeline_results = []
        mock_result.metric_results = []

        # Mock ExecutorConfig to avoid Hydra instantiation issues
        mock_executor_config = MagicMock()

        with (
            patch("autorag_research.cli.commands.run.DBConnection") as mock_db_conn_class,
            patch("autorag_research.cli.commands.run.Executor") as mock_executor_class,
            patch(
                "autorag_research.cli.commands.run.build_executor_config",
                return_value=mock_executor_config,
            ),
        ):
            # Setup DBConnection mock
            mock_db_conn = MagicMock()
            mock_db_conn.database = "test_db"
            mock_db_conn.host = "localhost"
            mock_db_conn.port = 5432
            mock_db_conn.get_schema.return_value = MagicMock()
            mock_db_conn.get_session_factory.return_value = MagicMock()
            mock_db_conn_class.from_config.return_value = mock_db_conn

            # Setup Executor mock
            mock_executor = MagicMock()
            mock_executor.run.return_value = mock_result
            mock_executor_class.return_value = mock_executor

            result = cli_runner.invoke(app, ["run"])

            # Verify Executor was instantiated and run was called
            assert result.exit_code == 0
            mock_executor_class.assert_called_once()
            mock_executor.run.assert_called_once()

    def test_run_command_fails_without_db_name(self, cli_runner, real_config_path, tmp_path):
        """run command exits with code 1 when db_name is missing."""
        # Create experiment.yaml without db_name
        config_without_db = tmp_path / "configs"
        config_without_db.mkdir()

        experiment_yaml = config_without_db / "experiment.yaml"
        experiment_yaml.write_text(
            """
# Missing db_name
max_retries: 3
pipelines:
  retrieval: [bm25]
metrics:
  retrieval: [recall]
"""
        )

        # Override config path to use tmp_path
        with patch(
            "autorag_research.cli.commands.run.get_config_dir",
            return_value=config_without_db,
        ):
            result = cli_runner.invoke(app, ["run"])

        assert result.exit_code == 1
        assert "db_name is required" in result.output

    def test_run_command_fails_with_missing_config(self, cli_runner, real_config_path):
        """run command exits with code 1 when config file doesn't exist."""
        result = cli_runner.invoke(app, ["run", "--config-name=nonexistent"])

        assert result.exit_code == 1
        assert "Config file not found" in result.output
