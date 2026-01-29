"""Tests for autorag_research.cli.app module."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import autorag_research.cli as cli
from autorag_research.cli.app import app


class TestMainCallback:
    """Tests for the main_callback function (global options)."""

    def test_config_path_set_from_option(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """--config-path option sets cli.CONFIG_PATH."""
        config_dir = tmp_path / "my_configs"
        config_dir.mkdir()

        cli_runner.invoke(app, ["--config-path", str(config_dir), "list", "pipelines"])

        assert config_dir.resolve() == cli.CONFIG_PATH

    def test_config_path_defaults_to_cwd_configs(self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without --config-path, defaults to ./configs."""
        # Mock cwd to a temp directory
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.chdir(tmpdir)
            cli_runner.invoke(app, ["list", "pipelines"])

            expected = (Path(tmpdir) / "configs").resolve()
            assert expected == cli.CONFIG_PATH

    def test_config_path_from_env_var(
        self, cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AUTORAG_RESEARCH_CONFIG_PATH env var sets config path."""
        config_dir = tmp_path / "env_configs"
        config_dir.mkdir()

        monkeypatch.setenv("AUTORAG_RESEARCH_CONFIG_PATH", str(config_dir))
        cli_runner.invoke(app, ["list", "pipelines"])

        assert config_dir.resolve() == cli.CONFIG_PATH

    def test_cli_option_overrides_env_var(
        self, cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI --config-path takes precedence over env var."""
        env_dir = tmp_path / "env_configs"
        env_dir.mkdir()
        cli_dir = tmp_path / "cli_configs"
        cli_dir.mkdir()

        monkeypatch.setenv("AUTORAG_RESEARCH_CONFIG_PATH", str(env_dir))
        cli_runner.invoke(app, ["--config-path", str(cli_dir), "list", "pipelines"])

        assert cli_dir.resolve() == cli.CONFIG_PATH


class TestVersionFlag:
    """Tests for the --version flag."""

    def test_version_flag_long(self, cli_runner: CliRunner) -> None:
        """--version flag shows version and exits."""
        result = cli_runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert "autorag-research" in result.stdout
        # Version should match pattern like "0.0.1" or "1.2.3"
        import re

        assert re.search(r"\d+\.\d+\.\d+", result.stdout)

    def test_version_flag_short(self, cli_runner: CliRunner) -> None:
        """-V flag shows version and exits."""
        result = cli_runner.invoke(app, ["-V"])

        assert result.exit_code == 0
        assert "autorag-research" in result.stdout

    def test_version_flag_exits_early(self, cli_runner: CliRunner) -> None:
        """--version exits before processing other arguments."""
        # Even with invalid command, --version should work
        result = cli_runner.invoke(app, ["--version", "invalid-command"])

        assert result.exit_code == 0
        assert "autorag-research" in result.stdout


class TestAppStructure:
    """Tests for the Typer app structure."""

    def test_no_args_shows_help(self, cli_runner: CliRunner) -> None:
        """Running without arguments shows help (no_args_is_help=True)."""
        result = cli_runner.invoke(app, [])

        # no_args_is_help shows help but exits with code 2 (not 0)
        assert "AutoRAG-Research CLI" in result.stdout and "Usage:" in result.stdout

    def test_help_flag_shows_help(self, cli_runner: CliRunner) -> None:
        """--help flag displays help text."""
        result = cli_runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "AutoRAG-Research CLI" in result.stdout

    def test_app_has_list_command(self, cli_runner: CliRunner) -> None:
        """App has 'list' command registered."""
        result = cli_runner.invoke(app, ["list", "--help"])

        assert result.exit_code == 0
        assert "list" in result.stdout.lower() and "resources" in result.stdout.lower()

    def test_app_has_ingest_command(self, cli_runner: CliRunner) -> None:
        """App has 'ingest' command registered."""
        result = cli_runner.invoke(app, ["ingest", "--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.stdout

    def test_app_has_init_config_command(self, cli_runner: CliRunner) -> None:
        """App has 'init-config' command registered."""
        result = cli_runner.invoke(app, ["init-config", "--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "init-config" in result.stdout

    def test_app_has_run_command(self, cli_runner: CliRunner) -> None:
        """App has 'run' command registered."""
        result = cli_runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "run" in result.stdout

    def test_unknown_command_shows_error(self, cli_runner: CliRunner) -> None:
        """Unknown command shows error message."""
        result = cli_runner.invoke(app, ["havertz"])

        assert result.exit_code != 0
        assert "No such command" in result.stdout or "Error" in result.stdout or result.exit_code == 2
