"""Tests for autorag_research.cli.commands.drop module."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from autorag_research.cli.app import app


class TestDropDatabaseCommand:
    """Tests for 'drop database' command."""

    @staticmethod
    def _combined_output(result) -> str:
        return f"{result.stdout}\n{result.stderr}".lower()

    def test_drop_help_shows_options(self, cli_runner: CliRunner) -> None:
        """'drop database --help' shows available options."""
        result = cli_runner.invoke(app, ["drop", "database", "--help"])

        assert result.exit_code == 0
        output = self._combined_output(result)
        assert "database name to drop" in output
        assert "skip confirmation prompts" in output

    def test_drop_requires_db_name(self, cli_runner: CliRunner) -> None:
        """'drop database' requires --db-name option."""
        result = cli_runner.invoke(app, ["drop", "database"])

        assert result.exit_code != 0
        output = self._combined_output(result)
        assert "missing option" in output
        assert "autorag-research drop database" in output

    @patch("autorag_research.orm.connection.DBConnection.from_config")
    def test_drop_protects_system_database(self, mock_from_config: MagicMock, cli_runner: CliRunner) -> None:
        """Protected system databases cannot be dropped."""
        result = cli_runner.invoke(app, ["drop", "database", "--db-name", "postgres", "--yes"])

        assert result.exit_code == 1
        assert "Refusing to drop protected system database" in result.stderr
        mock_from_config.assert_not_called()

    @patch("autorag_research.orm.connection.DBConnection.from_config")
    def test_drop_aborts_when_confirmation_is_no(self, mock_from_config: MagicMock, cli_runner: CliRunner) -> None:
        """Prompted drop exits cleanly when user declines."""
        result = cli_runner.invoke(
            app,
            ["drop", "database", "--db-name", "my_db"],
            input="n\n",
        )

        assert result.exit_code == 0
        assert "Aborted." in result.stdout
        mock_from_config.assert_not_called()

    @patch("autorag_research.orm.connection.DBConnection.from_config")
    def test_drop_skips_confirmation_with_yes(self, mock_from_config: MagicMock, cli_runner: CliRunner) -> None:
        """--yes bypasses confirmation and executes drop."""
        mock_conn = MagicMock()
        mock_from_config.return_value = mock_conn

        result = cli_runner.invoke(app, ["drop", "database", "--db-name", "my_db", "--yes"])

        assert result.exit_code == 0
        assert "Continue?" not in result.stdout
        assert "Database 'my_db' dropped successfully." in result.stdout
        assert mock_conn.database == "my_db"
        mock_conn.terminate_connections.assert_called_once()
        mock_conn.drop_database.assert_called_once()

    @patch("autorag_research.orm.connection.DBConnection.from_config", side_effect=FileNotFoundError)
    def test_drop_fails_without_config(self, _: MagicMock, cli_runner: CliRunner) -> None:
        """Shows init hint when config file is missing."""
        result = cli_runner.invoke(app, ["drop", "database", "--db-name", "my_db", "--yes"])

        assert result.exit_code == 1
        assert "Config file not found. Run 'autorag-research init' first." in result.stderr

    @patch("autorag_research.orm.connection.DBConnection.from_config")
    def test_drop_fails_on_runtime_error(self, mock_from_config: MagicMock, cli_runner: CliRunner) -> None:
        """Shows failure message when drop fails."""
        mock_conn = MagicMock()
        mock_conn.drop_database.side_effect = RuntimeError("permission denied")
        mock_from_config.return_value = mock_conn

        result = cli_runner.invoke(app, ["drop", "database", "--db-name", "my_db", "--yes"])

        assert result.exit_code == 1
        assert "Drop failed: permission denied" in result.stderr
