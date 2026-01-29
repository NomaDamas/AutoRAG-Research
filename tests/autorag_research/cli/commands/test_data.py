"""Tests for autorag_research.cli.commands.data module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from autorag_research.cli.app import app


class TestDataDownloadCommand:
    """Tests for 'data download' command."""

    def test_download_help_shows_options(self, cli_runner: CliRunner) -> None:
        """'data download --help' shows available options."""
        result = cli_runner.invoke(app, ["data", "download", "--help"])

        assert result.exit_code == 0
        assert "INGESTOR" in result.stdout or "ingestor" in result.stdout.lower()
        assert "FILENAME" in result.stdout or "filename" in result.stdout.lower()

    @patch("autorag_research.data.hf_storage.download_dump")
    def test_download_success(self, mock_download: MagicMock, cli_runner: CliRunner) -> None:
        """'data download' successfully downloads file."""
        mock_download.return_value = Path("/cache/path/scifact_openai-small.dump")

        result = cli_runner.invoke(app, ["data", "download", "beir", "scifact_openai-small"])

        assert result.exit_code == 0
        assert "Downloaded:" in result.stdout
        mock_download.assert_called_once_with("beir", "scifact_openai-small")

    @patch("autorag_research.data.hf_storage.download_dump")
    def test_download_unknown_ingestor(self, mock_download: MagicMock, cli_runner: CliRunner) -> None:
        """'data download' with unknown ingestor shows error."""
        mock_download.side_effect = KeyError("Unknown ingestor or no HF repo configured: 'unknown'")

        result = cli_runner.invoke(app, ["data", "download", "unknown", "somefile"])

        assert result.exit_code == 1
        # Error is written to stderr, use output which combines stdout+stderr
        assert "Unknown ingestor" in result.output or "unknown" in result.output.lower()

    @patch("autorag_research.data.hf_storage.download_dump")
    def test_download_file_not_found(self, mock_download: MagicMock, cli_runner: CliRunner) -> None:
        """'data download' with non-existent file shows error."""
        from huggingface_hub.utils import EntryNotFoundError

        mock_download.side_effect = EntryNotFoundError("File not found")

        result = cli_runner.invoke(app, ["data", "download", "beir", "nonexistent"])

        assert result.exit_code == 1
        # Error is written to stderr, use output which combines stdout+stderr
        assert "not found" in result.output.lower()


class TestDataRestoreCommand:
    """Tests for 'data restore' command."""

    def test_restore_help_shows_options(self, cli_runner: CliRunner) -> None:
        """'data restore --help' shows available options."""
        result = cli_runner.invoke(app, ["data", "restore", "--help"])

        assert result.exit_code == 0
        assert "--db-name" in result.stdout
        assert "--clean" in result.stdout
        assert "--no-owner" in result.stdout

    @patch("autorag_research.orm.connection.DBConnection.from_config")
    @patch("autorag_research.data.hf_storage.download_dump")
    def test_restore_success(
        self, mock_download: MagicMock, mock_from_config: MagicMock, cli_runner: CliRunner
    ) -> None:
        """'data restore' successfully restores database."""
        mock_download.return_value = Path("/cache/path/scifact_openai-small.dump")
        mock_db_conn = MagicMock()
        mock_from_config.return_value = mock_db_conn

        result = cli_runner.invoke(app, ["data", "restore", "beir", "scifact_openai-small"])

        assert result.exit_code == 0
        assert "restored successfully" in result.stdout
        mock_download.assert_called_once_with("beir", "scifact_openai-small")
        mock_db_conn.restore_database.assert_called_once()

    @patch("autorag_research.orm.connection.DBConnection.from_config")
    @patch("autorag_research.data.hf_storage.download_dump")
    def test_restore_with_custom_db_name(
        self, mock_download: MagicMock, mock_from_config: MagicMock, cli_runner: CliRunner
    ) -> None:
        """'data restore' uses custom database name."""
        mock_download.return_value = Path("/cache/path/scifact_openai-small.dump")
        mock_db_conn = MagicMock()
        mock_from_config.return_value = mock_db_conn

        result = cli_runner.invoke(app, ["data", "restore", "beir", "scifact_openai-small", "--db-name=my_custom_db"])

        assert result.exit_code == 0
        assert mock_db_conn.database == "my_custom_db"

    @patch("autorag_research.orm.connection.DBConnection.from_config")
    @patch("autorag_research.data.hf_storage.download_dump")
    def test_restore_defaults_db_name_to_filename(
        self, mock_download: MagicMock, mock_from_config: MagicMock, cli_runner: CliRunner
    ) -> None:
        """'data restore' defaults database name to filename."""
        mock_download.return_value = Path("/cache/path/scifact_openai-small.dump")
        mock_db_conn = MagicMock()
        mock_from_config.return_value = mock_db_conn

        result = cli_runner.invoke(app, ["data", "restore", "beir", "scifact_openai-small"])

        assert result.exit_code == 0
        assert mock_db_conn.database == "scifact_openai-small"

    def test_restore_clean_requires_confirmation(self, cli_runner: CliRunner) -> None:
        """'data restore --clean' prompts for confirmation."""
        # Without --yes, user input is required
        result = cli_runner.invoke(app, ["data", "restore", "beir", "scifact_openai-small", "--clean"])

        # Should show confirmation prompt
        assert "DROP and recreate" in result.stdout or "Continue?" in result.stdout

    def test_restore_clean_user_declines(self, cli_runner: CliRunner) -> None:
        """'data restore --clean' aborts when user declines."""
        # Provide 'n' input to decline
        result = cli_runner.invoke(app, ["data", "restore", "beir", "scifact_openai-small", "--clean"], input="n\n")

        # Should abort
        assert result.exit_code == 0
        assert "Aborted" in result.stdout

    @patch("autorag_research.orm.connection.DBConnection.from_config")
    @patch("autorag_research.data.hf_storage.download_dump")
    def test_restore_clean_with_yes_flag(
        self, mock_download: MagicMock, mock_from_config: MagicMock, cli_runner: CliRunner
    ) -> None:
        """'data restore --clean --yes' skips confirmation."""
        mock_download.return_value = Path("/cache/path/scifact_openai-small.dump")
        mock_db_conn = MagicMock()
        mock_from_config.return_value = mock_db_conn

        result = cli_runner.invoke(app, ["data", "restore", "beir", "scifact_openai-small", "--clean", "--yes"])

        assert result.exit_code == 0
        assert "restored successfully" in result.stdout
        mock_db_conn.restore_database.assert_called_once()

    @patch("autorag_research.orm.connection.DBConnection.from_config")
    @patch("autorag_research.data.hf_storage.download_dump")
    def test_restore_clean_with_y_flag(
        self, mock_download: MagicMock, mock_from_config: MagicMock, cli_runner: CliRunner
    ) -> None:
        """'data restore --clean -y' skips confirmation (short form)."""
        mock_download.return_value = Path("/cache/path/scifact_openai-small.dump")
        mock_db_conn = MagicMock()
        mock_from_config.return_value = mock_db_conn

        result = cli_runner.invoke(app, ["data", "restore", "beir", "scifact_openai-small", "--clean", "-y"])

        assert result.exit_code == 0
        assert "restored successfully" in result.stdout

    @patch("autorag_research.orm.connection.DBConnection.from_config")
    @patch("autorag_research.data.hf_storage.download_dump")
    def test_restore_without_clean_no_confirmation_needed(
        self, mock_download: MagicMock, mock_from_config: MagicMock, cli_runner: CliRunner
    ) -> None:
        """'data restore' without --clean does not require confirmation."""
        mock_download.return_value = Path("/cache/path/scifact_openai-small.dump")
        mock_db_conn = MagicMock()
        mock_from_config.return_value = mock_db_conn

        result = cli_runner.invoke(app, ["data", "restore", "beir", "scifact_openai-small"])

        assert result.exit_code == 0
        assert "restored successfully" in result.stdout


class TestDataDumpCommand:
    """Tests for 'data dump' command."""

    def test_dump_help_shows_options(self, cli_runner: CliRunner) -> None:
        """'data dump --help' shows available options."""
        result = cli_runner.invoke(app, ["data", "dump", "--help"])

        assert result.exit_code == 0
        assert "--db-name" in result.stdout
        assert "--output" in result.stdout

    @patch("autorag_research.orm.connection.DBConnection.from_config")
    def test_dump_success(self, mock_from_config: MagicMock, cli_runner: CliRunner) -> None:
        """'data dump' successfully dumps database."""
        mock_db_conn = MagicMock()
        mock_from_config.return_value = mock_db_conn
        mock_db_conn.dump_database.return_value = Path("my_db.dump")

        result = cli_runner.invoke(app, ["data", "dump", "--db-name=my_db"])

        assert result.exit_code == 0
        assert "Dumped to:" in result.stdout
        mock_db_conn.dump_database.assert_called_once()

    @patch("autorag_research.orm.connection.DBConnection.from_config")
    def test_dump_with_custom_output(self, mock_from_config: MagicMock, cli_runner: CliRunner) -> None:
        """'data dump' uses custom output path."""
        mock_db_conn = MagicMock()
        mock_from_config.return_value = mock_db_conn
        mock_db_conn.dump_database.return_value = Path("./backup.dump")

        result = cli_runner.invoke(app, ["data", "dump", "--db-name=my_db", "--output=./backup.dump"])

        assert result.exit_code == 0
        # Verify the output path was passed
        call_args = mock_db_conn.dump_database.call_args
        assert call_args[0][0] == Path("./backup.dump")

    def test_dump_requires_db_name(self, cli_runner: CliRunner) -> None:
        """'data dump' requires --db-name option."""
        result = cli_runner.invoke(app, ["data", "dump"])

        assert result.exit_code != 0
        # Error is written to stderr, use output which combines stdout+stderr
        assert "--db-name" in result.output.lower() or "missing" in result.output.lower()

    @patch("autorag_research.orm.connection.DBConnection.from_config")
    def test_dump_runtime_error(self, mock_from_config: MagicMock, cli_runner: CliRunner) -> None:
        """'data dump' handles pg_dump failure."""
        mock_db_conn = MagicMock()
        mock_from_config.return_value = mock_db_conn
        mock_db_conn.dump_database.side_effect = RuntimeError("pg_dump command not found")

        result = cli_runner.invoke(app, ["data", "dump", "--db-name=my_db"])

        assert result.exit_code == 1
        # Error is written to stderr, use output which combines stdout+stderr
        assert "failed" in result.output.lower() or "pg_dump" in result.output


class TestDataUploadCommand:
    """Tests for 'data upload' command."""

    def test_upload_help_shows_options(self, cli_runner: CliRunner) -> None:
        """'data upload --help' shows available options."""
        result = cli_runner.invoke(app, ["data", "upload", "--help"])

        assert result.exit_code == 0
        assert "FILE" in result.stdout or "file" in result.stdout.lower()
        assert "INGESTOR" in result.stdout or "ingestor" in result.stdout.lower()
        assert "--message" in result.stdout

    @patch("autorag_research.data.hf_storage.upload_dump")
    def test_upload_success(self, mock_upload: MagicMock, cli_runner: CliRunner, tmp_path: Path) -> None:
        """'data upload' successfully uploads file."""
        dump_file = tmp_path / "test.dump"
        dump_file.write_text("test content")
        mock_upload.return_value = "https://huggingface.co/datasets/NomaDamas/beir-dumps/blob/main/test.dump"

        result = cli_runner.invoke(app, ["data", "upload", str(dump_file), "beir", "test_dump"])

        assert result.exit_code == 0
        assert "Uploaded:" in result.stdout
        mock_upload.assert_called_once_with(dump_file, "beir", "test_dump", commit_message=None)

    @patch("autorag_research.data.hf_storage.upload_dump")
    def test_upload_with_message(self, mock_upload: MagicMock, cli_runner: CliRunner, tmp_path: Path) -> None:
        """'data upload' passes commit message."""
        dump_file = tmp_path / "test.dump"
        dump_file.write_text("test content")
        mock_upload.return_value = "https://huggingface.co/..."

        result = cli_runner.invoke(app, ["data", "upload", str(dump_file), "beir", "test_dump", "-m", "Custom message"])

        assert result.exit_code == 0
        mock_upload.assert_called_once_with(dump_file, "beir", "test_dump", commit_message="Custom message")

    def test_upload_file_not_found(self, cli_runner: CliRunner) -> None:
        """'data upload' with non-existent file shows error."""
        result = cli_runner.invoke(app, ["data", "upload", "/nonexistent/file.dump", "beir", "test"])

        assert result.exit_code == 1
        # Error is written to stderr, use output which combines stdout+stderr
        assert "not found" in result.output.lower() or "File not found" in result.output

    @patch("autorag_research.data.hf_storage.upload_dump")
    def test_upload_auth_error(self, mock_upload: MagicMock, cli_runner: CliRunner, tmp_path: Path) -> None:
        """'data upload' handles authentication errors."""
        from huggingface_hub.utils import HfHubHTTPError

        dump_file = tmp_path / "test.dump"
        dump_file.write_text("test content")
        mock_upload.side_effect = HfHubHTTPError("401 Unauthorized")

        result = cli_runner.invoke(app, ["data", "upload", str(dump_file), "beir", "test"])

        assert result.exit_code == 1
        # Error is written to stderr, use output which combines stdout+stderr
        assert "Authentication" in result.output or "HF_TOKEN" in result.output
