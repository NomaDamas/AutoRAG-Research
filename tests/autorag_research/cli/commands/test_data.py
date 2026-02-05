"""Tests for autorag_research.cli.commands.data module."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from autorag_research.cli.app import app
from autorag_research.orm.connection import DBConnection


@pytest.mark.data
@patch("autorag_research.data.hf_storage.download_dump")
def test_dump_and_restore_commands(
    mock_download: MagicMock, cli_runner: CliRunner, db_connection: DBConnection, tmp_path: Path
) -> None:
    config_path = Path(__file__).parent.parent.parent.parent.parent / "configs"

    dump_filepath = tmp_path / "test.dump"
    result = cli_runner.invoke(
        app,
        [
            "--config-path",
            str(config_path),
            "data",
            "dump",
            "--db-name",
            db_connection.database,
            "--output",
            str(dump_filepath),
        ],
    )
    assert result.exit_code == 0
    assert "Dumped to:" in result.stdout
    assert dump_filepath.exists()

    mock_download.return_value = dump_filepath
    result = cli_runner.invoke(
        app, ["--config-path", str(config_path), "data", "restore", "beir", "havertz", "--db-name", "havertz_test"]
    )

    assert result.exit_code == 0
    assert "Downloaded" in result.stdout

    new_conn = deepcopy(db_connection)
    new_conn.database = "havertz_test"

    session = new_conn.get_session_factory()
    with session() as sess:
        from sqlalchemy import text

        result = sess.execute(text("SELECT COUNT(*) FROM document"))
        doc_count = result.scalar()
        assert doc_count > 0, "Documents should be restored"

        result = sess.execute(text("SELECT COUNT(*) FROM chunk"))
        chunk_count = result.scalar()
        assert chunk_count > 0, "Chunks should be restored"

    new_conn.terminate_connections()
    new_conn.drop_database()


class TestDataUploadCommand:
    """Tests for 'data upload' command."""

    def test_upload_help_shows_options(self, cli_runner: CliRunner) -> None:
        """'data upload --help' shows available options."""
        result = cli_runner.invoke(app, ["data", "upload", "--help"])

        assert result.exit_code == 0
        assert "FILE" in result.stdout or "file" in result.stdout.lower()
        assert "INGESTOR" in result.stdout or "ingestor" in result.stdout.lower()

    @patch("autorag_research.data.hf_storage.upload_dump")
    def test_upload_success(self, mock_upload: MagicMock, cli_runner: CliRunner, tmp_path: Path) -> None:
        """'data upload' successfully uploads file."""
        dump_file = tmp_path / "test.dump"
        dump_file.write_text("test content")
        mock_upload.return_value = "https://huggingface.co/datasets/NomaDamas/beir-dumps/blob/main/test.dump"

        result = cli_runner.invoke(app, ["data", "upload", str(dump_file), "beir", "test_dump"])

        assert result.exit_code == 0
        assert "Uploaded:" in result.stdout
        mock_upload.assert_called_once_with(dump_file, "beir", "test_dump", repo_id=None, commit_message=None)

    @patch("autorag_research.data.hf_storage.upload_dump")
    def test_upload_with_custom_repo(self, mock_upload: MagicMock, cli_runner: CliRunner, tmp_path: Path) -> None:
        """'data upload --repo' uses custom repo ID."""
        dump_file = tmp_path / "test.dump"
        dump_file.write_text("test content")
        mock_upload.return_value = "https://huggingface.co/datasets/myorg/custom-repo/blob/main/test.dump"

        result = cli_runner.invoke(
            app, ["data", "upload", str(dump_file), "beir", "test_dump", "--repo", "myorg/custom-repo"]
        )

        assert result.exit_code == 0
        assert "Uploaded:" in result.stdout
        mock_upload.assert_called_once_with(
            dump_file, "beir", "test_dump", repo_id="myorg/custom-repo", commit_message=None
        )

    @patch("autorag_research.data.hf_storage.upload_dump")
    def test_upload_with_custom_repo_short_option(
        self, mock_upload: MagicMock, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """'data upload -r' uses custom repo ID (short option)."""
        dump_file = tmp_path / "test.dump"
        dump_file.write_text("test content")
        mock_upload.return_value = "https://huggingface.co/datasets/myorg/custom-repo/blob/main/test.dump"

        result = cli_runner.invoke(
            app, ["data", "upload", str(dump_file), "beir", "test_dump", "-r", "myorg/custom-repo"]
        )

        assert result.exit_code == 0
        assert "Uploaded:" in result.stdout
        mock_upload.assert_called_once_with(
            dump_file, "beir", "test_dump", repo_id="myorg/custom-repo", commit_message=None
        )

    @patch("autorag_research.data.hf_storage.upload_dump")
    def test_upload_with_custom_repo_and_message(
        self, mock_upload: MagicMock, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """'data upload --repo --message' uses both options."""
        dump_file = tmp_path / "test.dump"
        dump_file.write_text("test content")
        mock_upload.return_value = "https://huggingface.co/datasets/myorg/custom-repo/blob/main/test.dump"

        result = cli_runner.invoke(
            app,
            [
                "data",
                "upload",
                str(dump_file),
                "beir",
                "test_dump",
                "--repo",
                "myorg/custom-repo",
                "-m",
                "Custom upload",
            ],
        )

        assert result.exit_code == 0
        assert "Uploaded:" in result.stdout
        mock_upload.assert_called_once_with(
            dump_file, "beir", "test_dump", repo_id="myorg/custom-repo", commit_message="Custom upload"
        )
