"""Tests for autorag_research.cli.commands.init_config module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

import autorag_research.cli as cli
from autorag_research.cli.app import app
from autorag_research.cli.commands.init import fetch_config_files_from_github, init


@pytest.fixture
def mock_github_api_response() -> list[dict]:
    """Create mock GitHub API response with files and directories."""
    return [
        {"name": "db.yaml", "type": "file", "path": "configs/db.yaml"},
        {"name": "pipelines", "type": "dir", "path": "configs/pipelines"},
        {"name": "metrics", "type": "dir", "path": "configs/metrics"},
    ]


@pytest.fixture
def mock_github_pipelines_response() -> list[dict]:
    """Create mock GitHub API response for pipelines directory."""
    return [
        {"name": "bm25.yaml", "type": "file", "path": "configs/pipelines/bm25.yaml"},
        {"name": "dense.yaml", "type": "file", "path": "configs/pipelines/dense.yaml"},
    ]


@pytest.fixture
def mock_github_metrics_response() -> list[dict]:
    """Create mock GitHub API response for metrics directory."""
    return [
        {"name": "ndcg.yaml", "type": "file", "path": "configs/metrics/ndcg.yaml"},
    ]


class TestInitConfigCommand:
    """Tests for the init CLI command."""

    def test_init_help(self, cli_runner: CliRunner) -> None:
        """'init --help' shows help."""
        result = cli_runner.invoke(app, ["init", "--help"])

        assert result.exit_code == 0

    @patch("autorag_research.cli.commands.init.httpx.Client")
    def test_init_config_creates_directory(
        self, mock_client_class: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """init creates config directory if it doesn't exist."""
        config_dir = tmp_path / "new_configs"
        monkeypatch.setattr(cli, "CONFIG_PATH", config_dir)

        # Setup mock client
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

        # Mock fetch to return a simple file
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"name": "db.yaml", "type": "file", "path": "configs/db.yaml"}]
        mock_response.text = "host: localhost"
        mock_client.get.return_value = mock_response

        init()

        assert config_dir.exists()

    @patch("autorag_research.cli.commands.init_config.httpx.Client")
    def test_init_config_downloads_files(
        self, mock_client_class: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """init_config downloads files from GitHub."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        monkeypatch.setattr(cli, "CONFIG_PATH", config_dir)

        # Setup mock client
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

        # Mock API response
        api_response = MagicMock()
        api_response.status_code = 200
        api_response.json.return_value = [{"name": "db.yaml", "type": "file", "path": "configs/db.yaml"}]

        # Mock raw file response
        raw_response = MagicMock()
        raw_response.status_code = 200
        raw_response.text = "host: localhost\nport: 5432"

        mock_client.get.side_effect = [api_response, raw_response]

        init()

        assert (config_dir / "db.yaml").exists()
        assert "localhost" in (config_dir / "db.yaml").read_text()

    @patch("autorag_research.cli.commands.init_config.httpx.Client")
    def test_init_config_skips_existing_files(
        self,
        mock_client_class: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """init_config skips files that already exist."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        # Create existing file
        (config_dir / "db.yaml").write_text("existing content")
        monkeypatch.setattr(cli, "CONFIG_PATH", config_dir)

        # Setup mock client
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

        # Mock API response
        api_response = MagicMock()
        api_response.status_code = 200
        api_response.json.return_value = [{"name": "db.yaml", "type": "file", "path": "configs/db.yaml"}]
        mock_client.get.return_value = api_response

        init()

        # File should still have original content
        assert (config_dir / "db.yaml").read_text() == "existing content"

    @patch("autorag_research.cli.commands.init_config.httpx.Client")
    def test_init_config_raises_on_empty_response(
        self, mock_client_class: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """init_config raises RuntimeError when no files found."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        monkeypatch.setattr(cli, "CONFIG_PATH", config_dir)

        # Setup mock client
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

        # Mock API response with empty list
        api_response = MagicMock()
        api_response.status_code = 200
        api_response.json.return_value = []
        mock_client.get.return_value = api_response

        with pytest.raises(RuntimeError, match="Failed to fetch config files"):
            init()


class TestFetchConfigFilesFromGitHub:
    """Tests for fetch_config_files_from_github function."""

    def test_returns_yaml_files_only(
        self, mock_github_api_response: list[dict], mock_github_pipelines_response: list[dict]
    ) -> None:
        """Only returns .yaml and .yml files."""
        mock_client = MagicMock()

        # First call returns root with file and directory
        root_response = MagicMock()
        root_response.status_code = 200
        root_response.json.return_value = [
            {"name": "db.yaml", "type": "file", "path": "configs/db.yaml"},
            {"name": "README.md", "type": "file", "path": "configs/README.md"},
            {"name": "pipelines", "type": "dir", "path": "configs/pipelines"},
        ]

        # Second call returns pipelines directory
        pipelines_response = MagicMock()
        pipelines_response.status_code = 200
        pipelines_response.json.return_value = mock_github_pipelines_response

        mock_client.get.side_effect = [root_response, pipelines_response]

        result = fetch_config_files_from_github(mock_client)

        # Should not include README.md
        assert "README.md" not in result
        assert "db.yaml" in result

    def test_returns_sorted_list(self) -> None:
        """Returns files in sorted order."""
        mock_client = MagicMock()

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = [
            {"name": "zebra.yaml", "type": "file", "path": "configs/zebra.yaml"},
            {"name": "alpha.yaml", "type": "file", "path": "configs/alpha.yaml"},
            {"name": "middle.yaml", "type": "file", "path": "configs/middle.yaml"},
        ]
        mock_client.get.return_value = response

        result = fetch_config_files_from_github(mock_client)

        assert result == sorted(result)

    def test_handles_nested_directories(
        self,
        mock_github_api_response: list[dict],
        mock_github_pipelines_response: list[dict],
        mock_github_metrics_response: list[dict],
    ) -> None:
        """Recursively fetches files from nested directories."""
        mock_client = MagicMock()

        # Setup responses for each directory
        root_response = MagicMock()
        root_response.status_code = 200
        root_response.json.return_value = mock_github_api_response

        pipelines_response = MagicMock()
        pipelines_response.status_code = 200
        pipelines_response.json.return_value = mock_github_pipelines_response

        metrics_response = MagicMock()
        metrics_response.status_code = 200
        metrics_response.json.return_value = mock_github_metrics_response

        mock_client.get.side_effect = [root_response, pipelines_response, metrics_response]

        result = fetch_config_files_from_github(mock_client)

        assert "db.yaml" in result
        assert "pipelines/bm25.yaml" in result
        assert "metrics/ndcg.yaml" in result

    def test_handles_api_error(self) -> None:
        """Returns empty list on API error."""
        mock_client = MagicMock()

        response = MagicMock()
        response.status_code = 403
        mock_client.get.return_value = response

        result = fetch_config_files_from_github(mock_client)

        assert result == []

    def test_strips_configs_prefix(self) -> None:
        """Strips 'configs/' prefix from returned paths."""
        mock_client = MagicMock()

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = [
            {"name": "db.yaml", "type": "file", "path": "configs/db.yaml"},
        ]
        mock_client.get.return_value = response

        result = fetch_config_files_from_github(mock_client)

        assert "db.yaml" in result
        assert "configs/db.yaml" not in result
