"""Tests for autorag_research.cli.commands.health_check module."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from autorag_research.cli.app import app

_INJECTION = "autorag_research.injection"


@pytest.fixture
def cli_runner() -> CliRunner:
    """Return a Typer CliRunner for testing commands."""
    return CliRunner(env={"NO_COLOR": "1"})


class TestHealthCheckHelp:
    """Tests for health-check command help text."""

    def test_help_displays_correctly(self, cli_runner: CliRunner) -> None:
        """Health-check --help shows usage information."""
        result = cli_runner.invoke(app, ["health-check", "--help"])

        assert result.exit_code == 0
        assert "embedding" in result.stdout
        assert "llm" in result.stdout
        assert "reranker" in result.stdout


class TestHealthCheckEmbedding:
    """Tests for health-check with embedding model type."""

    @patch(f"{_INJECTION}.health_check_embedding", return_value=384)
    @patch(f"{_INJECTION}.load_embedding_model")
    def test_embedding_success_shows_dimension(
        self,
        mock_load: MagicMock,
        mock_health_check: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Successful embedding check shows PASS with dimension."""
        mock_load.return_value = MagicMock()

        result = cli_runner.invoke(app, ["health-check", "embedding", "mock"])

        assert "[PASS] mock (dimension: 384)" in result.stdout
        assert result.exit_code == 0

    @patch(f"{_INJECTION}.load_embedding_model", side_effect=RuntimeError("API key invalid"))
    def test_embedding_failure_shows_error(
        self,
        mock_load: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Failed embedding check shows FAIL with error message."""
        result = cli_runner.invoke(app, ["health-check", "embedding", "bad-model"])

        assert "[FAIL] bad-model" in result.stdout
        assert "API key invalid" in result.stdout
        assert result.exit_code == 1


class TestHealthCheckLLM:
    """Tests for health-check with llm model type."""

    @patch(f"{_INJECTION}.load_llm")
    def test_llm_success_shows_pass(
        self,
        mock_load: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Successful LLM check shows PASS."""
        mock_load.return_value = MagicMock()

        result = cli_runner.invoke(app, ["health-check", "llm", "mock-llm"])

        assert "[PASS] mock-llm" in result.stdout
        assert result.exit_code == 0

    @patch(f"{_INJECTION}.load_llm", side_effect=ConnectionError("Connection refused"))
    def test_llm_failure_shows_error(
        self,
        mock_load: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Failed LLM check shows FAIL with error message."""
        result = cli_runner.invoke(app, ["health-check", "llm", "bad-llm"])

        assert "[FAIL] bad-llm" in result.stdout
        assert "Connection refused" in result.stdout
        assert result.exit_code == 1


class TestHealthCheckReranker:
    """Tests for health-check with reranker model type."""

    @patch(f"{_INJECTION}.load_reranker")
    def test_reranker_success_shows_pass(
        self,
        mock_load: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Successful reranker check shows PASS."""
        mock_load.return_value = MagicMock()

        result = cli_runner.invoke(app, ["health-check", "reranker", "mock-reranker"])

        assert "[PASS] mock-reranker" in result.stdout
        assert result.exit_code == 0

    @patch(f"{_INJECTION}.load_reranker", side_effect=FileNotFoundError("Config file not found"))
    def test_reranker_failure_shows_error(
        self,
        mock_load: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Failed reranker check shows FAIL with error message."""
        result = cli_runner.invoke(app, ["health-check", "reranker", "bad-reranker"])

        assert "[FAIL] bad-reranker" in result.stdout
        assert result.exit_code == 1
