"""Tests for autorag_research.cli.commands.health_check module."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from autorag_research.cli.app import app

# Patch targets: lazy imports in _check_single_model import from autorag_research.injection
_INJECTION = "autorag_research.injection"
_DISCOVER = "autorag_research.cli.utils.discover_configs"


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
        assert "health-check" in result.stdout.lower() or "health check" in result.stdout.lower()
        assert "embedding" in result.stdout
        assert "--name" in result.stdout


class TestHealthCheckEmbedding:
    """Tests for health-check with embedding model type."""

    @patch(f"{_INJECTION}.health_check_embedding", return_value=384)
    @patch(f"{_INJECTION}.load_embedding_model")
    @patch(_DISCOVER, return_value={"mock": "FakeEmbeddings"})
    def test_embedding_success_shows_dimension(
        self,
        mock_discover: MagicMock,
        mock_load: MagicMock,
        mock_health_check: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Successful embedding check shows PASS with dimension."""
        mock_load.return_value = MagicMock()

        result = cli_runner.invoke(app, ["health-check", "embedding", "--name", "mock"])

        assert "[PASS] mock (dimension: 384)" in result.stdout
        assert "1 passed, 0 failed" in result.stdout
        assert result.exit_code == 0

    @patch(f"{_INJECTION}.load_embedding_model", side_effect=RuntimeError("API key invalid"))
    @patch(_DISCOVER, return_value={"bad-model": "SomeEmbeddings"})
    def test_embedding_failure_shows_error(
        self,
        mock_discover: MagicMock,
        mock_load: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Failed embedding check shows FAIL with error message."""
        result = cli_runner.invoke(app, ["health-check", "embedding", "--name", "bad-model"])

        assert "[FAIL] bad-model" in result.stdout
        assert "API key invalid" in result.stdout
        assert "0 passed, 1 failed" in result.stdout
        assert result.exit_code == 1

    @patch(f"{_INJECTION}.health_check_embedding", return_value=768)
    @patch(f"{_INJECTION}.load_embedding_model")
    @patch(_DISCOVER, return_value={"model-a": "Embeddings", "model-b": "Embeddings"})
    def test_checks_all_configs_when_no_name(
        self,
        mock_discover: MagicMock,
        mock_load: MagicMock,
        mock_health_check: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Checks all embedding configs when --name is not given."""
        mock_load.return_value = MagicMock()

        result = cli_runner.invoke(app, ["health-check", "embedding"])

        assert "[PASS] model-a" in result.stdout
        assert "[PASS] model-b" in result.stdout
        assert "2 passed, 0 failed" in result.stdout
        assert result.exit_code == 0


class TestHealthCheckLLM:
    """Tests for health-check with llm model type."""

    @patch(f"{_INJECTION}.load_llm")
    @patch(_DISCOVER, return_value={"mock-llm": "FakeLLM"})
    def test_llm_success_shows_pass(
        self,
        mock_discover: MagicMock,
        mock_load: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Successful LLM check shows PASS."""
        mock_load.return_value = MagicMock()

        result = cli_runner.invoke(app, ["health-check", "llm", "--name", "mock-llm"])

        assert "[PASS] mock-llm" in result.stdout
        assert "1 passed, 0 failed" in result.stdout
        assert result.exit_code == 0

    @patch(f"{_INJECTION}.load_llm", side_effect=ConnectionError("Connection refused"))
    @patch(_DISCOVER, return_value={"bad-llm": "SomeLLM"})
    def test_llm_failure_shows_error(
        self,
        mock_discover: MagicMock,
        mock_load: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Failed LLM check shows FAIL with error message."""
        result = cli_runner.invoke(app, ["health-check", "llm", "--name", "bad-llm"])

        assert "[FAIL] bad-llm" in result.stdout
        assert "Connection refused" in result.stdout
        assert result.exit_code == 1


class TestHealthCheckReranker:
    """Tests for health-check with reranker model type."""

    @patch(f"{_INJECTION}.load_reranker")
    @patch(_DISCOVER, return_value={"mock-reranker": "FakeReranker"})
    def test_reranker_success_shows_pass(
        self,
        mock_discover: MagicMock,
        mock_load: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Successful reranker check shows PASS."""
        mock_load.return_value = MagicMock()

        result = cli_runner.invoke(app, ["health-check", "reranker", "--name", "mock-reranker"])

        assert "[PASS] mock-reranker" in result.stdout
        assert "1 passed, 0 failed" in result.stdout
        assert result.exit_code == 0


class TestHealthCheckAll:
    """Tests for health-check with 'all' model type."""

    @patch(f"{_INJECTION}.load_reranker")
    @patch(f"{_INJECTION}.load_llm")
    @patch(f"{_INJECTION}.health_check_embedding", return_value=384)
    @patch(f"{_INJECTION}.load_embedding_model")
    @patch(_DISCOVER, return_value={"mock": "SomeModel"})
    def test_all_checks_all_model_types(
        self,
        mock_discover: MagicMock,
        mock_load_emb: MagicMock,
        mock_health_check: MagicMock,
        mock_load_llm: MagicMock,
        mock_load_reranker: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """'all' checks embedding, llm, and reranker."""
        mock_load_emb.return_value = MagicMock()
        mock_load_llm.return_value = MagicMock()
        mock_load_reranker.return_value = MagicMock()

        result = cli_runner.invoke(app, ["health-check", "all"])

        assert "[embedding]" in result.stdout
        assert "[llm]" in result.stdout
        assert "[reranker]" in result.stdout
        assert "3 passed, 0 failed" in result.stdout
        assert result.exit_code == 0


class TestHealthCheckMissingConfig:
    """Tests for missing config scenarios."""

    @patch(_DISCOVER, side_effect=FileNotFoundError)
    def test_missing_config_directory_shows_message(
        self,
        mock_discover: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Missing config directory shows helpful message."""
        result = cli_runner.invoke(app, ["health-check", "embedding"])

        assert "No config directory found" in result.stdout
        assert "autorag-research init" in result.stdout
        assert "0 passed, 0 failed" in result.stdout
        assert result.exit_code == 0

    @patch(_DISCOVER, return_value={"model-a": "Embeddings", "model-b": "Embeddings"})
    def test_missing_config_name_shows_available(
        self,
        mock_discover: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Specifying a non-existent config name shows available configs."""
        result = cli_runner.invoke(app, ["health-check", "embedding", "--name", "nonexistent"])

        assert "Config 'nonexistent' not found" in result.stdout
        assert "model-a" in result.stdout
        assert "model-b" in result.stdout
        assert result.exit_code == 1


class TestHealthCheckExitCode:
    """Tests for exit code behavior."""

    @patch(f"{_INJECTION}.health_check_embedding", return_value=384)
    @patch(f"{_INJECTION}.load_embedding_model")
    @patch(_DISCOVER, return_value={"bad": "Embeddings", "good": "Embeddings"})
    def test_exit_code_1_on_any_failure(
        self,
        mock_discover: MagicMock,
        mock_load: MagicMock,
        mock_health_check: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Exit code is 1 when any model fails."""
        # "bad" comes first alphabetically, so it gets the RuntimeError
        mock_load.side_effect = [RuntimeError("fail"), MagicMock()]

        result = cli_runner.invoke(app, ["health-check", "embedding"])

        assert "[FAIL] bad" in result.stdout
        assert "[PASS] good" in result.stdout
        assert "1 passed, 1 failed" in result.stdout
        assert result.exit_code == 1
