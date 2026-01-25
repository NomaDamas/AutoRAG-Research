"""Shared fixtures for CLI tests."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import autorag_research.cli as cli


@pytest.fixture
def cli_runner() -> CliRunner:
    """Return a Typer CliRunner for testing commands."""
    return CliRunner()


@pytest.fixture
def real_config_path(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set cli.CONFIG_PATH to project root's configs/ directory."""
    project_root = Path(__file__).parent.parent.parent.parent
    config_dir = project_root / "configs"
    monkeypatch.setattr(cli, "CONFIG_PATH", config_dir)
    return config_dir


@pytest.fixture
def reset_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset cli.CONFIG_PATH to None before test."""
    monkeypatch.setattr(cli, "CONFIG_PATH", None)


@pytest.fixture
def real_ingestor_meta():
    """Return the real beir ingestor metadata from the registry."""
    from autorag_research.data.registry import discover_ingestors

    return discover_ingestors()["beir"]
