"""AutoRAG-Research CLI module."""

from pathlib import Path

from autorag_research.cli.main import main

# Global config path, set by main() at CLI startup
CONFIG_PATH: Path | None = None

__all__ = ["CONFIG_PATH", "main"]
