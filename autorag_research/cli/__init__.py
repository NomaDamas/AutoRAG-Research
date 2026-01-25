"""AutoRAG-Research CLI module."""

from pathlib import Path

# Global config path, set by main_callback() at CLI startup
CONFIG_PATH: Path | None = None

__all__ = ["CONFIG_PATH"]
