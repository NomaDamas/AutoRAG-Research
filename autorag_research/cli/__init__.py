"""AutoRAG-Research CLI module."""

from pathlib import Path

# Global config path, set by main_callback() at CLI startup
CONFIG_PATH: Path | None = None


def main() -> None:
    """CLI entry point."""
    from autorag_research.cli.app import main as app_main

    app_main()


__all__ = ["CONFIG_PATH", "main"]
