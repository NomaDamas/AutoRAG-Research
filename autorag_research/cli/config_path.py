"""Configuration path manager using Singleton pattern."""

import threading
from pathlib import Path
from typing import ClassVar


class ConfigPathManager:
    """Singleton manager for the global configuration directory path.

    This class provides a centralized way to access the config directory
    throughout the application. It must be initialized once (typically in main.py)
    before any access to get_config_dir().

    Example:
        # In CLI entry point (main.py)
        ConfigPathManager.initialize(Path("/custom/configs"))

        # Anywhere else in the application
        config_dir = ConfigPathManager.get_config_dir()
    """

    _instance: ClassVar["ConfigPathManager | None"] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, config_path: Path) -> None:
        """Private constructor. Use initialize() instead."""
        self._config_path = config_path

    @classmethod
    def initialize(cls, config_path: Path | str | None = None) -> "ConfigPathManager":
        """Initialize the singleton with the given config path.

        Args:
            config_path: Path to the config directory. Defaults to CWD/configs.

        Returns:
            The initialized ConfigPathManager instance.

        Note:
            If already initialized, returns the existing instance without modification.
            Call reset() first if you need to reinitialize with a different path.
        """
        with cls._lock:
            if cls._instance is not None:
                return cls._instance

            if config_path is None:
                path = Path.cwd() / "configs"
            elif isinstance(config_path, str):
                path = Path(config_path)
            else:
                path = config_path

            cls._instance = cls(path.resolve())
            return cls._instance

    @classmethod
    def get_instance(cls) -> "ConfigPathManager":
        """Get the singleton instance.

        Returns:
            The ConfigPathManager instance.

        Raises:
            RuntimeError: If not initialized.
        """
        if cls._instance is None:
            raise RuntimeError("ConfigPathManager not initialized. Call ConfigPathManager.initialize() first.")  # noqa: TRY003
        return cls._instance

    @classmethod
    def get_config_dir(cls) -> Path:
        """Get the configuration directory path.

        Returns:
            Path to the config directory.

        Raises:
            RuntimeError: If not initialized.
        """
        return cls.get_instance()._config_path

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the singleton is initialized."""
        return cls._instance is not None

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (primarily for testing)."""
        with cls._lock:
            cls._instance = None
