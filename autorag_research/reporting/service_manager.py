"""Reporting service lifecycle management for the leaderboard UI."""

from autorag_research.reporting.service import ReportingService


class _ServiceManager:
    """Manage the ReportingService singleton without a module global."""

    _instance: ReportingService | None = None

    @classmethod
    def get(cls) -> ReportingService:
        """Get or create the reporting service."""
        if cls._instance is None:
            cls._instance = ReportingService()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Close and clear the reporting service."""
        if cls._instance is not None:
            cls._instance.close()
            cls._instance = None


def get_service() -> ReportingService:
    """Get or create the reporting service."""
    return _ServiceManager.get()


def reset_service() -> None:
    """Close and clear the reporting service."""
    _ServiceManager.reset()
