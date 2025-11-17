"""Abstract base classes for dataset ingestion workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DataIngestor(ABC):
    """Template for implementing dataset ingestion pipelines."""

    def __init__(self, dsn: str, *, schema: str | None = None) -> None:
        self.dsn = dsn
        self.schema = schema

    @abstractmethod
    def prepare(self) -> None:
        """Prepare external resources prior to ingestion."""
        pass

    @abstractmethod
    def ingest(self) -> None:
        """Perform the actual ingestion into the target system."""
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Optional hook executed after ingestion succeeds."""
        pass

    def run(self) -> None:
        """Execute the full ingestion lifecycle."""
        self.prepare()
        self.ingest()
        self.finalize()

    def configure(self, **kwargs: Any) -> None:
        """Apply runtime configuration to the ingestor."""
        for key, value in kwargs.items():
            setattr(self, key, value)
