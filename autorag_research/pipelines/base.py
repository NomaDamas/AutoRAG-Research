from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy.orm import Session, sessionmaker


class BasePipeline(ABC):
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        schema: Any | None = None,
    ):
        self.session_factory = session_factory
        self.name = name
        self._schema = schema

    @abstractmethod
    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return the pipeline configuration dictionary.

        Returns:
            Configuration dict to store in the pipeline table.
        """
        pass

    @abstractmethod
    def run(self, *args, **kwargs) -> dict[str, Any]:
        """Run the pipeline.

        Returns:
            Results of the pipeline execution.
        """
        pass
