"""
Base module definitions for AutoRAG nodes.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseModule(ABC):
    """
    Base class for all AutoRAG modules.

    Modules are reusable functional units that can be used across multiple pipelines.
    They follow a pure function pattern where outputs are determined by inputs.
    Model dependencies (rerankers, embeddings, LLMs) are injected from outside.
    """

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """
        Execute the module's main functionality.

        This method should be implemented by all subclasses.
        """
        pass
