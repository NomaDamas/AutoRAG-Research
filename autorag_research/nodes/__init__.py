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

    Subclasses using dynamic schemas should set `_schema` attribute in __init__.
    """

    _schema: Any | None = None

    def _get_chunk_model(self) -> type:
        """Get the Chunk model class from schema or default."""
        if self._schema is not None:
            return self._schema.Chunk
        from autorag_research.orm.schema import Chunk

        return Chunk

    def _get_query_model(self) -> type:
        """Get the Query model class from schema or default."""
        if self._schema is not None:
            return self._schema.Query
        from autorag_research.orm.schema import Query

        return Query

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """
        Execute the module's main functionality.

        This method should be implemented by all subclasses.
        """
        pass


def make_retrieval_result(chunk: Any, score: float) -> dict[str, Any]:
    """Create a standardized retrieval result dictionary.

    Args:
        chunk: Chunk entity with id and contents attributes.
        score: Relevance/similarity score.

    Returns:
        Dictionary with doc_id, score, and content keys.
    """
    return {
        "doc_id": chunk.id,
        "score": score,
        "content": chunk.contents,
    }
