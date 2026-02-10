"""Embedding models for AutoRAG-Research."""

from autorag_research.embeddings.base import (
    MultiVectorBaseEmbedding,
    MultiVectorEmbedding,
    MultiVectorMultiModalEmbedding,
)

__all__ = [
    "BiPaliEmbeddings",
    "ColPaliEmbeddings",
    "InfinityEmbeddings",
    "MultiVectorBaseEmbedding",
    "MultiVectorEmbedding",
    "MultiVectorMultiModalEmbedding",
]

# Mapping of class names to their module paths for lazy loading
_EMBEDDING_MODULES: dict[str, str] = {
    "ColPaliEmbeddings": "autorag_research.embeddings.colpali",
    "BiPaliEmbeddings": "autorag_research.embeddings.bipali",
    "InfinityEmbeddings": "autorag_research.embeddings.infinity",
}


def __getattr__(name: str):
    """Lazy import for embeddings with optional dependencies."""
    if name in _EMBEDDING_MODULES:
        import importlib

        module = importlib.import_module(_EMBEDDING_MODULES[name])
        return getattr(module, name)

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
