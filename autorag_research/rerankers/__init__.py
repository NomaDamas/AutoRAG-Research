"""Reranker wrappers for AutoRAG-Research."""

from autorag_research.rerankers.base import BaseReranker, RerankResult

__all__ = [
    "BaseReranker",
    "CohereReranker",
    "JinaReranker",
    "MixedbreadAIReranker",
    "RankGPTReranker",
    "RerankResult",
    "UPRReranker",
    "VoyageAIReranker",
]

# Mapping of class names to their module paths for lazy loading
_RERANKER_MODULES: dict[str, str] = {
    "CohereReranker": "autorag_research.rerankers.cohere",
    "JinaReranker": "autorag_research.rerankers.jina",
    "VoyageAIReranker": "autorag_research.rerankers.voyageai",
    "MixedbreadAIReranker": "autorag_research.rerankers.mixedbreadai",
    "RankGPTReranker": "autorag_research.rerankers.rankgpt",
    "UPRReranker": "autorag_research.rerankers.upr",
}


def __getattr__(name: str):
    """Lazy import for rerankers with optional dependencies."""
    if name in _RERANKER_MODULES:
        import importlib

        module = importlib.import_module(_RERANKER_MODULES[name])
        return getattr(module, name)

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
