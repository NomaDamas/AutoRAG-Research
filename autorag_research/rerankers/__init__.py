"""Reranker wrappers for AutoRAG-Research."""

from autorag_research.rerankers.base import BaseReranker, RerankResult

__all__ = [
    "BaseReranker",
    "CohereReranker",
    "ColBERTReranker",
    "FlagEmbeddingLLMReranker",
    "FlagEmbeddingReranker",
    "FlashRankReranker",
    "JinaReranker",
    "KoRerankerReranker",
    "MixedbreadAIReranker",
    "MonoT5Reranker",
    "OpenVINOReranker",
    "RankGPTReranker",
    "RerankResult",
    "SentenceTransformerReranker",
    "TARTReranker",
    "UPRReranker",
    "VoyageAIReranker",
]

# Mapping of class names to their module paths for lazy loading
_RERANKER_MODULES: dict[str, str] = {
    "CohereReranker": "autorag_research.rerankers.cohere",
    "ColBERTReranker": "autorag_research.rerankers.colbert",
    "FlagEmbeddingLLMReranker": "autorag_research.rerankers.flag_embedding_llm",
    "FlagEmbeddingReranker": "autorag_research.rerankers.flag_embedding",
    "FlashRankReranker": "autorag_research.rerankers.flashrank",
    "JinaReranker": "autorag_research.rerankers.jina",
    "VoyageAIReranker": "autorag_research.rerankers.voyageai",
    "KoRerankerReranker": "autorag_research.rerankers.koreranker",
    "MixedbreadAIReranker": "autorag_research.rerankers.mixedbreadai",
    "MonoT5Reranker": "autorag_research.rerankers.monot5",
    "OpenVINOReranker": "autorag_research.rerankers.openvino",
    "RankGPTReranker": "autorag_research.rerankers.rankgpt",
    "SentenceTransformerReranker": "autorag_research.rerankers.sentence_transformer",
    "TARTReranker": "autorag_research.rerankers.tart",
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
