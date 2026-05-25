"""Text retriever-reranker retrieval pipeline for AutoRAG-Research.

This pipeline wraps an existing text chunk retrieval pipeline, fetches a larger
candidate set, and reranks those candidate texts with a configured reranker. It
keeps the standard retrieval pipeline interface so generation pipelines can
compose with it by referencing the configured pipeline name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.orm.uow.retrieval_uow import RetrievalUnitOfWork
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.rerankers.base import BaseReranker, RerankResult

TEXT_RETRIEVAL_UNIT = "chunk"


def _get_wrapped_pipeline_config(pipeline: BaseRetrievalPipeline) -> dict[str, Any]:
    """Return the wrapped pipeline config when it is available as a dict."""
    get_config = getattr(pipeline, "_get_pipeline_config", None)
    if not callable(get_config):
        return {}

    config = get_config()
    return config if isinstance(config, dict) else {}


def _validate_text_retrieval_pipeline(pipeline: BaseRetrievalPipeline) -> None:
    """Reject wrapped pipelines whose persisted results are not text chunks."""
    config = _get_wrapped_pipeline_config(pipeline)
    retrieval_unit = config.get("retrieval_unit", TEXT_RETRIEVAL_UNIT)
    if retrieval_unit != TEXT_RETRIEVAL_UNIT:
        pipeline_type = config.get("type", type(pipeline).__name__)
        msg = (
            "RerankRetrievalPipeline only supports text chunk retrieval pipelines "
            f"(retrieval_unit='chunk'); got retrieval_unit={retrieval_unit!r} from {pipeline_type!r}."
        )
        raise ValueError(msg)


@dataclass(kw_only=True)
class RerankRetrievalPipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for the text chunk retriever-reranker retrieval wrapper."""

    retrieval_pipeline_name: str
    reranker: str | BaseReranker
    candidate_top_k: int = 50
    _retrieval_pipeline: BaseRetrievalPipeline | None = field(default=None, repr=False)

    def __setattr__(self, name: str, value: Any) -> None:
        """Auto-convert string reranker config names to reranker instances."""
        if name == "reranker" and isinstance(value, str):
            from autorag_research.injection import load_reranker

            value = load_reranker(value)
        super().__setattr__(name, value)

    def inject_retrieval_pipeline(self, pipeline: BaseRetrievalPipeline) -> None:
        """Inject the wrapped text chunk retrieval pipeline instance."""
        _validate_text_retrieval_pipeline(pipeline)
        self._retrieval_pipeline = pipeline

    def get_pipeline_class(self) -> type[RerankRetrievalPipeline]:
        """Return the RerankRetrievalPipeline class."""
        return RerankRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for RerankRetrievalPipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)

        return {
            "retrieval_pipeline": self._retrieval_pipeline,
            "reranker": self.reranker,
            "candidate_top_k": self.candidate_top_k,
        }


class RerankRetrievalPipeline(BaseRetrievalPipeline):
    """Text chunk retriever → reranker retrieval wrapper."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        retrieval_pipeline: BaseRetrievalPipeline,
        reranker: BaseReranker,
        candidate_top_k: int = 50,
        schema: Any | None = None,
    ):
        """Initialize the rerank retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            retrieval_pipeline: Wrapped first-stage text chunk retrieval pipeline.
            reranker: Reranker used to rescore candidate texts.
            candidate_top_k: Number of first-stage candidates to fetch before reranking.
            schema: Optional dynamic schema namespace.
        """
        if candidate_top_k < 1:
            msg = "candidate_top_k must be >= 1"
            raise ValueError(msg)
        _validate_text_retrieval_pipeline(retrieval_pipeline)

        self._retrieval_pipeline = retrieval_pipeline
        self.reranker = reranker
        self.candidate_top_k = candidate_top_k

        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return rerank pipeline configuration for storage."""
        return {
            "type": "rerank",
            "retrieval_unit": TEXT_RETRIEVAL_UNIT,
            "candidate_top_k": self.candidate_top_k,
            "reranker_model": getattr(self.reranker, "model_name", type(self.reranker).__name__),
            "retrieval_pipeline_id": getattr(self._retrieval_pipeline, "pipeline_id", None),
            "wrapped_pipeline_type": type(self._retrieval_pipeline).__name__,
        }

    def _ensure_candidate_contents(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Backfill missing candidate contents from the chunk table."""
        missing_ids = [candidate["doc_id"] for candidate in candidates if not candidate.get("content")]
        if not missing_ids:
            return candidates

        with RetrievalUnitOfWork(self.session_factory, self._schema) as uow:
            chunks = uow.chunks.get_by_ids(missing_ids)

        contents_by_id = {chunk.id: chunk.contents for chunk in chunks}
        unresolved_ids = sorted({doc_id for doc_id in missing_ids if doc_id not in contents_by_id}, key=str)
        if unresolved_ids:
            missing_ids_text = ", ".join(str(doc_id) for doc_id in unresolved_ids)
            msg = f"Missing chunk content for candidate doc_ids: {missing_ids_text}"
            raise ValueError(msg)

        enriched_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            enriched_candidate = dict(candidate)
            if not enriched_candidate.get("content"):
                enriched_candidate["content"] = contents_by_id[candidate["doc_id"]]
            enriched_candidates.append(enriched_candidate)

        return enriched_candidates

    @staticmethod
    def _map_rerank_result(candidate: dict[str, Any], rerank_result: RerankResult) -> dict[str, Any]:
        """Map one reranker result back to the standard retrieval result format."""
        return {
            "doc_id": candidate["doc_id"],
            "score": float(rerank_result.score),
            "content": rerank_result.text,
        }

    def _map_rerank_results(
        self,
        candidates: list[dict[str, Any]],
        rerank_results: list[RerankResult],
    ) -> list[dict[str, Any]]:
        """Map reranker output indexes back to wrapped retrieval candidates."""
        mapped_results: list[dict[str, Any]] = []
        for rerank_result in rerank_results:
            if rerank_result.index < 0 or rerank_result.index >= len(candidates):
                msg = f"Reranker returned out-of-range candidate index: {rerank_result.index}"
                raise ValueError(msg)
            mapped_results.append(self._map_rerank_result(candidates[rerank_result.index], rerank_result))
        return mapped_results

    async def _rerank_candidates(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rerank candidate contents and return standard retrieval results."""
        documents = [str(candidate.get("content") or "") for candidate in candidates]
        rerank_results = await self.reranker.arerank(query_text, documents, top_k=top_k)
        return self._map_rerank_results(candidates, rerank_results)

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        """Fetch query text by ID, then rerank wrapped retrieval candidates."""
        query_texts = self._service.fetch_query_texts([query_id])
        if not query_texts:
            return []

        return await self._retrieve_by_text(query_texts[0], top_k)

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve first-stage candidates from the wrapped pipeline, then rerank them."""
        candidate_top_k = max(top_k, self.candidate_top_k)
        candidates = await self._retrieval_pipeline.retrieve(query_text, candidate_top_k)
        if not candidates:
            return []

        enriched_candidates = self._ensure_candidate_contents(candidates)
        return await self._rerank_candidates(query_text, enriched_candidates, top_k)


__all__ = [
    "RerankRetrievalPipeline",
    "RerankRetrievalPipelineConfig",
]
