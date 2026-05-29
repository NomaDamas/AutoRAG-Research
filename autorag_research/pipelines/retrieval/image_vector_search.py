"""ImageChunk vector retrieval pipeline for multimodal late-interaction models.

The built-in ``vector_search`` pipeline searches the text ``chunk`` table. ViDoRe
datasets ingest document pages as ``image_chunk`` rows, so ColPali/ColFlor-style
evaluation needs an image-specific retrieval pipeline that persists results into
``image_chunk_retrieved_result``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.embeddings.base import MultiVectorBaseEmbedding, SingleVectorMultiModalEmbedding
from autorag_research.exceptions import EmbeddingError
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline


@dataclass(kw_only=True)
class ImageVectorSearchPipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for ImageChunk vector search.

    ``search_mode=multi`` uses stored query ``embeddings`` and image_chunk
    ``embeddings`` with VectorChord MaxSim. This is the mode used for ColFlor.
    ``search_mode=single`` is included for completeness.
    """

    search_mode: Literal["single", "multi"] = field(default="multi")
    embedding_model: SingleVectorMultiModalEmbedding | MultiVectorBaseEmbedding | str | None = field(default=None)

    def get_pipeline_class(self) -> type[ImageVectorSearchRetrievalPipeline]:
        return ImageVectorSearchRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        return {"search_mode": self.search_mode, "embedding_model": self.embedding_model}

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "embedding_model" and isinstance(value, str):
            from autorag_research.injection import load_embedding_model

            value = load_embedding_model(value)
        super().__setattr__(name, value)


class ImageVectorSearchRetrievalPipeline(BaseRetrievalPipeline):
    """Vector search over ``image_chunk`` rows."""

    retrieval_unit = "image_chunk"

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        search_mode: Literal["single", "multi"] = "multi",
        embedding_model: SingleVectorMultiModalEmbedding | MultiVectorBaseEmbedding | None = None,
        schema: Any | None = None,
    ):
        self.search_mode = search_mode
        self._embedding_model = embedding_model
        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        return {"type": "image_vector_search", "retrieval_unit": self.retrieval_unit, "search_mode": self.search_mode}

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        with self._service._create_uow() as uow:
            query = uow.queries.get_by_id(query_id)
            if query is None:
                raise ValueError(f"Query {query_id} not found")  # noqa: TRY003

            if self.search_mode == "multi":
                if query.embeddings is None:
                    raise ValueError(f"Query {query_id} has no multi-vector embeddings")  # noqa: TRY003
                query_vectors = [list(vector) for vector in query.embeddings]
                n_query_vectors = max(1, len(query_vectors))
                results = uow.image_chunks.maxsim_search(
                    query_vectors=query_vectors,
                    vector_column="embeddings",
                    limit=top_k,
                )
                return [
                    {"doc_id": image_chunk.id, "score": -distance / n_query_vectors, "content": None}
                    for image_chunk, distance in results
                ]

            if query.embedding is None:
                raise ValueError(f"Query {query_id} has no single-vector embedding")  # noqa: TRY003
            results = uow.image_chunks.vector_search_with_scores(query_vector=list(query.embedding), limit=top_k)
            return [
                {"doc_id": image_chunk.id, "score": 1 - distance, "content": None} for image_chunk, distance in results
            ]

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        if self._embedding_model is None:
            raise EmbeddingError

        with self._service._create_uow() as uow:
            if self.search_mode == "multi":
                query_vectors = await self._embedding_model.aembed_query(query_text)
                n_query_vectors = max(1, len(query_vectors))
                results = uow.image_chunks.maxsim_search(
                    query_vectors=query_vectors,  # type: ignore[arg-type]
                    vector_column="embeddings",
                    limit=top_k,
                )
                return [
                    {"doc_id": image_chunk.id, "score": -distance / n_query_vectors, "content": None}
                    for image_chunk, distance in results
                ]

            query_vector = await self._embedding_model.aembed_query(query_text)
            results = uow.image_chunks.vector_search_with_scores(query_vector=query_vector, limit=top_k)
            return [
                {"doc_id": image_chunk.id, "score": 1 - distance, "content": None} for image_chunk, distance in results
            ]


__all__ = ["ImageVectorSearchPipelineConfig", "ImageVectorSearchRetrievalPipeline"]
