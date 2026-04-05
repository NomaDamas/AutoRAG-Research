"""HEAVEN retrieval pipeline for visually rich retrieval workloads.

This staged AutoRAG adaptation keeps the paper's two-step structure while
mapping it onto the repository's existing multimodal primitives:

1. Stage 1: single-vector candidate generation over ImageChunk embeddings
2. Stage 2: multi-vector reranking over the stage-1 candidate set
   with a linguistically guided query-vector budget

The paper's VS-Page preprocessing is intentionally treated as an offline
indexing concern. In this repository, `ImageChunk` is the retrieval unit.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any

import nltk
from langchain_core.embeddings import Embeddings
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.embeddings.base import MultiVectorBaseEmbedding
from autorag_research.exceptions import EmbeddingError
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def _estimate_key_vector_count(query_text: str, total_query_vectors: int, default_keep_ratio: float) -> int:
    """Estimate how many query vectors to keep for key-token scoring.

    The paper filters query tokens by linguistic importance. AutoRAG does not
    persist token-to-vector alignments, so this approximation derives a vector
    budget from noun density in the raw query text.
    """
    if total_query_vectors <= 0:
        return 0

    tokens = _TOKEN_PATTERN.findall(query_text.lower())
    ratio = default_keep_ratio
    if tokens:
        try:
            tagged_tokens = nltk.pos_tag(tokens)
        except LookupError:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            tagged_tokens = nltk.pos_tag(tokens)
        noun_count = sum(1 for _, tag in tagged_tokens if tag.startswith("NN"))
        if noun_count > 0:
            ratio = noun_count / len(tokens)

    keep_count = math.ceil(total_query_vectors * ratio)
    return max(1, min(total_query_vectors, keep_count))


def _split_query_vectors(
    query_vectors: list[list[float]],
    key_vector_count: int,
) -> tuple[list[list[float]], list[list[float]]]:
    """Split query vectors into key and non-key groups."""
    key_vector_count = max(0, min(len(query_vectors), key_vector_count))
    return query_vectors[:key_vector_count], query_vectors[key_vector_count:]


def _refine_candidate_ids(
    stage1_results: list[dict[str, Any]],
    key_scores: dict[int | str, float],
    refine_count: int,
) -> list[int | str]:
    """Select the candidate IDs that advance to the non-key reranking stage."""
    if refine_count <= 0:
        return []

    stage1_order = {result["doc_id"]: index for index, result in enumerate(stage1_results)}
    ranked = sorted(
        key_scores.items(),
        key=lambda item: (-item[1], stage1_order.get(item[0], math.inf)),
    )
    return [doc_id for doc_id, _ in ranked[:refine_count]]


def _combine_heaven_scores(
    stage1_results: list[dict[str, Any]],
    key_scores: dict[int | str, float],
    non_key_scores: dict[int | str, float],
    refine_count: int,
    stage1_weight: float,
    top_k: int,
) -> list[dict[str, Any]]:
    """Combine stage-1, key-token, and non-key-token scores."""
    refined_ids = set(_refine_candidate_ids(stage1_results, key_scores, refine_count))
    combined_results: list[dict[str, Any]] = []

    for result in stage1_results:
        doc_id = result["doc_id"]
        stage1_score = float(result["score"])
        key_score = key_scores.get(doc_id, 0.0)
        non_key_score = non_key_scores.get(doc_id, 0.0) if doc_id in refined_ids else 0.0
        final_score = stage1_weight * stage1_score + (1 - stage1_weight) * (key_score + non_key_score)
        combined_results.append({"doc_id": doc_id, "score": final_score})

    combined_results.sort(key=lambda item: item["score"], reverse=True)
    return combined_results[:top_k]


@dataclass(kw_only=True)
class HEAVENPipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for the HEAVEN retrieval pipeline."""

    stage1_candidate_count: int = 200
    stage2_refine_ratio: float = 0.25
    stage1_weight: float = 0.3
    default_key_token_ratio: float = 0.5
    single_vector_embedding_model: Embeddings | str | None = field(default=None)
    multi_vector_embedding_model: MultiVectorBaseEmbedding | str | None = field(default=None)

    def get_pipeline_class(self) -> type["HEAVENRetrievalPipeline"]:
        """Return the HEAVENRetrievalPipeline class."""
        return HEAVENRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for HEAVENRetrievalPipeline constructor."""
        return {
            "stage1_candidate_count": self.stage1_candidate_count,
            "stage2_refine_ratio": self.stage2_refine_ratio,
            "stage1_weight": self.stage1_weight,
            "default_key_token_ratio": self.default_key_token_ratio,
            "single_vector_embedding_model": self.single_vector_embedding_model,
            "multi_vector_embedding_model": self.multi_vector_embedding_model,
        }

    def __setattr__(self, name: str, value: Any) -> None:
        """Load embedding configs passed by name."""
        if name in {"single_vector_embedding_model", "multi_vector_embedding_model"} and isinstance(value, str):
            from autorag_research.injection import load_embedding_model

            value = load_embedding_model(value)
        super().__setattr__(name, value)


class HEAVENRetrievalPipeline(BaseRetrievalPipeline):
    """Two-stage HEAVEN retrieval pipeline over ImageChunk records."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        stage1_candidate_count: int = 200,
        stage2_refine_ratio: float = 0.25,
        stage1_weight: float = 0.3,
        default_key_token_ratio: float = 0.5,
        single_vector_embedding_model: Embeddings | None = None,
        multi_vector_embedding_model: MultiVectorBaseEmbedding | None = None,
        schema: Any | None = None,
    ):
        if stage1_candidate_count <= 0:
            msg = "stage1_candidate_count must be positive"
            raise ValueError(msg)
        if not 0 < stage2_refine_ratio <= 1:
            msg = "stage2_refine_ratio must be in (0, 1]"
            raise ValueError(msg)
        if not 0 <= stage1_weight <= 1:
            msg = "stage1_weight must be in [0, 1]"
            raise ValueError(msg)
        if not 0 < default_key_token_ratio <= 1:
            msg = "default_key_token_ratio must be in (0, 1]"
            raise ValueError(msg)

        self.stage1_candidate_count = stage1_candidate_count
        self.stage2_refine_ratio = stage2_refine_ratio
        self.stage1_weight = stage1_weight
        self.default_key_token_ratio = default_key_token_ratio
        self._single_vector_embedding_model = single_vector_embedding_model
        self._multi_vector_embedding_model = multi_vector_embedding_model

        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return HEAVEN pipeline configuration."""
        return {
            "type": "heaven",
            "retrieval_unit": "image_chunk",
            "stage1_candidate_count": self.stage1_candidate_count,
            "stage2_refine_ratio": self.stage2_refine_ratio,
            "stage1_weight": self.stage1_weight,
            "default_key_token_ratio": self.default_key_token_ratio,
        }

    def _fetch_stored_query(self, query_id: int | str) -> tuple[str, list[float], list[list[float]]]:
        """Fetch stored query text and embeddings from the database."""
        with self._service._create_uow() as uow:
            query = uow.queries.get_by_id(query_id)
            if query is None:
                raise ValueError(f"Query {query_id} not found")  # noqa: TRY003
            if query.embedding is None:
                raise ValueError(f"Query {query_id} has no single-vector embedding")  # noqa: TRY003
            if query.embeddings is None:
                raise ValueError(f"Query {query_id} has no multi-vector embeddings")  # noqa: TRY003
            return (
                query.contents,
                list(query.embedding),
                [list(vector) for vector in query.embeddings],
            )

    def _run_stage1_search_from_embedding(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Run stage-1 single-vector search over ImageChunk embeddings."""
        with self._service._create_uow() as uow:
            results = uow.image_chunks.vector_search_with_scores(query_vector=query_embedding, limit=limit)
            return [
                {
                    "doc_id": image_chunk.id,
                    "score": 1 - distance,
                    "content": image_chunk.contents,
                }
                for image_chunk, distance in results
            ]

    def _fetch_candidate_multi_embeddings(
        self,
        candidate_ids: list[int | str],
    ) -> dict[int | str, list[list[float]]]:
        """Fetch multi-vector embeddings for the candidate image chunks."""
        if not candidate_ids:
            return {}

        with self._service._create_uow() as uow:
            image_chunks = uow.image_chunks.get_by_ids(candidate_ids)

        image_chunk_map = {image_chunk.id: image_chunk for image_chunk in image_chunks}
        candidate_embeddings: dict[int | str, list[list[float]]] = {}
        for candidate_id in candidate_ids:
            image_chunk = image_chunk_map.get(candidate_id)
            if image_chunk is None or image_chunk.embeddings is None:
                continue
            candidate_embeddings[candidate_id] = [list(vector) for vector in image_chunk.embeddings]
        return candidate_embeddings

    @staticmethod
    def _score_candidates(
        query_vectors: list[list[float]],
        candidate_embeddings: dict[int | str, list[list[float]]],
    ) -> dict[int | str, float]:
        """Score candidates with MaxSim-style late interaction."""
        if not query_vectors:
            return dict.fromkeys(candidate_embeddings, 0.0)

        scores: dict[int | str, float] = {}
        for candidate_id, doc_vectors in candidate_embeddings.items():
            if not doc_vectors:
                scores[candidate_id] = 0.0
                continue

            total_score = 0.0
            for query_vector in query_vectors:
                total_score += max(
                    sum(query_dim * doc_dim for query_dim, doc_dim in zip(query_vector, doc_vector, strict=True))
                    for doc_vector in doc_vectors
                )
            scores[candidate_id] = total_score / len(query_vectors)
        return scores

    def _run_heaven_search(
        self,
        query_text: str,
        single_vector_query: list[float],
        multi_vector_query: list[list[float]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Execute the full two-stage HEAVEN retrieval flow."""
        candidate_limit = max(top_k, self.stage1_candidate_count)
        stage1_results = self._run_stage1_search_from_embedding(single_vector_query, candidate_limit)
        if not stage1_results:
            return []

        candidate_ids = [result["doc_id"] for result in stage1_results]
        candidate_embeddings = self._fetch_candidate_multi_embeddings(candidate_ids)
        if not candidate_embeddings or not multi_vector_query:
            return stage1_results[:top_k]

        key_vector_count = _estimate_key_vector_count(
            query_text=query_text,
            total_query_vectors=len(multi_vector_query),
            default_keep_ratio=self.default_key_token_ratio,
        )
        key_vectors, non_key_vectors = _split_query_vectors(multi_vector_query, key_vector_count)
        key_scores = self._score_candidates(key_vectors, candidate_embeddings)

        refine_count = min(
            len(candidate_ids),
            max(top_k, math.ceil(len(candidate_ids) * self.stage2_refine_ratio)),
        )
        refined_ids = _refine_candidate_ids(stage1_results, key_scores, refine_count)
        refined_embeddings = {
            candidate_id: candidate_embeddings[candidate_id]
            for candidate_id in refined_ids
            if candidate_id in candidate_embeddings
        }
        non_key_scores = self._score_candidates(non_key_vectors, refined_embeddings) if non_key_vectors else {}

        return _combine_heaven_scores(
            stage1_results=stage1_results,
            key_scores=key_scores,
            non_key_scores=non_key_scores,
            refine_count=refine_count,
            stage1_weight=self.stage1_weight,
            top_k=top_k,
        )

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve image chunks for a stored query."""
        query_text, single_vector_query, multi_vector_query = self._fetch_stored_query(query_id)
        return self._run_heaven_search(query_text, single_vector_query, multi_vector_query, top_k)

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve image chunks for an ad-hoc query."""
        if self._single_vector_embedding_model is None or self._multi_vector_embedding_model is None:
            raise EmbeddingError

        single_vector_query = await self._single_vector_embedding_model.aembed_query(query_text)
        multi_vector_query = await self._multi_vector_embedding_model.aembed_query(query_text)
        return self._run_heaven_search(query_text, single_vector_query, multi_vector_query, top_k)

    def run(
        self,
        top_k: int = 10,
        batch_size: int = 128,
        max_concurrency: int = 16,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        query_limit: int | None = None,
    ) -> dict[str, Any]:
        """Run HEAVEN over stored queries and persist image retrieval results."""
        return self._service.run_image_pipeline(
            retrieval_func=self._retrieve_by_id,
            pipeline_id=self.pipeline_id,
            top_k=top_k,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            retry_delay=retry_delay,
            query_limit=query_limit,
        )


__all__ = [
    "HEAVENPipelineConfig",
    "HEAVENRetrievalPipeline",
    "_combine_heaven_scores",
    "_estimate_key_vector_count",
]
