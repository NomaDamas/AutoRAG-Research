"""Guided Query Refinement (GQR) hybrid retrieval pipeline.

This module adapts the GQR test-time optimization workflow to AutoRAG-Research
without adding new dependencies:

1. Retrieve first-pass candidates from a primary retriever.
2. Build complementary guidance scores from another retriever.
3. Optimize the primary query representation over the candidate pool so the
   primary score distribution approaches the hybrid guidance distribution.
4. Rerank the candidate pool with the optimized query representation.

When embedding-level refinement cannot be executed (for example, missing
query/chunk embeddings), the pipeline falls back to a score-space refinement
loop that preserves the same optimization objective.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.pipelines.retrieval.hybrid import HybridRetrievalPipeline
from autorag_research.util import normalize_zscore

CandidatePoolMode = Literal["primary", "union"]

_EPSILON = 1e-8


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Compute numerically stable softmax probabilities."""
    if scores.size == 0:
        return scores

    safe_temperature = max(temperature, _EPSILON)
    shifted = scores / safe_temperature
    shifted = shifted - np.max(shifted)
    exp_scores = np.exp(shifted)
    denom = float(np.sum(exp_scores))
    if not np.isfinite(denom) or denom <= _EPSILON:
        return np.full(scores.shape, 1.0 / scores.size, dtype=np.float64)
    return exp_scores / denom


def _missing_score_floor(score_map: dict[int | str, float]) -> float:
    """Return a conservative floor score for missing documents."""
    if not score_map:
        return -1.0

    values = list(score_map.values())
    min_score = min(values)
    max_score = max(values)
    spread = max_score - min_score
    return min_score - max(1.0, spread)


def _normalize_scores(scores: list[float]) -> np.ndarray:
    """Normalize score vectors with z-score normalization."""
    normalized = normalize_zscore(cast(list[float | None], scores))
    return np.asarray([0.0 if value is None else float(value) for value in normalized], dtype=np.float64)


def _cosine_scores(query_vector: np.ndarray, candidate_matrix: np.ndarray) -> np.ndarray:
    """Compute cosine-similarity scores between query and candidate vectors."""
    query_norm = np.linalg.norm(query_vector)
    if query_norm <= _EPSILON:
        return np.zeros(candidate_matrix.shape[0], dtype=np.float64)

    candidate_norms = np.linalg.norm(candidate_matrix, axis=1)
    safe_candidate_norms = np.maximum(candidate_norms, _EPSILON)
    return (candidate_matrix @ query_vector) / (safe_candidate_norms * query_norm)


def _cosine_gradients(
    query_vector: np.ndarray,
    candidate_matrix: np.ndarray,
    cosine_scores: np.ndarray,
) -> np.ndarray:
    """Compute gradients of cosine scores with respect to query vector."""
    query_norm = np.linalg.norm(query_vector)
    if query_norm <= _EPSILON:
        return np.zeros_like(candidate_matrix)

    candidate_norms = np.linalg.norm(candidate_matrix, axis=1)
    safe_candidate_norms = np.maximum(candidate_norms, _EPSILON)
    left = candidate_matrix / (safe_candidate_norms[:, None] * query_norm)
    right = (cosine_scores[:, None] * query_vector[None, :]) / (query_norm**2)
    return left - right


@dataclass(kw_only=True)
class GQRHybridRetrievalPipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for the GQR hybrid retrieval pipeline."""

    primary_retrieval_pipeline_name: str
    complementary_retrieval_pipeline_name: str
    fetch_k_multiplier: int = 2
    n_steps: int = 25
    learning_rate: float = 0.1
    temperature: float = 1.0
    mixture_alpha: float = 0.5
    candidate_pool_mode: CandidatePoolMode = "primary"

    def __post_init__(self) -> None:
        if self.fetch_k_multiplier <= 0:
            msg = "fetch_k_multiplier must be positive"
            raise ValueError(msg)
        if self.n_steps <= 0:
            msg = "n_steps must be positive"
            raise ValueError(msg)
        if self.learning_rate <= 0:
            msg = "learning_rate must be positive"
            raise ValueError(msg)
        if self.temperature <= 0:
            msg = "temperature must be positive"
            raise ValueError(msg)
        if not 0 <= self.mixture_alpha <= 1:
            msg = "mixture_alpha must be between 0 and 1"
            raise ValueError(msg)
        if self.candidate_pool_mode not in {"primary", "union"}:
            msg = "candidate_pool_mode must be either 'primary' or 'union'"
            raise ValueError(msg)

    def get_pipeline_class(self) -> type["GQRHybridRetrievalPipeline"]:
        """Return the GQRHybridRetrievalPipeline class."""
        return GQRHybridRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for GQRHybridRetrievalPipeline constructor."""
        return {
            "primary_retrieval_pipeline": self.primary_retrieval_pipeline_name,
            "complementary_retrieval_pipeline": self.complementary_retrieval_pipeline_name,
            "fetch_k_multiplier": self.fetch_k_multiplier,
            "n_steps": self.n_steps,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "mixture_alpha": self.mixture_alpha,
            "candidate_pool_mode": self.candidate_pool_mode,
        }


class GQRHybridRetrievalPipeline(BaseRetrievalPipeline):
    """Guided Query Refinement hybrid retrieval pipeline."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        primary_retrieval_pipeline: BaseRetrievalPipeline | str,
        complementary_retrieval_pipeline: BaseRetrievalPipeline | str,
        fetch_k_multiplier: int = 2,
        n_steps: int = 25,
        learning_rate: float = 0.1,
        temperature: float = 1.0,
        mixture_alpha: float = 0.5,
        candidate_pool_mode: CandidatePoolMode = "primary",
        schema: Any | None = None,
        config_dir: Path | None = None,
    ):
        if fetch_k_multiplier <= 0:
            msg = "fetch_k_multiplier must be positive"
            raise ValueError(msg)
        if n_steps <= 0:
            msg = "n_steps must be positive"
            raise ValueError(msg)
        if learning_rate <= 0:
            msg = "learning_rate must be positive"
            raise ValueError(msg)
        if temperature <= 0:
            msg = "temperature must be positive"
            raise ValueError(msg)
        if not 0 <= mixture_alpha <= 1:
            msg = "mixture_alpha must be between 0 and 1"
            raise ValueError(msg)
        if candidate_pool_mode not in {"primary", "union"}:
            msg = "candidate_pool_mode must be either 'primary' or 'union'"
            raise ValueError(msg)

        if isinstance(primary_retrieval_pipeline, str):
            primary_retrieval_pipeline = HybridRetrievalPipeline._load_pipeline(
                primary_retrieval_pipeline,
                session_factory,
                schema,
                config_dir,
            )
        if isinstance(complementary_retrieval_pipeline, str):
            complementary_retrieval_pipeline = HybridRetrievalPipeline._load_pipeline(
                complementary_retrieval_pipeline,
                session_factory,
                schema,
                config_dir,
            )

        self._primary_retrieval_pipeline = primary_retrieval_pipeline
        self._complementary_retrieval_pipeline = complementary_retrieval_pipeline
        self.fetch_k_multiplier = fetch_k_multiplier
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.mixture_alpha = mixture_alpha
        self.candidate_pool_mode = candidate_pool_mode

        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return GQR pipeline configuration."""
        return {
            "type": "gqr_hybrid",
            "primary_retrieval_pipeline": self._primary_retrieval_pipeline.name,
            "complementary_retrieval_pipeline": self._complementary_retrieval_pipeline.name,
            "fetch_k_multiplier": self.fetch_k_multiplier,
            "n_steps": self.n_steps,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "mixture_alpha": self.mixture_alpha,
            "candidate_pool_mode": self.candidate_pool_mode,
        }

    @staticmethod
    def _build_score_map(results: list[dict[str, Any]]) -> dict[int | str, float]:
        """Build doc_id -> score mapping."""
        return {result["doc_id"]: float(result["score"]) for result in results}

    def _build_candidate_ids(
        self,
        primary_results: list[dict[str, Any]],
        complementary_results: list[dict[str, Any]],
    ) -> list[int | str]:
        """Build candidate pool IDs based on configured strategy."""
        if self.candidate_pool_mode == "primary":
            return [result["doc_id"] for result in primary_results]

        seen: set[int | str] = set()
        candidate_ids: list[int | str] = []
        for result in [*primary_results, *complementary_results]:
            doc_id = result["doc_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            candidate_ids.append(doc_id)
        return candidate_ids

    @staticmethod
    def _build_score_vector(candidate_ids: list[int | str], score_map: dict[int | str, float]) -> np.ndarray:
        """Build a normalized score vector aligned to candidate_ids."""
        missing_floor = _missing_score_floor(score_map)
        raw_scores = [score_map.get(doc_id, missing_floor) for doc_id in candidate_ids]
        return _normalize_scores(raw_scores)

    @staticmethod
    def _sort_results(score_map: dict[int | str, float], top_k: int) -> list[dict[str, Any]]:
        """Sort results by score and truncate to top_k."""
        sorted_items = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
        return [{"doc_id": doc_id, "score": score} for doc_id, score in sorted_items[:top_k]]

    def _get_query_embedding_by_id(self, query_id: int | str) -> np.ndarray | None:
        """Fetch stored query embedding for optimization."""
        with self._service._create_uow() as uow:
            query = uow.queries.get_by_id(query_id)
            if query is None or query.embedding is None:
                return None
            return np.asarray(list(query.embedding), dtype=np.float64)

    async def _get_query_embedding_by_text(self, query_text: str) -> np.ndarray | None:
        """Build query embedding for ad-hoc text when the primary pipeline supports it."""
        embedding_model = getattr(self._primary_retrieval_pipeline, "_embedding_model", None)
        if embedding_model is None:
            return None

        query_embedding = await embedding_model.aembed_query(query_text)
        return np.asarray(query_embedding, dtype=np.float64)

    def _get_candidate_embeddings(
        self,
        candidate_ids: list[int | str],
    ) -> tuple[list[int | str], np.ndarray]:
        """Load candidate chunk embeddings from the database."""
        if not candidate_ids:
            return [], np.empty((0, 0), dtype=np.float64)

        with self._service._create_uow() as uow:
            chunks = uow.chunks.get_by_ids(candidate_ids)

        chunk_by_id = {chunk.id: chunk for chunk in chunks}
        embedding_ids: list[int | str] = []
        embedding_rows: list[list[float]] = []
        for doc_id in candidate_ids:
            chunk = chunk_by_id.get(doc_id)
            if chunk is None or chunk.embedding is None:
                continue
            embedding_ids.append(doc_id)
            embedding_rows.append(list(chunk.embedding))

        if not embedding_rows:
            return [], np.empty((0, 0), dtype=np.float64)

        return embedding_ids, np.asarray(embedding_rows, dtype=np.float64)

    def _optimize_in_score_space(
        self,
        primary_scores: np.ndarray,
        target_distribution: np.ndarray,
    ) -> np.ndarray:
        """Fallback optimization directly over primary score logits."""
        logits = primary_scores.astype(np.float64, copy=True)
        temperature = max(self.temperature, _EPSILON)
        for _ in range(self.n_steps):
            probs = _softmax(logits, temperature)
            grad_logits = (probs - target_distribution) / temperature
            logits -= self.learning_rate * grad_logits
        return logits

    def _optimize_query_embedding(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        target_distribution: np.ndarray,
    ) -> np.ndarray:
        """Optimize the primary query embedding with KL guidance."""
        refined_query = query_embedding.astype(np.float64, copy=True)
        temperature = max(self.temperature, _EPSILON)

        for _ in range(self.n_steps):
            cosine_scores = _cosine_scores(refined_query, candidate_embeddings)
            probs = _softmax(cosine_scores, temperature)
            grad_logits = (probs - target_distribution) / temperature
            grad_scores = _cosine_gradients(refined_query, candidate_embeddings, cosine_scores)
            grad_query = np.sum(grad_logits[:, None] * grad_scores, axis=0)
            refined_query -= self.learning_rate * grad_query

        return _cosine_scores(refined_query, candidate_embeddings)

    async def _run_gqr(
        self,
        *,
        top_k: int,
        query_embedding: np.ndarray | None,
        primary_results: list[dict[str, Any]],
        complementary_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Execute GQR refinement and rerank the candidate pool."""
        candidate_ids = self._build_candidate_ids(primary_results, complementary_results)
        if not candidate_ids:
            return []

        primary_score_map = self._build_score_map(primary_results)
        complementary_score_map = self._build_score_map(complementary_results)

        primary_scores = self._build_score_vector(candidate_ids, primary_score_map)
        complementary_scores = self._build_score_vector(candidate_ids, complementary_score_map)

        primary_distribution = _softmax(primary_scores, self.temperature)
        complementary_distribution = _softmax(complementary_scores, self.temperature)
        target_distribution = (
            (1 - self.mixture_alpha) * primary_distribution + self.mixture_alpha * complementary_distribution
        )

        final_score_map: dict[int | str, float] = {
            doc_id: float(primary_scores[index]) for index, doc_id in enumerate(candidate_ids)
        }

        embedding_ids, candidate_embeddings = self._get_candidate_embeddings(candidate_ids)
        if query_embedding is not None and embedding_ids and candidate_embeddings.size > 0:
            id_to_index = {doc_id: idx for idx, doc_id in enumerate(candidate_ids)}
            distribution_indices = [id_to_index[doc_id] for doc_id in embedding_ids]
            embedding_target = target_distribution[distribution_indices]
            embedding_target = embedding_target / max(float(embedding_target.sum()), _EPSILON)
            optimized_scores = self._optimize_query_embedding(
                query_embedding=query_embedding,
                candidate_embeddings=candidate_embeddings,
                target_distribution=embedding_target,
            )
            for doc_id, score in zip(embedding_ids, optimized_scores, strict=True):
                final_score_map[doc_id] = float(score)
        else:
            optimized_scores = self._optimize_in_score_space(primary_scores, target_distribution)
            final_score_map = {
                doc_id: float(optimized_scores[index]) for index, doc_id in enumerate(candidate_ids)
            }

        return self._sort_results(final_score_map, top_k)

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve with Guided Query Refinement using query ID."""
        fetch_k = top_k * self.fetch_k_multiplier
        primary_results = await self._primary_retrieval_pipeline._retrieve_by_id(query_id, fetch_k)
        complementary_results = await self._complementary_retrieval_pipeline._retrieve_by_id(query_id, fetch_k)

        query_embedding = self._get_query_embedding_by_id(query_id)
        return await self._run_gqr(
            top_k=top_k,
            query_embedding=query_embedding,
            primary_results=primary_results,
            complementary_results=complementary_results,
        )

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve with Guided Query Refinement using raw query text."""
        fetch_k = top_k * self.fetch_k_multiplier
        primary_results = await self._primary_retrieval_pipeline._retrieve_by_text(query_text, fetch_k)
        complementary_results = await self._complementary_retrieval_pipeline._retrieve_by_text(query_text, fetch_k)

        query_embedding = await self._get_query_embedding_by_text(query_text)
        return await self._run_gqr(
            top_k=top_k,
            query_embedding=query_embedding,
            primary_results=primary_results,
            complementary_results=complementary_results,
        )


__all__ = ["GQRHybridRetrievalPipeline", "GQRHybridRetrievalPipelineConfig"]
