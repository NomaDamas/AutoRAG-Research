"""Guided Query Refinement (GQR) hybrid retrieval pipeline.

This module adapts the GQR test-time optimization workflow to AutoRAG-Research
without adding new dependencies:

1. Retrieve a candidate pool from the union of primary and complementary retrievers.
2. Build primary and complementary distributions from raw native retriever scores.
3. Recompute the per-step consensus target from the current primary distribution
   and the fixed complementary distribution, then optimize the primary query representation.
4. Rerank the candidate pool with the optimized single-vector cosine scores.

When single-vector embedding refinement cannot be executed (for example, missing
query/chunk embeddings or a multi-vector primary retriever without chunk vectors),
the pipeline falls back to a score-space refinement loop. That fallback preserves
the per-step consensus objective but is a degraded AutoRAG-Research adaptation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.exceptions import EmbeddingError
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.pipelines.retrieval.hybrid import HybridRetrievalPipeline

CandidatePoolMode = Literal["primary", "union"]
ScorerMode = Literal["auto", "single", "multi"]

_EPSILON = 1e-8
logger = logging.getLogger("AutoRAG-Research")


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


def _maxsim_scores(query_matrix: np.ndarray, candidate_embeddings: list[np.ndarray]) -> np.ndarray:
    """Compute normalized late-interaction MaxSim scores for candidates."""
    if query_matrix.size == 0:
        return np.zeros(len(candidate_embeddings), dtype=np.float64)

    n_query_vectors = max(query_matrix.shape[0], 1)
    scores: list[float] = []
    for candidate_matrix in candidate_embeddings:
        if candidate_matrix.size == 0:
            scores.append(0.0)
            continue
        similarities = query_matrix @ candidate_matrix.T
        scores.append(float(np.max(similarities, axis=1).sum() / n_query_vectors))
    return np.asarray(scores, dtype=np.float64)


def _maxsim_gradients(query_matrix: np.ndarray, candidate_embeddings: list[np.ndarray]) -> np.ndarray:
    """Compute MaxSim argmax subgradients with respect to query vectors."""
    gradients = np.zeros((len(candidate_embeddings), *query_matrix.shape), dtype=np.float64)
    if query_matrix.size == 0:
        return gradients

    n_query_vectors = max(query_matrix.shape[0], 1)
    for candidate_index, candidate_matrix in enumerate(candidate_embeddings):
        if candidate_matrix.size == 0:
            continue
        similarities = query_matrix @ candidate_matrix.T
        argmax_indices = np.argmax(similarities, axis=1)
        gradients[candidate_index] = candidate_matrix[argmax_indices] / n_query_vectors
    return gradients


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
    candidate_pool_mode: CandidatePoolMode = "union"
    scorer_mode: ScorerMode = "auto"

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
        if self.scorer_mode not in {"auto", "single", "multi"}:
            msg = "scorer_mode must be one of 'auto', 'single', or 'multi'"
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
            "scorer_mode": self.scorer_mode,
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
        candidate_pool_mode: CandidatePoolMode = "union",
        scorer_mode: ScorerMode = "auto",
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
        if scorer_mode not in {"auto", "single", "multi"}:
            msg = "scorer_mode must be one of 'auto', 'single', or 'multi'"
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
        self.scorer_mode = scorer_mode

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
            "scorer_mode": self.scorer_mode,
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
        """Build a raw score vector aligned to candidate_ids."""
        missing_floor = _missing_score_floor(score_map)
        return np.asarray([score_map.get(doc_id, missing_floor) for doc_id in candidate_ids], dtype=np.float64)

    @staticmethod
    def _sort_results(score_map: dict[int | str, float], top_k: int) -> list[dict[str, Any]]:
        """Sort results by score and truncate to top_k."""
        sorted_items = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
        return [{"doc_id": doc_id, "score": score} for doc_id, score in sorted_items[:top_k]]

    def _get_query_embedding_by_id(self, query_id: int | str) -> np.ndarray | None:
        """Fetch stored query embedding for optimization."""
        query_embedding = self._service.get_query_embedding(query_id)
        if query_embedding is None:
            return None
        return np.asarray(query_embedding, dtype=np.float64)

    async def _get_query_embedding_by_text(self, query_text: str) -> np.ndarray | None:
        """Build query embedding for ad-hoc text when the primary pipeline supports it."""
        embedding_model = getattr(self._primary_retrieval_pipeline, "_embedding_model", None)
        if embedding_model is None:
            return None

        query_embedding = await embedding_model.aembed_query(query_text)
        return np.asarray(query_embedding, dtype=np.float64)

    def _get_query_multi_embedding_by_id(self, query_id: int | str) -> np.ndarray | None:
        """Fetch stored multi-vector query embedding for MaxSim optimization."""
        query_embedding = self._service.get_query_multi_embedding(query_id)
        if query_embedding is None:
            return None
        return np.asarray(query_embedding, dtype=np.float64)

    def _get_candidate_embeddings(
        self,
        candidate_ids: list[int | str],
    ) -> tuple[list[int | str], np.ndarray]:
        """Load candidate chunk embeddings through the retrieval service."""
        if not candidate_ids:
            return [], np.empty((0, 0), dtype=np.float64)

        embeddings_by_id = self._service.get_chunk_embeddings(candidate_ids)
        embedding_ids = [doc_id for doc_id in candidate_ids if doc_id in embeddings_by_id]
        if not embedding_ids:
            return [], np.empty((0, 0), dtype=np.float64)

        embedding_rows = [embeddings_by_id[doc_id] for doc_id in embedding_ids]
        return embedding_ids, np.asarray(embedding_rows, dtype=np.float64)

    def _get_candidate_multi_embeddings(
        self,
        candidate_ids: list[int | str],
    ) -> tuple[list[int | str], list[np.ndarray]]:
        """Load candidate multi-vector chunk embeddings through the retrieval service."""
        if not candidate_ids:
            return [], []

        embeddings_by_id = self._service.get_chunk_multi_embeddings(candidate_ids)
        embedding_ids = [doc_id for doc_id in candidate_ids if doc_id in embeddings_by_id]
        return embedding_ids, [np.asarray(embeddings_by_id[doc_id], dtype=np.float64) for doc_id in embedding_ids]

    def _optimize_in_score_space(
        self,
        primary_scores: np.ndarray,
        complementary_distribution: np.ndarray,
    ) -> np.ndarray:
        """Fallback optimization directly over primary score logits with per-step consensus guidance."""
        logits = primary_scores.astype(np.float64, copy=True)
        temperature = max(self.temperature, _EPSILON)
        for _ in range(self.n_steps):
            probs = _softmax(logits, temperature)
            target_distribution = (1 - self.mixture_alpha) * probs + self.mixture_alpha * complementary_distribution
            grad_logits = (probs - target_distribution) / temperature
            logits -= self.learning_rate * grad_logits
        return logits

    def _optimize_query_embedding(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        complementary_distribution: np.ndarray,
    ) -> np.ndarray:
        """Optimize the primary query embedding with per-step KL consensus guidance."""
        refined_query = query_embedding.astype(np.float64, copy=True)
        temperature = max(self.temperature, _EPSILON)

        for _ in range(self.n_steps):
            cosine_scores = _cosine_scores(refined_query, candidate_embeddings)
            probs = _softmax(cosine_scores, temperature)
            target_distribution = (1 - self.mixture_alpha) * probs + self.mixture_alpha * complementary_distribution
            grad_logits = (probs - target_distribution) / temperature
            grad_scores = _cosine_gradients(refined_query, candidate_embeddings, cosine_scores)
            grad_query = np.sum(grad_logits[:, None] * grad_scores, axis=0)
            refined_query -= self.learning_rate * grad_query

        return _cosine_scores(refined_query, candidate_embeddings)

    def _optimize_query_multi_embedding(
        self,
        query_matrix: np.ndarray,
        candidate_embeddings: list[np.ndarray],
        complementary_distribution: np.ndarray,
    ) -> np.ndarray:
        """Optimize a multi-vector query matrix with MaxSim consensus guidance."""
        refined_query = query_matrix.astype(np.float64, copy=True)
        temperature = max(self.temperature, _EPSILON)

        for _ in range(self.n_steps):
            maxsim_scores = _maxsim_scores(refined_query, candidate_embeddings)
            probs = _softmax(maxsim_scores, temperature)
            target_distribution = (1 - self.mixture_alpha) * probs + self.mixture_alpha * complementary_distribution
            grad_logits = (probs - target_distribution) / temperature
            grad_scores = _maxsim_gradients(refined_query, candidate_embeddings)
            grad_query = np.sum(grad_logits[:, None, None] * grad_scores, axis=0)
            refined_query -= self.learning_rate * grad_query

        return _maxsim_scores(refined_query, candidate_embeddings)

    def _resolve_scorer_mode(self) -> Literal["single", "multi"]:
        """Resolve configured scorer mode against the primary pipeline."""
        if self.scorer_mode != "auto":
            return self.scorer_mode
        if getattr(self._primary_retrieval_pipeline, "search_mode", "single") == "multi":
            return "multi"
        return "single"

    async def _run_gqr(
        self,
        *,
        top_k: int,
        query_embedding: np.ndarray | None,
        query_multi_embedding: np.ndarray | None,
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
        complementary_distribution = _softmax(complementary_scores, self.temperature)

        scorer_mode = self._resolve_scorer_mode()
        if scorer_mode == "multi":
            embedding_ids, candidate_multi_embeddings = self._get_candidate_multi_embeddings(candidate_ids)
            all_candidates_have_embeddings = len(embedding_ids) == len(candidate_ids)
            if query_multi_embedding is not None and all_candidates_have_embeddings and candidate_multi_embeddings:
                optimized_scores = self._optimize_query_multi_embedding(
                    query_matrix=query_multi_embedding,
                    candidate_embeddings=candidate_multi_embeddings,
                    complementary_distribution=complementary_distribution,
                )
                final_score_map = {
                    doc_id: float(score) for doc_id, score in zip(embedding_ids, optimized_scores, strict=True)
                }
            else:
                optimized_scores = self._optimize_in_score_space(primary_scores, complementary_distribution)
                final_score_map = {doc_id: float(optimized_scores[index]) for index, doc_id in enumerate(candidate_ids)}
        else:
            embedding_ids, candidate_embeddings = self._get_candidate_embeddings(candidate_ids)
            all_candidates_have_embeddings = len(embedding_ids) == len(candidate_ids)
            if query_embedding is not None and all_candidates_have_embeddings and candidate_embeddings.size > 0:
                optimized_scores = self._optimize_query_embedding(
                    query_embedding=query_embedding,
                    candidate_embeddings=candidate_embeddings,
                    complementary_distribution=complementary_distribution,
                )
                final_score_map = {
                    doc_id: float(score) for doc_id, score in zip(embedding_ids, optimized_scores, strict=True)
                }
            else:
                optimized_scores = self._optimize_in_score_space(primary_scores, complementary_distribution)
                final_score_map = {doc_id: float(optimized_scores[index]) for index, doc_id in enumerate(candidate_ids)}

        return self._sort_results(final_score_map, top_k)

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve with Guided Query Refinement using query ID."""
        fetch_k = top_k * self.fetch_k_multiplier
        primary_results = await self._primary_retrieval_pipeline._retrieve_by_id(query_id, fetch_k)
        complementary_results = await self._complementary_retrieval_pipeline._retrieve_by_id(query_id, fetch_k)

        query_embedding = self._get_query_embedding_by_id(query_id)
        query_multi_embedding = (
            self._get_query_multi_embedding_by_id(query_id) if self._resolve_scorer_mode() == "multi" else None
        )
        return await self._run_gqr(
            top_k=top_k,
            query_embedding=query_embedding,
            query_multi_embedding=query_multi_embedding,
            primary_results=primary_results,
            complementary_results=complementary_results,
        )

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve with Guided Query Refinement using raw query text."""
        fetch_k = top_k * self.fetch_k_multiplier
        primary_results: list[dict[str, Any]]
        complementary_results: list[dict[str, Any]]

        try:
            primary_results = await self._primary_retrieval_pipeline._retrieve_by_text(query_text, fetch_k)
        except EmbeddingError:
            complementary_results = await self._complementary_retrieval_pipeline._retrieve_by_text(query_text, fetch_k)
            primary_results = complementary_results
        else:
            try:
                complementary_results = await self._complementary_retrieval_pipeline._retrieve_by_text(
                    query_text, fetch_k
                )
            except EmbeddingError:
                complementary_results = primary_results

        query_embedding = await self._get_query_embedding_by_text(query_text)
        query_multi_embedding = None
        if self._resolve_scorer_mode() == "multi":
            logger.info(
                "GQR multi-vector by-text retrieval has no standard embedding interface; using score-space fallback"
            )
        return await self._run_gqr(
            top_k=top_k,
            query_embedding=query_embedding,
            query_multi_embedding=query_multi_embedding,
            primary_results=primary_results,
            complementary_results=complementary_results,
        )


__all__ = ["GQRHybridRetrievalPipeline", "GQRHybridRetrievalPipelineConfig"]
