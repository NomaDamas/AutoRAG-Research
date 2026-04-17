"""Power of Noise retrieval pipeline for AutoRAG-Research.

This module implements a configurable retrieval wrapper inspired by
*The Power of Noise: Redefining Retrieval for RAG Systems*.
It combines an existing retrieval pipeline with seeded random noise
samples from the corpus so researchers can study how noisy context
composition affects downstream RAG behavior.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.pipelines.retrieval.hybrid import HybridRetrievalPipeline

NoiseOrder = Literal["retrieved_first", "noise_first", "interleave"]
NoiseMode = Literal["random", "answer_aware_random"]


@dataclass(kw_only=True)
class PowerOfNoiseRetrievalPipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for the Power of Noise retrieval pipeline."""

    base_retrieval_pipeline_name: str
    noise_count: int = 0
    noise_ratio: float | None = None
    noise_order: NoiseOrder = "retrieved_first"
    noise_mode: NoiseMode = "random"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.noise_count < 0:
            msg = "noise_count must be >= 0"
            raise ValueError(msg)
        if self.noise_ratio is not None and not 0 <= self.noise_ratio <= 1:
            msg = "noise_ratio must be between 0 and 1"
            raise ValueError(msg)

    def get_pipeline_class(self) -> type["PowerOfNoiseRetrievalPipeline"]:
        """Return the PowerOfNoiseRetrievalPipeline class."""
        return PowerOfNoiseRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for PowerOfNoiseRetrievalPipeline constructor."""
        return {
            "base_retrieval_pipeline": self.base_retrieval_pipeline_name,
            "noise_count": self.noise_count,
            "noise_ratio": self.noise_ratio,
            "noise_order": self.noise_order,
            "noise_mode": self.noise_mode,
            "seed": self.seed,
        }


class PowerOfNoiseRetrievalPipeline(BaseRetrievalPipeline):
    """Wrap a base retriever and inject seeded corpus noise."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        base_retrieval_pipeline: BaseRetrievalPipeline | str,
        noise_count: int = 0,
        noise_ratio: float | None = None,
        noise_order: NoiseOrder = "retrieved_first",
        noise_mode: NoiseMode = "random",
        seed: int = 0,
        schema: Any | None = None,
        config_dir: Path | None = None,
    ):
        if noise_count < 0:
            msg = "noise_count must be >= 0"
            raise ValueError(msg)
        if noise_ratio is not None and not 0 <= noise_ratio <= 1:
            msg = "noise_ratio must be between 0 and 1"
            raise ValueError(msg)

        if isinstance(base_retrieval_pipeline, str):
            base_retrieval_pipeline = HybridRetrievalPipeline._load_pipeline(
                base_retrieval_pipeline,
                session_factory,
                schema,
                config_dir,
            )

        self._base_retrieval_pipeline = base_retrieval_pipeline
        self.noise_count = noise_count
        self.noise_ratio = noise_ratio
        self.noise_order = noise_order
        self.noise_mode = noise_mode
        self.seed = seed

        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return pipeline configuration for persistence."""
        return {
            "type": "power_of_noise",
            "base_retrieval_pipeline": self._base_retrieval_pipeline.name,
            "noise_count": self.noise_count,
            "noise_ratio": self.noise_ratio,
            "noise_order": self.noise_order,
            "noise_mode": self.noise_mode,
            "seed": self.seed,
        }

    def _resolve_noise_count(self, top_k: int) -> int:
        """Resolve how many noisy documents to inject for a request."""
        if top_k <= 0:
            return 0

        if self.noise_count > 0:
            return min(top_k, self.noise_count)

        if self.noise_ratio is None:
            return 0

        return min(top_k, max(0, round(top_k * self.noise_ratio)))

    def _compose_results(
        self,
        base_results: list[dict[str, Any]],
        noise_results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Compose retrieved and noisy documents into the final context order."""
        if self.noise_order == "retrieved_first":
            combined = [*base_results, *noise_results]
        elif self.noise_order == "noise_first":
            combined = [*noise_results, *base_results]
        else:
            combined: list[dict[str, Any]] = []
            for index in range(max(len(base_results), len(noise_results))):
                if index < len(base_results):
                    combined.append(base_results[index])
                if index < len(noise_results):
                    combined.append(noise_results[index])

        return combined[:top_k]

    def _build_seed_key(self, query_value: int | str) -> str:
        """Build a deterministic random seed key per query."""
        return f"{self.seed}:{query_value}"

    def _load_chunk_results(self, chunk_ids: list[int | str]) -> list[dict[str, Any]]:
        """Load chunk contents and return them in retrieval-result shape."""
        if not chunk_ids:
            return []

        with self._service._create_uow() as uow:
            chunks = uow.chunks.get_by_ids(chunk_ids)
            chunk_by_id = {chunk.id: chunk for chunk in chunks}

        return [
            {
                "doc_id": chunk_id,
                "score": 0.0,
                "content": chunk_by_id[chunk_id].contents,
            }
            for chunk_id in chunk_ids
            if chunk_id in chunk_by_id
        ]

    def _get_noise_candidate_ids(
        self,
        *,
        excluded_doc_ids: set[int | str],
        query_id: int | str | None = None,
    ) -> list[int | str]:
        """Return chunk IDs eligible for noise injection."""
        with self._service._create_uow() as uow:
            excluded_ids = set(excluded_doc_ids)

            if self.noise_mode == "answer_aware_random" and query_id is not None:
                query = uow.queries.get_with_retrieval_relations(query_id)
                if query is not None:
                    excluded_ids.update(
                        relation.chunk_id for relation in query.retrieval_relations if relation.chunk_id is not None
                    )

                    answers = [answer.casefold() for answer in (query.generation_gt or []) if answer]
                    if answers:
                        candidate_ids: list[int | str] = []
                        for chunk in uow.chunks.get_all():
                            if chunk.id in excluded_ids:
                                continue
                            contents = (chunk.contents or "").casefold()
                            if any(answer in contents for answer in answers):
                                continue
                            candidate_ids.append(cast(int | str, chunk.id))
                        sorted_candidate_ids = sorted(candidate_ids, key=str)
                        return cast(list[int | str], sorted_candidate_ids)

            all_chunk_ids = [cast(int | str, chunk_id) for chunk_id in uow.chunks.get_all_ids()]
            sorted_all_chunk_ids = sorted(
                (chunk_id for chunk_id in all_chunk_ids if chunk_id not in excluded_ids),
                key=str,
            )
            return cast(list[int | str], sorted_all_chunk_ids)

    def _sample_noise_results(
        self,
        *,
        sample_size: int,
        query_value: int | str,
        excluded_doc_ids: set[int | str],
        query_id: int | str | None = None,
    ) -> list[dict[str, Any]]:
        """Sample deterministic noise documents from the corpus."""
        if sample_size <= 0:
            return []

        candidate_ids = self._get_noise_candidate_ids(query_id=query_id, excluded_doc_ids=excluded_doc_ids)
        if not candidate_ids:
            return []

        rng = random.Random(self._build_seed_key(query_value))
        selected_ids = rng.sample(candidate_ids, k=min(sample_size, len(candidate_ids)))
        return self._load_chunk_results(selected_ids)

    async def _retrieve_with_noise(
        self,
        *,
        query_value: int | str,
        top_k: int,
        base_retriever: Any,
        query_id: int | str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve with the base pipeline and inject noise when configured."""
        planned_noise_count = self._resolve_noise_count(top_k)
        base_top_k = max(top_k - planned_noise_count, 0)
        base_results = await base_retriever(base_top_k) if base_top_k > 0 else []

        if planned_noise_count == 0:
            return base_results[:top_k]

        excluded_doc_ids = {result["doc_id"] for result in base_results if result.get("doc_id") is not None}
        noise_results = self._sample_noise_results(
            sample_size=planned_noise_count,
            query_value=query_value,
            query_id=query_id,
            excluded_doc_ids=excluded_doc_ids,
        )
        return self._compose_results(base_results, noise_results, top_k)

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve from the base pipeline, then inject seeded noise docs."""
        return await self._retrieve_with_noise(
            query_value=query_id,
            query_id=query_id,
            top_k=top_k,
            base_retriever=lambda base_top_k: self._base_retrieval_pipeline._retrieve_by_id(query_id, base_top_k),
        )

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve from the base pipeline for ad-hoc text and add seeded noise."""
        return await self._retrieve_with_noise(
            query_value=query_text,
            top_k=top_k,
            base_retriever=lambda base_top_k: self._base_retrieval_pipeline._retrieve_by_text(query_text, base_top_k),
        )


__all__ = ["PowerOfNoiseRetrievalPipeline", "PowerOfNoiseRetrievalPipelineConfig"]
