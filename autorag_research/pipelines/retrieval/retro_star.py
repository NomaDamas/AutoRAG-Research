"""RETRO* retrieval pipeline for AutoRAG-Research.

This pipeline implements a phase-1 inference-time RETRO*-style reranking flow
inspired by "Retro*: Optimizing LLMs for Reasoning-Intensive Document Retrieval"
(ICLR 2026).

Scope note:
- This implementation wraps an existing retrieval pipeline and reranks its
  candidates with rubric-guided LLM scoring.
- The paper's training recipe (SFT + RL) is intentionally out of scope here.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.injection import health_check_llm
from autorag_research.orm.uow.retrieval_uow import RetrievalUnitOfWork
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.util import truncate_texts

DEFAULT_RETRO_STAR_RELEVANCE_DEFINITION = (
    "A document is relevant when it helps answer the query, including evidence that is indirect "
    "but still necessary for the required reasoning."
)

DEFAULT_RETRO_STAR_PROMPT_TEMPLATE = """You are performing reasoning-intensive document retrieval.

Relevance definition:
{relevance_definition}

Given the query ({query_type}) and the candidate document ({document_type}), do three things:
1. Analyze what information the query is really asking for.
2. Analyze how well the document supports that need, including indirect reasoning value.
3. Assign one integer relevance score from 0 to 100.

Use this score guide:
- 80-100: highly relevant and directly useful
- 60-79: relevant, with most important information present
- 40-59: moderately relevant, partially useful
- 20-39: slightly relevant, limited value
- 0-19: irrelevant

End your response with the final score only inside <score> tags, for example <score>87</score>.

Query ({query_type}):
[Begin Query]
{query}
[End Query]

Document ({document_type}):
[Begin Document]
{doc}
[End Document]"""

_SCORE_PATTERN = re.compile(r"<score>\s*(-?\d{1,3})\s*</score>", re.IGNORECASE | re.DOTALL)


def _parse_retro_score(response_text: str) -> int:
    """Parse the final RETRO* relevance score from LLM output."""
    match = _SCORE_PATTERN.search(response_text)
    if match is None:
        msg = "RETRO* response must contain a final integer score inside <score> tags"
        raise ValueError(msg)

    return max(0, min(100, int(match.group(1))))


def _integrate_retro_scores(scores: list[int | float], weights: list[float] | None = None) -> float:
    """Integrate multiple sampled RETRO* scores into one final score."""
    if not scores:
        msg = "scores must not be empty"
        raise ValueError(msg)

    if weights is None:
        return float(sum(scores) / len(scores))

    _validate_sample_weights(weights, len(scores))

    total_weight = sum(weights)
    if total_weight <= 0:
        msg = "weights must sum to a positive value"
        raise ValueError(msg)

    weighted_sum = sum(score * weight for score, weight in zip(scores, weights, strict=True))
    return float(weighted_sum / total_weight)


def _validate_sample_weights(
    weights: list[float],
    expected_length: int,
    *,
    length_error_message: str = "weights must have the same length as scores",
    negative_error_message: str = "weights must not contain negative values",
) -> None:
    """Validate optional RETRO* sample weights before integration."""
    if len(weights) != expected_length:
        msg = length_error_message
        raise ValueError(msg)
    if any(weight < 0 for weight in weights):
        msg = negative_error_message
        raise ValueError(msg)


@dataclass(kw_only=True)
class RetroStarPipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for the RETRO* retrieval pipeline."""

    llm: str | BaseLanguageModel
    retrieval_pipeline_name: str
    candidate_top_k: int = 100
    prompt_template: str = field(default=DEFAULT_RETRO_STAR_PROMPT_TEMPLATE)
    relevance_definition: str = field(default=DEFAULT_RETRO_STAR_RELEVANCE_DEFINITION)
    query_type: str = "query"
    document_type: str = "document"
    num_samples: int = 1
    sample_weights: list[float] | None = None
    max_document_tokens: int = 768
    max_query_tokens: int = 256
    max_rerank_concurrency: int = 4
    _retrieval_pipeline: BaseRetrievalPipeline | None = field(default=None, repr=False)

    def __setattr__(self, name: str, value: Any) -> None:
        """Auto-convert string LLM config names to model instances."""
        if name == "llm" and isinstance(value, str):
            from autorag_research.injection import load_llm

            value = load_llm(value)
            health_check_llm(value)
        super().__setattr__(name, value)

    def inject_retrieval_pipeline(self, pipeline: BaseRetrievalPipeline) -> None:
        """Inject the wrapped retrieval pipeline instance."""
        self._retrieval_pipeline = pipeline

    def get_pipeline_class(self) -> type[RetroStarRetrievalPipeline]:
        """Return the RetroStarRetrievalPipeline class."""
        return RetroStarRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for RetroStarRetrievalPipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)

        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "candidate_top_k": self.candidate_top_k,
            "prompt_template": self.prompt_template,
            "relevance_definition": self.relevance_definition,
            "query_type": self.query_type,
            "document_type": self.document_type,
            "num_samples": self.num_samples,
            "sample_weights": self.sample_weights,
            "max_document_tokens": self.max_document_tokens,
            "max_query_tokens": self.max_query_tokens,
            "max_rerank_concurrency": self.max_rerank_concurrency,
        }


class RetroStarRetrievalPipeline(BaseRetrievalPipeline):
    """RETRO*-style rubric-based reranking over wrapped retrieval candidates."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        candidate_top_k: int = 100,
        prompt_template: str = DEFAULT_RETRO_STAR_PROMPT_TEMPLATE,
        relevance_definition: str = DEFAULT_RETRO_STAR_RELEVANCE_DEFINITION,
        query_type: str = "query",
        document_type: str = "document",
        num_samples: int = 1,
        sample_weights: list[float] | None = None,
        max_document_tokens: int = 768,
        max_query_tokens: int = 256,
        max_rerank_concurrency: int = 4,
        schema: Any | None = None,
    ):
        """Initialize a RETRO* retrieval pipeline wrapper."""
        if "{query}" not in prompt_template or "{doc}" not in prompt_template:
            msg = "prompt_template must contain both '{query}' and '{doc}' placeholders"
            raise ValueError(msg)
        if candidate_top_k < 1:
            msg = "candidate_top_k must be >= 1"
            raise ValueError(msg)
        if num_samples < 1:
            msg = "num_samples must be >= 1"
            raise ValueError(msg)
        if max_document_tokens < 1 or max_query_tokens < 1:
            msg = "max_document_tokens and max_query_tokens must be >= 1"
            raise ValueError(msg)
        if max_rerank_concurrency < 1:
            msg = "max_rerank_concurrency must be >= 1"
            raise ValueError(msg)
        if sample_weights is not None:
            _validate_sample_weights(
                sample_weights,
                num_samples,
                length_error_message="sample_weights must match num_samples when provided",
                negative_error_message="sample_weights must not contain negative values",
            )

        self.llm = llm
        self._retrieval_pipeline = retrieval_pipeline
        self.candidate_top_k = candidate_top_k
        self.prompt_template = prompt_template
        self.relevance_definition = relevance_definition
        self.query_type = query_type
        self.document_type = document_type
        self.num_samples = num_samples
        self.sample_weights = sample_weights
        self.max_document_tokens = max_document_tokens
        self.max_query_tokens = max_query_tokens
        self.max_rerank_concurrency = max_rerank_concurrency

        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return RETRO* pipeline configuration for storage."""
        model_name = getattr(self.llm, "model_name", None)
        if model_name is None or not isinstance(model_name, str):
            model_name = type(self.llm).__name__

        return {
            "type": "retro_star",
            "candidate_top_k": self.candidate_top_k,
            "prompt_template": self.prompt_template,
            "relevance_definition": self.relevance_definition,
            "query_type": self.query_type,
            "document_type": self.document_type,
            "num_samples": self.num_samples,
            "sample_weights": self.sample_weights,
            "max_document_tokens": self.max_document_tokens,
            "max_query_tokens": self.max_query_tokens,
            "max_rerank_concurrency": self.max_rerank_concurrency,
            "retrieval_pipeline_id": getattr(self._retrieval_pipeline, "pipeline_id", None),
            "wrapped_pipeline_type": type(self._retrieval_pipeline).__name__,
            "llm_model": model_name,
        }

    @staticmethod
    def _extract_response_content(response: Any) -> str:
        """Extract text content from an LLM response."""
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)

    def _build_prompt(self, query_text: str, document_text: str) -> str:
        """Build the RETRO* scoring prompt for one query-document pair."""
        truncated_query = truncate_texts([query_text], self.max_query_tokens)[0]
        truncated_document = truncate_texts([document_text], self.max_document_tokens)[0]
        return self.prompt_template.format(
            relevance_definition=self.relevance_definition,
            query_type=self.query_type,
            document_type=self.document_type,
            query=truncated_query,
            doc=truncated_document,
        )

    async def _score_document(self, query_text: str, document_text: str) -> float:
        """Score a single candidate document with RETRO* prompting."""
        prompt = self._build_prompt(query_text, document_text)
        sampled_scores: list[float] = []
        for _ in range(self.num_samples):
            response = await self.llm.ainvoke(prompt)
            sampled_scores.append(float(_parse_retro_score(self._extract_response_content(response))))

        return _integrate_retro_scores(sampled_scores, self.sample_weights)

    def _ensure_candidate_contents(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Backfill missing candidate contents from the chunk table."""
        missing_ids = [candidate["doc_id"] for candidate in candidates if not candidate.get("content")]
        if not missing_ids:
            return candidates

        with RetrievalUnitOfWork(self.session_factory, self._schema) as uow:
            chunks = uow.chunks.get_by_ids(missing_ids)

        contents_by_id = {chunk.id: chunk.contents for chunk in chunks}
        enriched_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            enriched_candidate = dict(candidate)
            if not enriched_candidate.get("content"):
                enriched_candidate["content"] = contents_by_id.get(candidate["doc_id"], "")
            enriched_candidates.append(enriched_candidate)

        return enriched_candidates

    async def _score_candidate(self, query_text: str, candidate: dict[str, Any]) -> dict[str, Any]:
        """Compute a RETRO* score for one wrapped retrieval candidate."""
        retro_score = await self._score_document(query_text, candidate.get("content", ""))
        return {
            "doc_id": candidate["doc_id"],
            "score": retro_score,
            "content": candidate.get("content", ""),
            "_wrapped_score": float(candidate.get("score", 0.0)),
        }

    async def _rerank_candidates(
        self,
        query_text: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rerank candidates with RETRO* scoring."""
        semaphore = asyncio.Semaphore(self.max_rerank_concurrency)

        async def score_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await self._score_candidate(query_text, candidate)

        scored_candidates = await asyncio.gather(*(score_candidate(candidate) for candidate in candidates))
        scored_candidates.sort(
            key=lambda candidate: (candidate["score"], candidate["_wrapped_score"]),
            reverse=True,
        )

        return [
            {
                "doc_id": candidate["doc_id"],
                "score": candidate["score"],
                "content": candidate["content"],
            }
            for candidate in scored_candidates[:top_k]
        ]

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        """Fetch query text by ID, then rerank candidates with RETRO* scoring."""
        query_texts = self._service.fetch_query_texts([query_id])
        if not query_texts:
            return []

        return await self._retrieve_by_text(query_texts[0], top_k)

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve candidates from the wrapped pipeline, then rerank them."""
        candidate_top_k = max(top_k, self.candidate_top_k)
        candidates = await self._retrieval_pipeline.retrieve(query_text, candidate_top_k)
        if not candidates:
            return []

        enriched_candidates = self._ensure_candidate_contents(candidates)
        return await self._rerank_candidates(query_text, enriched_candidates, top_k)


__all__ = [
    "DEFAULT_RETRO_STAR_PROMPT_TEMPLATE",
    "DEFAULT_RETRO_STAR_RELEVANCE_DEFINITION",
    "RetroStarPipelineConfig",
    "RetroStarRetrievalPipeline",
    "_integrate_retro_scores",
    "_parse_retro_score",
]
