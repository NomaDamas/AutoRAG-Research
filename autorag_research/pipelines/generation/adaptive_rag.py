"""AdaptiveRAG generation pipeline for AutoRAG-Research.

AdaptiveRAG routes each query to a retrieval strategy based on predicted complexity:
- zero: answer directly without retrieval
- single: single-pass retrieval then generation
- multi: iterative retrieval with follow-up query generation
"""

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseGenerationPipelineConfig
from autorag_research.orm.service.generation_pipeline import GenerationResult
from autorag_research.pipelines.generation.base import BaseGenerationPipeline
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.util import TokenUsageTracker

DEFAULT_COMPLEXITY_PROMPT_TEMPLATE = """Classify the following question complexity as exactly one of: simple, moderate, complex.

- simple: can be answered directly without retrieval
- moderate: needs one retrieval pass
- complex: needs iterative multi-step retrieval

Question: {query}

Return only one word: simple, moderate, or complex."""

DEFAULT_ZERO_RETRIEVAL_PROMPT_TEMPLATE = """Answer the question directly.

Question: {query}

Answer:"""

DEFAULT_SINGLE_RETRIEVAL_PROMPT_TEMPLATE = """Answer the question with the provided context.

Context:
{context}

Question: {query}

Answer:"""

DEFAULT_MULTI_RETRIEVAL_QUERY_PROMPT_TEMPLATE = """You are performing iterative retrieval for a complex question.

Question: {query}

Current Context:
{context}

Previous Follow-up Queries:
{follow_up_queries}

Generate the next short retrieval query to gather missing evidence.
If enough evidence is already available, respond exactly with {stop_query_signal}.

Next Retrieval Query:"""

DEFAULT_MULTI_RETRIEVAL_ANSWER_PROMPT_TEMPLATE = """Answer the question using the collected evidence.

Question: {query}

Collected Context:
{context}

Follow-up Queries Used:
{follow_up_queries}

Answer:"""


@dataclass(kw_only=True)
class AdaptiveRAGPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for AdaptiveRAG generation pipeline."""

    complexity_prompt_template: str = field(default=DEFAULT_COMPLEXITY_PROMPT_TEMPLATE)
    zero_retrieval_prompt_template: str = field(default=DEFAULT_ZERO_RETRIEVAL_PROMPT_TEMPLATE)
    single_retrieval_prompt_template: str = field(default=DEFAULT_SINGLE_RETRIEVAL_PROMPT_TEMPLATE)
    multi_retrieval_query_prompt_template: str = field(default=DEFAULT_MULTI_RETRIEVAL_QUERY_PROMPT_TEMPLATE)
    multi_retrieval_answer_prompt_template: str = field(default=DEFAULT_MULTI_RETRIEVAL_ANSWER_PROMPT_TEMPLATE)
    route_for_simple: Literal["zero", "single", "multi"] = "zero"
    route_for_moderate: Literal["zero", "single", "multi"] = "single"
    route_for_complex: Literal["zero", "single", "multi"] = "multi"
    max_multi_steps: int = 2
    stop_query_signal: str = "STOP"

    def get_pipeline_class(self) -> type["AdaptiveRAGPipeline"]:
        """Return the AdaptiveRAGPipeline class."""
        return AdaptiveRAGPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for AdaptiveRAGPipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)

        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "complexity_prompt_template": self.complexity_prompt_template,
            "zero_retrieval_prompt_template": self.zero_retrieval_prompt_template,
            "single_retrieval_prompt_template": self.single_retrieval_prompt_template,
            "multi_retrieval_query_prompt_template": self.multi_retrieval_query_prompt_template,
            "multi_retrieval_answer_prompt_template": self.multi_retrieval_answer_prompt_template,
            "route_for_simple": self.route_for_simple,
            "route_for_moderate": self.route_for_moderate,
            "route_for_complex": self.route_for_complex,
            "max_multi_steps": self.max_multi_steps,
            "stop_query_signal": self.stop_query_signal,
        }


class AdaptiveRAGPipeline(BaseGenerationPipeline):
    """Adaptive routing generation pipeline with zero/single/multi retrieval modes."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        complexity_prompt_template: str = DEFAULT_COMPLEXITY_PROMPT_TEMPLATE,
        zero_retrieval_prompt_template: str = DEFAULT_ZERO_RETRIEVAL_PROMPT_TEMPLATE,
        single_retrieval_prompt_template: str = DEFAULT_SINGLE_RETRIEVAL_PROMPT_TEMPLATE,
        multi_retrieval_query_prompt_template: str = DEFAULT_MULTI_RETRIEVAL_QUERY_PROMPT_TEMPLATE,
        multi_retrieval_answer_prompt_template: str = DEFAULT_MULTI_RETRIEVAL_ANSWER_PROMPT_TEMPLATE,
        route_for_simple: Literal["zero", "single", "multi"] = "zero",
        route_for_moderate: Literal["zero", "single", "multi"] = "single",
        route_for_complex: Literal["zero", "single", "multi"] = "multi",
        max_multi_steps: int = 2,
        stop_query_signal: str = "STOP",
        schema: Any | None = None,
    ):
        """Initialize AdaptiveRAG pipeline."""
        self._complexity_prompt_template = complexity_prompt_template
        self._zero_retrieval_prompt_template = zero_retrieval_prompt_template
        self._single_retrieval_prompt_template = single_retrieval_prompt_template
        self._multi_retrieval_query_prompt_template = multi_retrieval_query_prompt_template
        self._multi_retrieval_answer_prompt_template = multi_retrieval_answer_prompt_template
        self._route_for_simple = route_for_simple
        self._route_for_moderate = route_for_moderate
        self._route_for_complex = route_for_complex
        self._max_multi_steps = max_multi_steps
        self._stop_query_signal = stop_query_signal

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return AdaptiveRAG pipeline configuration."""
        model_name = getattr(self._llm, "model_name", None)
        if model_name is None or not isinstance(model_name, str):
            model_name = type(self._llm).__name__

        return {
            "type": "adaptive_rag",
            "complexity_prompt_template": self._complexity_prompt_template,
            "zero_retrieval_prompt_template": self._zero_retrieval_prompt_template,
            "single_retrieval_prompt_template": self._single_retrieval_prompt_template,
            "multi_retrieval_query_prompt_template": self._multi_retrieval_query_prompt_template,
            "multi_retrieval_answer_prompt_template": self._multi_retrieval_answer_prompt_template,
            "route_for_simple": self._route_for_simple,
            "route_for_moderate": self._route_for_moderate,
            "route_for_complex": self._route_for_complex,
            "max_multi_steps": self._max_multi_steps,
            "stop_query_signal": self._stop_query_signal,
            "retrieval_pipeline_id": self._retrieval_pipeline.pipeline_id,
            "llm_model": model_name,
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text content from an LLM response."""
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def _normalize_score(score: Any) -> float:
        """Normalize retrieval score to float."""
        if isinstance(score, (int, float)):
            return float(score)
        return 0.0

    @staticmethod
    def _parse_complexity_tier(response_text: str) -> Literal["simple", "moderate", "complex"]:
        """Parse complexity output, defaulting to moderate for unknown outputs."""
        normalized = response_text.strip().lower()

        if normalized == "simple":
            return "simple"
        if normalized == "moderate":
            return "moderate"
        if normalized == "complex":
            return "complex"
        if normalized == "medium":
            return "moderate"

        tier_tokens = {
            "moderate" if label == "medium" else label
            for label in re.findall(r"\b(simple|moderate|medium|complex)\b", normalized)
        }
        if len(tier_tokens) == 1:
            tier = tier_tokens.pop()
            if tier == "simple":
                return "simple"
            if tier == "complex":
                return "complex"

        return "moderate"

    @staticmethod
    def _normalize_route(route: str) -> Literal["zero", "single", "multi"]:
        """Normalize route labels to supported route names."""
        normalized = route.strip().lower()
        if normalized in {"zero", "single", "multi"}:
            if normalized == "zero":
                return "zero"
            if normalized == "single":
                return "single"
            if normalized == "multi":
                return "multi"
        return "single"

    def _select_route(
        self, complexity_tier: Literal["simple", "moderate", "complex"]
    ) -> Literal["zero", "single", "multi"]:
        """Select route name from tier-specific mapping."""
        route_map = {
            "simple": self._route_for_simple,
            "moderate": self._route_for_moderate,
            "complex": self._route_for_complex,
        }
        return self._normalize_route(route_map[complexity_tier])

    @staticmethod
    def _merge_retrieval_results(result_sets: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Merge retrieval results by doc_id, keeping the highest score per document."""
        merged: dict[int | str, dict[str, Any]] = {}

        for result_set in result_sets:
            for result in result_set:
                doc_id = result.get("doc_id")
                if doc_id is None:
                    continue

                score = AdaptiveRAGPipeline._normalize_score(result.get("score"))
                existing = merged.get(doc_id)
                if existing is None or score > AdaptiveRAGPipeline._normalize_score(existing.get("score")):
                    merged[doc_id] = {
                        "doc_id": doc_id,
                        "score": score,
                    }

        return sorted(
            merged.values(),
            key=lambda item: (-AdaptiveRAGPipeline._normalize_score(item.get("score")), str(item.get("doc_id"))),
        )

    def _build_context(self, chunk_ids: list[int | str]) -> str:
        """Build context text from chunk IDs."""
        if not chunk_ids:
            return ""
        chunk_contents = self._service.get_chunk_contents(chunk_ids)
        return "\n\n".join(content for content in chunk_contents if content)

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate answer using adaptive route selection."""
        query_text = self._service.get_query_text(query_id)
        tracker = TokenUsageTracker()

        complexity_prompt = self._complexity_prompt_template.format(query=query_text)
        complexity_response = await self._llm.ainvoke(complexity_prompt)
        tracker.record(complexity_response)
        complexity_tier = self._parse_complexity_tier(self._extract_text(complexity_response))
        route = self._select_route(complexity_tier)

        if route == "zero":
            prompt = self._zero_retrieval_prompt_template.format(query=query_text)
            answer_response = await self._llm.ainvoke(prompt)
            tracker.record(answer_response)
            answer_text = self._extract_text(answer_response)

            return GenerationResult(
                text=answer_text,
                token_usage=tracker.total,
                metadata={
                    "complexity_tier": complexity_tier,
                    "route": route,
                    "retrieved_chunk_ids": [],
                    "retrieved_scores": [],
                    "follow_up_queries": [],
                },
            )

        if route == "single":
            retrieved_results = await self._retrieval_pipeline._retrieve_by_id(query_id, top_k)
            chunk_ids = [item["doc_id"] for item in retrieved_results if item.get("doc_id") is not None]
            context = self._build_context(chunk_ids)
            prompt = self._single_retrieval_prompt_template.format(context=context, query=query_text)
            answer_response = await self._llm.ainvoke(prompt)
            tracker.record(answer_response)
            answer_text = self._extract_text(answer_response)

            return GenerationResult(
                text=answer_text,
                token_usage=tracker.total,
                metadata={
                    "complexity_tier": complexity_tier,
                    "route": route,
                    "retrieved_chunk_ids": chunk_ids,
                    "retrieved_scores": [self._normalize_score(item.get("score")) for item in retrieved_results],
                    "follow_up_queries": [],
                },
            )

        # multi route
        retrieved_sets: list[list[dict[str, Any]]] = [
            await self._retrieval_pipeline._retrieve_by_id(query_id, top_k),
        ]
        follow_up_queries: list[str] = []

        for _ in range(self._max_multi_steps):
            merged_so_far = self._merge_retrieval_results(retrieved_sets)[:top_k]
            context = self._build_context([item["doc_id"] for item in merged_so_far])
            previous_queries = "\n".join(follow_up_queries) if follow_up_queries else "(none)"
            next_query_prompt = self._multi_retrieval_query_prompt_template.format(
                query=query_text,
                context=context,
                follow_up_queries=previous_queries,
                stop_query_signal=self._stop_query_signal,
            )

            next_query_response = await self._llm.ainvoke(next_query_prompt)
            tracker.record(next_query_response)
            next_query = self._extract_text(next_query_response).strip()

            if not next_query or next_query.upper() == self._stop_query_signal.upper():
                break

            follow_up_queries.append(next_query)
            retrieved_sets.append(await self._retrieval_pipeline.retrieve(next_query, top_k))

        merged_results = self._merge_retrieval_results(retrieved_sets)[:top_k]
        merged_chunk_ids = [item["doc_id"] for item in merged_results]
        final_context = self._build_context(merged_chunk_ids)
        final_follow_ups = "\n".join(follow_up_queries) if follow_up_queries else "(none)"
        answer_prompt = self._multi_retrieval_answer_prompt_template.format(
            query=query_text,
            context=final_context,
            follow_up_queries=final_follow_ups,
        )
        answer_response = await self._llm.ainvoke(answer_prompt)
        tracker.record(answer_response)
        answer_text = self._extract_text(answer_response)

        return GenerationResult(
            text=answer_text,
            token_usage=tracker.total,
            metadata={
                "complexity_tier": complexity_tier,
                "route": route,
                "retrieved_chunk_ids": merged_chunk_ids,
                "retrieved_scores": [self._normalize_score(item.get("score")) for item in merged_results],
                "follow_up_queries": follow_up_queries,
            },
        )
