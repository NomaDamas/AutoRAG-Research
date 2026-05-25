"""Hybrid Deep Searcher generation pipeline for AutoRAG-Research.

This pipeline implements a practical inference-only Hybrid Deep Searcher (HDS)
baseline: an LLM alternates sequential reasoning with parallel fan-out query
proposal, retrieves evidence for those subqueries through an existing retrieval
pipeline, merges/deduplicates evidence, and either refines the search or emits a
final answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseGenerationPipelineConfig
from autorag_research.orm.service.generation_pipeline import GenerationResult
from autorag_research.pipelines.generation.base import BaseGenerationPipeline
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.util import TokenUsageTracker, run_with_concurrency_limit

logger = logging.getLogger("AutoRAG-Research")

DEFAULT_HDS_PLAN_PROMPT = """You are a Hybrid Deep Searcher agent.
Combine parallel exploration with sequential reasoning. At each turn, either:
- propose multiple parallel search queries inside <queries>...</queries>, one query per line, or
- finish with <answer>final answer</answer>.

Question:
{query}

Turn: {turn}/{max_turns}
Evidence gathered so far:
{evidence}

Reasoning trace:
{trace}

Next action:"""

DEFAULT_HDS_FINAL_PROMPT = """Answer the question using the Hybrid Deep Searcher evidence and reasoning trace.

Question:
{query}

Evidence:
{evidence}

Reasoning trace:
{trace}

Final answer:"""

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_QUERIES_RE = re.compile(r"<queries>\s*(.*?)\s*</queries>", re.IGNORECASE | re.DOTALL)
_QUERY_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|query\s*\d*\s*:|[A-Za-z][.)])\s*", re.IGNORECASE)


@dataclass(frozen=True)
class HybridDeepSearchAction:
    """Parsed Hybrid Deep Searcher action."""

    kind: str
    text: str = ""
    queries: tuple[str, ...] = ()


@dataclass(frozen=True)
class HybridDeepSearchRetrievalFailure:
    """Failure details for one fan-out retrieval query."""

    query: str
    error: str


@dataclass(frozen=True)
class HybridDeepSearchRetrievalResult:
    """Result envelope for one fan-out retrieval query."""

    query: str
    results: list[dict[str, Any]] = field(default_factory=list)
    failure: HybridDeepSearchRetrievalFailure | None = None


def parse_hybrid_deep_search_action(response_text: str, max_queries: int) -> HybridDeepSearchAction:
    """Parse an HDS answer or parallel-query action."""
    answer_match = _ANSWER_RE.search(response_text)
    if answer_match is not None:
        return HybridDeepSearchAction(kind="answer", text=answer_match.group(1).strip())

    queries_match = _QUERIES_RE.search(response_text)
    query_block = queries_match.group(1) if queries_match is not None else response_text
    queries: list[str] = []
    for line in query_block.splitlines():
        cleaned = _QUERY_PREFIX_RE.sub("", line).strip()
        if cleaned and cleaned not in queries:
            queries.append(cleaned)
        if len(queries) >= max_queries:
            break

    if queries:
        return HybridDeepSearchAction(kind="queries", queries=tuple(queries))

    return HybridDeepSearchAction(kind="answer", text=response_text.strip())


@dataclass(kw_only=True)
class HybridDeepSearcherPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for the Hybrid Deep Searcher generation pipeline."""

    plan_prompt_template: str = field(default=DEFAULT_HDS_PLAN_PROMPT)
    final_prompt_template: str = field(default=DEFAULT_HDS_FINAL_PROMPT)
    max_turns: int = 3
    max_parallel_queries: int = 4
    k_per_query: int = 3
    evidence_budget: int = 12
    retrieval_concurrency: int = 4
    fallback_to_final_prompt: bool = True
    allow_partial_retrieval_failures: bool = False

    def get_pipeline_class(self) -> type[HybridDeepSearcherPipeline]:
        """Return the HybridDeepSearcherPipeline class."""
        return HybridDeepSearcherPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for HybridDeepSearcherPipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)
        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "plan_prompt_template": self.plan_prompt_template,
            "final_prompt_template": self.final_prompt_template,
            "max_turns": self.max_turns,
            "max_parallel_queries": self.max_parallel_queries,
            "k_per_query": self.k_per_query,
            "evidence_budget": self.evidence_budget,
            "retrieval_concurrency": self.retrieval_concurrency,
            "fallback_to_final_prompt": self.fallback_to_final_prompt,
            "allow_partial_retrieval_failures": self.allow_partial_retrieval_failures,
        }


class HybridDeepSearcherPipeline(BaseGenerationPipeline):
    """Generation pipeline that mixes parallel fan-out search with sequential reasoning."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        plan_prompt_template: str = DEFAULT_HDS_PLAN_PROMPT,
        final_prompt_template: str = DEFAULT_HDS_FINAL_PROMPT,
        max_turns: int = 3,
        max_parallel_queries: int = 4,
        k_per_query: int = 3,
        evidence_budget: int = 12,
        retrieval_concurrency: int = 4,
        fallback_to_final_prompt: bool = True,
        allow_partial_retrieval_failures: bool = False,
        schema: Any | None = None,
    ):
        """Initialize Hybrid Deep Searcher."""
        if max_turns < 1:
            msg = "max_turns must be >= 1"
            raise ValueError(msg)
        if max_parallel_queries < 1:
            msg = "max_parallel_queries must be >= 1"
            raise ValueError(msg)
        if k_per_query < 1:
            msg = "k_per_query must be >= 1"
            raise ValueError(msg)
        if evidence_budget < 1 or retrieval_concurrency < 1:
            msg = "evidence_budget and retrieval_concurrency must be >= 1"
            raise ValueError(msg)

        self.plan_prompt_template = plan_prompt_template
        self.final_prompt_template = final_prompt_template
        self.max_turns = max_turns
        self.max_parallel_queries = max_parallel_queries
        self.k_per_query = k_per_query
        self.evidence_budget = evidence_budget
        self.retrieval_concurrency = retrieval_concurrency
        self.fallback_to_final_prompt = fallback_to_final_prompt
        self.allow_partial_retrieval_failures = allow_partial_retrieval_failures

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return HDS pipeline configuration."""
        model_name = getattr(self._llm, "model_name", None)
        if model_name is None or not isinstance(model_name, str):
            model_name = type(self._llm).__name__
        return {
            "type": "hybrid_deep_searcher",
            "plan_prompt_template": self.plan_prompt_template,
            "final_prompt_template": self.final_prompt_template,
            "max_turns": self.max_turns,
            "max_parallel_queries": self.max_parallel_queries,
            "k_per_query": self.k_per_query,
            "evidence_budget": self.evidence_budget,
            "retrieval_concurrency": self.retrieval_concurrency,
            "fallback_to_final_prompt": self.fallback_to_final_prompt,
            "allow_partial_retrieval_failures": self.allow_partial_retrieval_failures,
            "retrieval_pipeline_id": self._retrieval_pipeline.pipeline_id,
            "llm_model": model_name,
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract response text."""
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def _format_evidence(evidence: list[str]) -> str:
        """Format evidence list for prompts."""
        if not evidence:
            return "(none)"
        return "\n\n".join(f"[{index + 1}] {item}" for index, item in enumerate(evidence))

    @staticmethod
    def _format_trace(trace: list[str]) -> str:
        """Format search trace for prompts."""
        return "\n".join(trace) if trace else "(none)"

    def _build_plan_prompt(self, query: str, evidence: list[str], trace: list[str], turn: int) -> str:
        """Build one sequential HDS planning prompt."""
        return self.plan_prompt_template.format(
            query=query,
            evidence=self._format_evidence(evidence),
            trace=self._format_trace(trace),
            turn=turn,
            max_turns=self.max_turns,
        )

    def _build_final_prompt(self, query: str, evidence: list[str], trace: list[str]) -> str:
        """Build fallback final-answer prompt."""
        return self.final_prompt_template.format(
            query=query,
            evidence=self._format_evidence(evidence),
            trace=self._format_trace(trace),
        )

    def _contents_from_results(self, results: list[dict[str, Any]]) -> tuple[list[int | str], list[str]]:
        """Extract IDs and contents, backfilling missing chunk text."""
        doc_ids: list[int | str] = []
        contents: list[str | None] = []
        missing_positions: list[int] = []
        missing_ids: list[int | str] = []
        for result in results:
            doc_id = result.get("doc_id")
            if doc_id is None:
                continue
            doc_ids.append(doc_id)
            content = result.get("content")
            if content:
                contents.append(str(content))
            else:
                missing_positions.append(len(contents))
                missing_ids.append(doc_id)
                contents.append(None)
        if missing_ids:
            fetched_contents = self._service.get_chunk_contents(missing_ids)
            for position, fetched_content in zip(missing_positions, fetched_contents, strict=False):
                contents[position] = fetched_content
        return doc_ids, [content or "" for content in contents]

    async def _retrieve_query(self, query_and_k: tuple[str, int]) -> HybridDeepSearchRetrievalResult:
        """Retrieve evidence for one fan-out query."""
        query, top_k = query_and_k
        try:
            return HybridDeepSearchRetrievalResult(
                query=query,
                results=await self._retrieval_pipeline.retrieve(query, top_k),
            )
        except Exception as exc:
            logger.exception("Hybrid Deep Searcher retrieval failed for generated query: %s", query)
            return HybridDeepSearchRetrievalResult(
                query=query,
                failure=HybridDeepSearchRetrievalFailure(query=query, error=str(exc)),
            )

    async def _retrieve_parallel(
        self,
        queries: tuple[str, ...],
        top_k: int,
    ) -> tuple[list[list[dict[str, Any]]], list[HybridDeepSearchRetrievalFailure]]:
        """Retrieve multiple fan-out queries with bounded concurrency."""
        outcomes = await run_with_concurrency_limit(
            [(query, top_k) for query in queries],
            self._retrieve_query,
            max_concurrency=min(self.retrieval_concurrency, len(queries)),
            error_message="Hybrid Deep Searcher retrieval failed",
        )
        result_sets: list[list[dict[str, Any]]] = []
        failures: list[HybridDeepSearchRetrievalFailure] = []
        for query, outcome in zip(queries, outcomes, strict=False):
            if outcome is None:
                failures.append(HybridDeepSearchRetrievalFailure(query=query, error="unknown retrieval error"))
            elif outcome.failure is not None:
                failures.append(outcome.failure)
            else:
                result_sets.append(outcome.results)

        if failures and not self.allow_partial_retrieval_failures:
            failure_summary = "; ".join(f"{failure.query}: {failure.error}" for failure in failures)
            msg = f"Hybrid Deep Searcher retrieval failed for {len(failures)} fan-out query(s): {failure_summary}"
            raise RuntimeError(msg)

        return result_sets, failures

    @staticmethod
    def _merge_results(result_sets: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Merge retrieval results by doc_id, keeping the highest score."""
        merged: dict[int | str, dict[str, Any]] = {}
        for result_set in result_sets:
            for result in result_set:
                doc_id = result.get("doc_id")
                if doc_id is None:
                    continue
                score = result.get("score", 0.0)
                score_value = float(score) if isinstance(score, (int, float)) else 0.0
                existing = merged.get(doc_id)
                if existing is None or score_value > float(existing.get("score", 0.0)):
                    merged[doc_id] = {**result, "score": score_value}
        return sorted(merged.values(), key=lambda item: (-float(item.get("score", 0.0)), str(item.get("doc_id"))))

    def _append_evidence(self, evidence: list[str], new_contents: list[str]) -> list[str]:
        """Append deduplicated evidence within evidence budget."""
        updated = list(evidence)
        for content in new_contents:
            if content and content not in updated:
                updated.append(content)
        return updated[-self.evidence_budget :]

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate an answer with parallel fan-out search and sequential refinement."""
        query_text = self._service.get_query_text(query_id)
        tracker = TokenUsageTracker()
        evidence: list[str] = []
        trace: list[str] = []
        retrieved_doc_ids: list[int | str] = []
        retrieval_failures: list[HybridDeepSearchRetrievalFailure] = []
        final_answer = ""
        terminated_by = "max_turns"
        for turn in range(1, self.max_turns + 1):
            plan_prompt = self._build_plan_prompt(query_text, evidence, trace, turn)
            response = await self._llm.ainvoke(plan_prompt)
            tracker.record(response)
            action = parse_hybrid_deep_search_action(self._extract_text(response), self.max_parallel_queries)
            if action.kind == "answer":
                final_answer = action.text
                trace.append(f"answer: {final_answer}")
                terminated_by = "answer"
                break

            trace.append(f"turn {turn} queries: {' | '.join(action.queries)}")
            effective_k = max(1, min(top_k, self.k_per_query)) if top_k > 0 else self.k_per_query
            result_sets, failures = await self._retrieve_parallel(action.queries, effective_k)
            retrieval_failures.extend(failures)
            merged_results = self._merge_results(result_sets)
            doc_ids, contents = self._contents_from_results(merged_results)
            for doc_id in doc_ids:
                if doc_id not in retrieved_doc_ids:
                    retrieved_doc_ids.append(doc_id)
            evidence = self._append_evidence(evidence, contents)

        if not final_answer and self.fallback_to_final_prompt:
            final_prompt = self._build_final_prompt(query_text, evidence, trace)
            final_response = await self._llm.ainvoke(final_prompt)
            tracker.record(final_response)
            final_answer = self._extract_text(final_response)
            terminated_by = f"{terminated_by}_fallback"

        return GenerationResult(
            text=final_answer,
            token_usage=tracker.total,
            metadata={
                "trace": trace,
                "evidence": evidence,
                "retrieved_chunk_ids": retrieved_doc_ids,
                "retrieval_failures": [
                    {"query": failure.query, "error": failure.error} for failure in retrieval_failures
                ],
                "terminated_by": terminated_by,
            },
        )


__all__ = [
    "DEFAULT_HDS_FINAL_PROMPT",
    "DEFAULT_HDS_PLAN_PROMPT",
    "HybridDeepSearchAction",
    "HybridDeepSearchRetrievalFailure",
    "HybridDeepSearchRetrievalResult",
    "HybridDeepSearcherPipeline",
    "HybridDeepSearcherPipelineConfig",
    "parse_hybrid_deep_search_action",
]
