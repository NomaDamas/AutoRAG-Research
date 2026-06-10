"""Hybrid Deep Searcher generation pipeline for AutoRAG-Research.

This pipeline implements the paper-style inference protocol for Hybrid Deep
Searcher (HDS): the model emits reasoning plus parallel search-query blocks, the
environment appends query-labelled search-result blocks to a rolling interaction
log, and the loop stops with a final answer or when turn/search-call budgets are
exhausted.
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
- think, then propose parallel search queries inside <|begin search queries|>...<|end search queries|>, separated by semicolons or newlines, or
- finish with a final answer in \\boxed{{...}}.

Question:
{query}

Turn: {turn}/{max_turns}
Remaining search calls: {remaining_search_calls}
Interaction log:
{interaction_log}

Next action:"""

DEFAULT_HDS_FINAL_PROMPT = """Answer the question using the Hybrid Deep Searcher rolling interaction log.

Question:
{query}

Interaction log:
{interaction_log}

Final answer:"""

_BOXED_ANSWER_RE = re.compile(r"\\boxed\s*\{(.*?)\}", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_QUERIES_RE = re.compile(r"<\|begin search queries\|>\s*(.*?)\s*<\|end search queries\|>", re.IGNORECASE | re.DOTALL)
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


def _sanitize_retrieval_error(exc: Exception) -> str:
    """Return a persistence-safe retrieval error summary.

    Full backend exceptions remain available in logs via ``logger.exception``.
    Result metadata is durable and user-visible, so it stores only the exception
    type plus a controlled message instead of raw DSNs, SQL, payloads, or keys.
    """
    return f"{type(exc).__name__}: retrieval failed"


def parse_hybrid_deep_search_action(response_text: str, max_queries: int) -> HybridDeepSearchAction:
    """Parse an HDS boxed answer, tolerant answer tag, or paper search-query action."""
    boxed_match = _BOXED_ANSWER_RE.search(response_text)
    if boxed_match is not None:
        return HybridDeepSearchAction(kind="answer", text=boxed_match.group(1).strip())

    answer_match = _ANSWER_RE.search(response_text)
    if answer_match is not None:
        return HybridDeepSearchAction(kind="answer", text=answer_match.group(1).strip())

    queries_match = _QUERIES_RE.search(response_text)
    query_block = queries_match.group(1) if queries_match is not None else response_text
    queries: list[str] = []
    seen: set[str] = set()
    for raw_query in re.split(r"[;\n]+", query_block):
        cleaned = _QUERY_PREFIX_RE.sub("", raw_query).strip()
        if cleaned and cleaned not in seen:
            queries.append(cleaned)
            seen.add(cleaned)
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
    max_search_calls: int = 8
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
            "max_search_calls": self.max_search_calls,
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
        max_search_calls: int = 8,
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
        if evidence_budget < 1 or max_search_calls < 1 or retrieval_concurrency < 1:
            msg = "evidence_budget, max_search_calls, and retrieval_concurrency must be >= 1"
            raise ValueError(msg)

        self.plan_prompt_template = plan_prompt_template
        self.final_prompt_template = final_prompt_template
        self.max_turns = max_turns
        self.max_parallel_queries = max_parallel_queries
        self.k_per_query = k_per_query
        self.evidence_budget = evidence_budget
        self.max_search_calls = max_search_calls
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
            "max_search_calls": self.max_search_calls,
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
    def _format_interaction_log(interaction_log: list[str]) -> str:
        """Format the rolling HDS interaction log for prompts."""
        return "\n".join(interaction_log) if interaction_log else "(none)"

    def _build_plan_prompt(self, query: str, interaction_log: list[str], turn: int, remaining_search_calls: int) -> str:
        """Build one sequential HDS planning prompt."""
        return self.plan_prompt_template.format(
            query=query,
            interaction_log=self._format_interaction_log(interaction_log),
            turn=turn,
            max_turns=self.max_turns,
            remaining_search_calls=remaining_search_calls,
        )

    def _build_final_prompt(self, query: str, interaction_log: list[str]) -> str:
        """Build fallback final-answer prompt."""
        return self.final_prompt_template.format(
            query=query,
            interaction_log=self._format_interaction_log(interaction_log),
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
                failure=HybridDeepSearchRetrievalFailure(query=query, error=_sanitize_retrieval_error(exc)),
            )

    async def _retrieve_parallel(
        self,
        queries: tuple[str, ...],
        top_k: int,
    ) -> tuple[list[HybridDeepSearchRetrievalResult], list[HybridDeepSearchRetrievalFailure]]:
        """Retrieve multiple fan-out queries with bounded concurrency while preserving query pairing."""
        outcomes = await run_with_concurrency_limit(
            [(query, top_k) for query in queries],
            self._retrieve_query,
            max_concurrency=min(self.retrieval_concurrency, len(queries)),
            error_message="Hybrid Deep Searcher retrieval failed",
        )
        paired_results: list[HybridDeepSearchRetrievalResult] = []
        failures: list[HybridDeepSearchRetrievalFailure] = []
        for query, outcome in zip(queries, outcomes, strict=False):
            if outcome is None:
                failure = HybridDeepSearchRetrievalFailure(query=query, error="unknown retrieval error")
                failures.append(failure)
                paired_results.append(HybridDeepSearchRetrievalResult(query=query, failure=failure))
            elif outcome.failure is not None:
                failures.append(outcome.failure)
                paired_results.append(outcome)
            else:
                paired_results.append(outcome)

        if failures and not self.allow_partial_retrieval_failures:
            failure_summary = "; ".join(f"{failure.query}: {failure.error}" for failure in failures)
            msg = f"Hybrid Deep Searcher retrieval failed for {len(failures)} fan-out query(s): {failure_summary}"
            raise RuntimeError(msg)

        return paired_results, failures

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

    def _format_search_results_block(
        self, retrieval_results: list[HybridDeepSearchRetrievalResult]
    ) -> tuple[str, list[int | str], list[str]]:
        """Format one paper-style query-labelled search results block."""
        lines = ["<|begin search results|>"]
        block_doc_ids: list[int | str] = []
        block_contents: list[str] = []
        for retrieval_result in retrieval_results:
            if retrieval_result.failure is not None:
                lines.append(f"{retrieval_result.query}: [retrieval failed]")
                continue
            doc_ids, contents = self._contents_from_results(retrieval_result.results[: self.evidence_budget])
            block_doc_ids.extend(doc_ids)
            block_contents.extend([content for content in contents if content])
            joined_contents = " ".join(content for content in contents[: self.evidence_budget] if content)
            lines.append(
                f"{retrieval_result.query}: {joined_contents}" if joined_contents else f"{retrieval_result.query}:"
            )
        lines.append("<|end search results|>")
        return "\n".join(lines), block_doc_ids, block_contents

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate an answer with paper-style HDS rolling search context."""
        query_text = self._service.get_query_text(query_id)
        tracker = TokenUsageTracker()
        interaction_log: list[str] = []
        retrieved_doc_ids: list[int | str] = []
        retrieval_failures: list[HybridDeepSearchRetrievalFailure] = []
        final_answer = ""
        search_calls_used = 0
        terminated_by = "max_turns"
        for turn in range(1, self.max_turns + 1):
            remaining_search_calls = self.max_search_calls - search_calls_used
            if remaining_search_calls <= 0:
                terminated_by = "max_search_calls"
                break

            plan_prompt = self._build_plan_prompt(query_text, interaction_log, turn, remaining_search_calls)
            response = await self._llm.ainvoke(plan_prompt)
            tracker.record(response)
            response_text = self._extract_text(response)
            interaction_log.append(response_text)
            action = parse_hybrid_deep_search_action(response_text, self.max_parallel_queries)
            if action.kind == "answer":
                final_answer = action.text
                terminated_by = "answer"
                break

            queries = action.queries[:remaining_search_calls]
            effective_k = max(1, min(top_k, self.k_per_query)) if top_k > 0 else self.k_per_query
            retrieval_results, failures = await self._retrieve_parallel(queries, effective_k)
            search_calls_used += len(queries)
            retrieval_failures.extend(failures)
            merged_results = self._merge_results([result.results for result in retrieval_results])
            for doc_id in (result.get("doc_id") for result in merged_results):
                if doc_id is not None and doc_id not in retrieved_doc_ids:
                    retrieved_doc_ids.append(doc_id)
            result_block, _block_doc_ids, _block_contents = self._format_search_results_block(retrieval_results)
            interaction_log.append(result_block)
            if search_calls_used >= self.max_search_calls:
                terminated_by = "max_search_calls"
                break

        if not final_answer and self.fallback_to_final_prompt:
            final_prompt = self._build_final_prompt(query_text, interaction_log)
            final_response = await self._llm.ainvoke(final_prompt)
            tracker.record(final_response)
            final_answer = self._extract_text(final_response)
            terminated_by = f"{terminated_by}_fallback"

        return GenerationResult(
            text=final_answer,
            token_usage=tracker.total,
            metadata={
                "trace": interaction_log,
                "interaction_log": interaction_log,
                "retrieved_chunk_ids": retrieved_doc_ids,
                "retrieval_failures": [
                    {"query": failure.query, "error": failure.error} for failure in retrieval_failures
                ],
                "search_calls_used": search_calls_used,
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
