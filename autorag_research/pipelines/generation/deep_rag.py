"""DeepRAG-style adaptive iterative generation pipeline.

DeepRAG is best represented in AutoRAG-Research as an inference-time generation
pipeline that decomposes a question into stepwise subqueries, decides whether a
retrieval step is needed, accumulates evidence, and stops once enough evidence is
available. Training/data-construction parts from the paper are intentionally out
of scope for this repo-native baseline.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger("AutoRAG-Research")

DeepRAGActionKind = Literal["retrieve", "reason", "answer"]

DEFAULT_DEEPRAG_STEP_PROMPT = """You are a DeepRAG controller for adaptive multi-step question answering.
At each step, decide whether retrieval is needed or whether you can reason from existing evidence.
Return exactly one action:
- <retrieve>standalone subquery</retrieve> when more evidence is needed
- <reason>short reasoning update</reason> when no retrieval is needed for this step
- <answer>final answer</answer> when enough evidence has been gathered

Question:
{query}

Step: {step}/{max_steps}
Evidence:
{evidence}

Reasoning trace:
{trace}

Next action:"""

DEFAULT_DEEPRAG_FINAL_PROMPT = """Answer the question using the DeepRAG reasoning trace and evidence.

Question:
{query}

Evidence:
{evidence}

Reasoning trace:
{trace}

Final answer:"""

_RETRIEVE_RE = re.compile(r"<retrieve>\s*(.*?)\s*</retrieve>", re.IGNORECASE | re.DOTALL)
_REASON_RE = re.compile(r"<reason>\s*(.*?)\s*</reason>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class DeepRAGAction:
    """Parsed DeepRAG controller action."""

    kind: DeepRAGActionKind
    text: str


def parse_deeprag_action(response_text: str) -> DeepRAGAction:
    """Parse a DeepRAG controller action from an LLM response."""
    answer_match = _ANSWER_RE.search(response_text)
    if answer_match is not None:
        return DeepRAGAction(kind="answer", text=answer_match.group(1).strip())

    retrieve_match = _RETRIEVE_RE.search(response_text)
    if retrieve_match is not None:
        return DeepRAGAction(kind="retrieve", text=retrieve_match.group(1).strip())

    reason_match = _REASON_RE.search(response_text)
    if reason_match is not None:
        return DeepRAGAction(kind="reason", text=reason_match.group(1).strip())

    stripped_response = response_text.strip()
    if stripped_response.lower().startswith("answer:"):
        return DeepRAGAction(kind="answer", text=stripped_response.split(":", 1)[1].strip())
    return DeepRAGAction(kind="reason", text=stripped_response)


@dataclass(kw_only=True)
class DeepRAGPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for DeepRAGPipeline."""

    step_prompt_template: str = field(default=DEFAULT_DEEPRAG_STEP_PROMPT)
    final_prompt_template: str = field(default=DEFAULT_DEEPRAG_FINAL_PROMPT)
    max_steps: int = 5
    k_per_retrieval: int = 5
    evidence_budget: int = 12
    fallback_to_final_prompt: bool = True

    def get_pipeline_class(self) -> type[DeepRAGPipeline]:
        """Return the DeepRAGPipeline class."""
        return DeepRAGPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for DeepRAGPipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)
        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "step_prompt_template": self.step_prompt_template,
            "final_prompt_template": self.final_prompt_template,
            "max_steps": self.max_steps,
            "k_per_retrieval": self.k_per_retrieval,
            "evidence_budget": self.evidence_budget,
            "fallback_to_final_prompt": self.fallback_to_final_prompt,
        }


class DeepRAGPipeline(BaseGenerationPipeline):
    """Adaptive iterative retrieve-or-reason generation pipeline."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        step_prompt_template: str = DEFAULT_DEEPRAG_STEP_PROMPT,
        final_prompt_template: str = DEFAULT_DEEPRAG_FINAL_PROMPT,
        max_steps: int = 5,
        k_per_retrieval: int = 5,
        evidence_budget: int = 12,
        fallback_to_final_prompt: bool = True,
        schema: Any | None = None,
    ):
        """Initialize DeepRAGPipeline."""
        if max_steps < 1:
            msg = "max_steps must be >= 1"
            raise ValueError(msg)
        if k_per_retrieval < 1:
            msg = "k_per_retrieval must be >= 1"
            raise ValueError(msg)
        if evidence_budget < 1:
            msg = "evidence_budget must be >= 1"
            raise ValueError(msg)

        self.step_prompt_template = step_prompt_template
        self.final_prompt_template = final_prompt_template
        self.max_steps = max_steps
        self.k_per_retrieval = k_per_retrieval
        self.evidence_budget = evidence_budget
        self.fallback_to_final_prompt = fallback_to_final_prompt

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return DeepRAG configuration for storage."""
        model_name = getattr(self._llm, "model_name", None)
        if model_name is None or not isinstance(model_name, str):
            model_name = type(self._llm).__name__
        return {
            "type": "deeprag",
            "step_prompt_template": self.step_prompt_template,
            "final_prompt_template": self.final_prompt_template,
            "max_steps": self.max_steps,
            "k_per_retrieval": self.k_per_retrieval,
            "evidence_budget": self.evidence_budget,
            "fallback_to_final_prompt": self.fallback_to_final_prompt,
            "retrieval_pipeline_id": self._retrieval_pipeline.pipeline_id,
            "llm_model": model_name,
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract content from a LangChain response."""
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def _format_items(items: list[str], empty: str = "(none)") -> str:
        """Format a list for prompts."""
        if not items:
            return empty
        return "\n\n".join(f"[{index + 1}] {item}" for index, item in enumerate(items))

    def _build_step_prompt(self, query: str, evidence: list[str], trace: list[str], step: int) -> str:
        """Build one DeepRAG control prompt."""
        return self.step_prompt_template.format(
            query=query,
            evidence=self._format_items(evidence),
            trace=self._format_items(trace),
            step=step,
            max_steps=self.max_steps,
        )

    def _build_final_prompt(self, query: str, evidence: list[str], trace: list[str]) -> str:
        """Build fallback final prompt."""
        return self.final_prompt_template.format(
            query=query,
            evidence=self._format_items(evidence),
            trace=self._format_items(trace),
        )

    def _contents_from_results(self, results: list[dict[str, Any]]) -> tuple[list[int | str], list[str], list[float]]:
        """Extract doc IDs, contents, and scores from retrieval results."""
        doc_ids: list[int | str] = []
        contents: list[str | None] = []
        scores: list[float] = []
        missing_positions: list[int] = []
        missing_ids: list[int | str] = []
        for result in results:
            doc_id = result.get("doc_id")
            if doc_id is None:
                continue
            doc_ids.append(doc_id)
            score = result.get("score", 0.0)
            scores.append(float(score) if isinstance(score, (int, float)) else 0.0)
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
        return doc_ids, [content or "" for content in contents], scores

    def _append_evidence(self, evidence: list[str], new_contents: list[str]) -> list[str]:
        """Append non-empty deduplicated evidence within budget."""
        updated = list(evidence)
        for content in new_contents:
            if content and content not in updated:
                updated.append(content)
        return updated[-self.evidence_budget :]

    def _resolve_retrieval_k(self, top_k: int) -> int:
        """Resolve per-step retrieval size using top_k as a global runner cap when provided."""
        return max(1, min(top_k, self.k_per_retrieval)) if top_k > 0 else self.k_per_retrieval

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate an answer using adaptive DeepRAG control actions."""
        query_text = self._service.get_query_text(query_id)
        retrieval_k = self._resolve_retrieval_k(top_k)
        tracker = TokenUsageTracker()
        evidence: list[str] = []
        trace: list[str] = []
        follow_up_queries: list[str] = []
        retrieved_chunk_ids: list[int | str] = []
        retrieved_scores: list[float] = []
        final_answer = ""
        terminated_by = "max_steps"

        for step in range(1, self.max_steps + 1):
            prompt = self._build_step_prompt(query_text, evidence, trace, step)
            response = await self._llm.ainvoke(prompt)
            tracker.record(response)
            action = parse_deeprag_action(self._extract_text(response))
            if action.kind == "answer":
                final_answer = action.text
                trace.append(f"answer: {final_answer}")
                terminated_by = "answer"
                break
            if action.kind == "reason":
                trace.append(f"reason: {action.text}")
                continue

            follow_up_queries.append(action.text)
            trace.append(f"retrieve: {action.text}")
            results = await self._retrieval_pipeline.retrieve(action.text, retrieval_k)
            doc_ids, contents, scores = self._contents_from_results(results)
            for doc_id, score in zip(doc_ids, scores, strict=False):
                if doc_id not in retrieved_chunk_ids:
                    retrieved_chunk_ids.append(doc_id)
                    retrieved_scores.append(score)
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
                "follow_up_queries": follow_up_queries,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieved_scores": retrieved_scores,
                "effective_retrieval_k": retrieval_k,
                "top_k_cap": top_k,
                "terminated_by": terminated_by,
            },
        )


__all__ = [
    "DEFAULT_DEEPRAG_FINAL_PROMPT",
    "DEFAULT_DEEPRAG_STEP_PROMPT",
    "DeepRAGAction",
    "DeepRAGPipeline",
    "DeepRAGPipelineConfig",
    "parse_deeprag_action",
]
