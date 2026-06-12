"""DeepRAG subquery-level MDP generation pipeline.

DeepRAG is represented here as an inference-time Markov decision process over
standalone subqueries. At each step the controller either terminates with a
final answer or emits a subquery plus an atomic retrieve/parametric decision;
a second model call then produces the intermediate answer for that subquery.
Training/data-construction parts from the paper are intentionally out of scope
for this repo-native baseline.
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

DeepRAGActionKind = Literal["retrieve", "parametric", "answer", "invalid"]
DeepRAGDecision = Literal["retrieve", "parametric"]

DEFAULT_DEEPRAG_STEP_PROMPT = """You are a DeepRAG controller for subquery-level question answering.
At each step, inspect the question and ordered trajectory, then return exactly one action:
- <retrieve>standalone subquery</retrieve> when the subquery needs external evidence
- <parametric>standalone subquery</parametric> when model knowledge is enough for the subquery
- <answer>final answer</answer> when the trajectory is sufficient to answer

Question:
{query}

Step: {step}/{max_steps}

Ordered trajectory (each retrieve step includes its own Context passages):
{trajectory}

Next action:"""

DEFAULT_DEEPRAG_INTERMEDIATE_PROMPT = """Generate the DeepRAG intermediate answer for the current standalone subquery.
Answer only the subquery, using retrieved passages when provided; otherwise answer from parametric knowledge.

Original question:
{query}

Ordered trajectory so far:
{trajectory}

Current subquery:
{subquery}
{passages_section}
Intermediate answer:"""

DEFAULT_DEEPRAG_FINAL_PROMPT = """Answer the original question from the full DeepRAG subquery trajectory.
This fallback is a documented repo adaptation for untrained controller models that may never emit <answer>.

Question:
{query}

Ordered trajectory (each retrieve step includes its own Context passages):
{trajectory}

Final answer:"""

_RETRIEVE_RE = re.compile(r"<retrieve>\s*(.*?)\s*</retrieve>", re.IGNORECASE | re.DOTALL)
_PARAMETRIC_RE = re.compile(r"<parametric>\s*(.*?)\s*</parametric>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)

_TRAJECTORY_PASSAGE_CHARS = 800


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

    parametric_match = _PARAMETRIC_RE.search(response_text)
    if parametric_match is not None:
        return DeepRAGAction(kind="parametric", text=parametric_match.group(1).strip())

    stripped_response = response_text.strip()
    if stripped_response.lower().startswith("answer:"):
        return DeepRAGAction(kind="answer", text=stripped_response.split(":", 1)[1].strip())
    return DeepRAGAction(kind="invalid", text=stripped_response)


@dataclass(kw_only=True)
class DeepRAGPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for DeepRAGPipeline."""

    step_prompt_template: str = field(default=DEFAULT_DEEPRAG_STEP_PROMPT)
    intermediate_prompt_template: str = field(default=DEFAULT_DEEPRAG_INTERMEDIATE_PROMPT)
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
            "intermediate_prompt_template": self.intermediate_prompt_template,
            "final_prompt_template": self.final_prompt_template,
            "max_steps": self.max_steps,
            "k_per_retrieval": self.k_per_retrieval,
            "evidence_budget": self.evidence_budget,
            "fallback_to_final_prompt": self.fallback_to_final_prompt,
        }


class DeepRAGPipeline(BaseGenerationPipeline):
    """Adaptive DeepRAG subquery-level MDP generation pipeline."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        step_prompt_template: str = DEFAULT_DEEPRAG_STEP_PROMPT,
        intermediate_prompt_template: str = DEFAULT_DEEPRAG_INTERMEDIATE_PROMPT,
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
        self.intermediate_prompt_template = intermediate_prompt_template
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
            "intermediate_prompt_template": self.intermediate_prompt_template,
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

    @staticmethod
    def _format_trajectory(trajectory: list[dict[str, Any]], empty: str = "(none)") -> str:
        """Format the ordered DeepRAG trajectory using paper-style narrative records.

        Retrieve records carry their own bounded passage context so the trajectory is the
        complete MDP state ``(q_i, r_i)`` including retrieved documents.
        """
        if not trajectory:
            return empty
        formatted_steps: list[str] = []
        for record in trajectory:
            lines = [f"Follow up: {record['subquery']}"]
            passages = record.get("retrieved_passages", [])
            if record["decision"] == "retrieve":
                lines.append("Context:")
                if passages:
                    lines.extend(f"[{index + 1}] {passage}" for index, passage in enumerate(passages))
                else:
                    lines.append("(no passages retrieved)")
            lines.append(f"Intermediate answer: {record['intermediate_answer']}")
            formatted_steps.append("\n".join(lines))
        return "\n\n".join(formatted_steps)

    def _build_step_prompt(
        self,
        query: str,
        evidence: list[str],
        trajectory: list[dict[str, Any]],
        step: int,
    ) -> str:
        """Build one DeepRAG controller prompt."""
        return self.step_prompt_template.format(
            query=query,
            evidence=self._format_items(evidence),
            trajectory=self._format_trajectory(trajectory),
            step=step,
            max_steps=self.max_steps,
        )

    def _build_intermediate_prompt(
        self,
        query: str,
        trajectory: list[dict[str, Any]],
        subquery: str,
        passages: list[str],
    ) -> str:
        """Build the intermediate-answer prompt for a retrieve or parametric decision."""
        passages_section = ""
        if passages:
            passages_section = f"\nRetrieved passages:\n{self._format_items(passages)}\n"
        return self.intermediate_prompt_template.format(
            query=query,
            trajectory=self._format_trajectory(trajectory),
            subquery=subquery,
            passages=self._format_items(passages),
            passages_section=passages_section,
        )

    def _build_final_prompt(self, query: str, evidence: list[str], trajectory: list[dict[str, Any]]) -> str:
        """Build fallback final prompt."""
        return self.final_prompt_template.format(
            query=query,
            evidence=self._format_items(evidence),
            trajectory=self._format_trajectory(trajectory),
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
        """Generate an answer using DeepRAG subquery-level control actions."""
        query_text = self._service.get_query_text(query_id)
        retrieval_k = self._resolve_retrieval_k(top_k)
        tracker = TokenUsageTracker()
        evidence: list[str] = []
        trajectory: list[dict[str, Any]] = []
        follow_up_queries: list[str] = []
        retrieved_chunk_ids: list[int | str] = []
        retrieved_scores: list[float] = []
        final_answer = ""
        terminated_by = "max_steps"

        for step in range(1, self.max_steps + 1):
            prompt = self._build_step_prompt(query_text, evidence, trajectory, step)
            response = await self._llm.ainvoke(prompt)
            tracker.record(response)
            action = parse_deeprag_action(self._extract_text(response))
            if action.kind == "answer":
                final_answer = action.text
                terminated_by = "answer"
                break
            if action.kind == "invalid":
                logger.warning("Invalid DeepRAG controller output at step %s: %s", step, action.text)
                trajectory.append({
                    "step": step,
                    "subquery": "(invalid controller output)",
                    "decision": "parametric",
                    "retrieved_chunk_ids": [],
                    "retrieved_passages": [],
                    "intermediate_answer": f"Invalid controller output ignored: {action.text}",
                })
                continue

            subquery = action.text
            follow_up_queries.append(subquery)
            contents: list[str] = []
            doc_ids: list[int | str] = []
            if action.kind == "retrieve":
                results = await self._retrieval_pipeline.retrieve(subquery, retrieval_k)
                doc_ids, contents, scores = self._contents_from_results(results)
                for doc_id, score in zip(doc_ids, scores, strict=False):
                    if doc_id not in retrieved_chunk_ids:
                        retrieved_chunk_ids.append(doc_id)
                        retrieved_scores.append(score)
                evidence = self._append_evidence(evidence, contents)

            intermediate_prompt = self._build_intermediate_prompt(query_text, trajectory, subquery, contents)
            intermediate_response = await self._llm.ainvoke(intermediate_prompt)
            tracker.record(intermediate_response)
            trajectory.append({
                "step": step,
                "subquery": subquery,
                "decision": action.kind,
                "retrieved_chunk_ids": doc_ids,
                "retrieved_passages": [content[:_TRAJECTORY_PASSAGE_CHARS] for content in contents if content],
                "intermediate_answer": self._extract_text(intermediate_response),
            })

        if not final_answer and self.fallback_to_final_prompt:
            final_prompt = self._build_final_prompt(query_text, evidence, trajectory)
            final_response = await self._llm.ainvoke(final_prompt)
            tracker.record(final_response)
            final_answer = self._extract_text(final_response)
            terminated_by = f"{terminated_by}_fallback"

        return GenerationResult(
            text=final_answer,
            token_usage=tracker.total,
            metadata={
                "trajectory": trajectory,
                "trace": self._format_trajectory(trajectory),
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
    "DEFAULT_DEEPRAG_INTERMEDIATE_PROMPT",
    "DEFAULT_DEEPRAG_STEP_PROMPT",
    "DeepRAGAction",
    "DeepRAGPipeline",
    "DeepRAGPipelineConfig",
    "parse_deeprag_action",
]
