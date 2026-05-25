"""INTERACT-RAG-style inference pipeline for AutoRAG-Research.

The ICLR 2026 INTERACT-RAG paper equips an agent with corpus-interaction
primitives beyond black-box query issuing: semantic/exact search, weighted
fusion, entity matching, include/exclude document controls, and retrieval-scale
adjustment. This module implements the training-free inference slice that fits
AutoRAG-Research by mapping those primitives onto an existing retrieval pipeline
and preserving rich interaction traces for evaluation.
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

InteractActionKind = Literal[
    "semantic_search",
    "exact_search",
    "weighted_fusion",
    "entity_match",
    "include_docs",
    "exclude_docs",
    "adjust_scale",
    "answer",
]

DEFAULT_INTERACT_RAG_STEP_PROMPT = """You are an INTERACT-RAG agent that can reason and interact with the retrieval corpus.
Use exactly one XML-like action per step:
- <semantic_search>query</semantic_search>: dense/semantic evidence search
- <exact_search>keywords</exact_search>: lexical/exact evidence search
- <weighted_fusion semantic="0.6" exact="0.4">query</weighted_fusion>: combine semantic and exact intent
- <entity_match>entity name</entity_match>: focus retrieval around an entity
- <include_docs>1, 2</include_docs>: force known useful doc IDs into context
- <exclude_docs>3, 4</exclude_docs>: remove noisy doc IDs from future context
- <adjust_scale>8</adjust_scale>: change retrieval scale for later searches
- <answer>final answer</answer>: finish

Question:
{query}

Interaction budget: {steps_used}/{max_steps} steps used.
Current retrieval scale: {current_scale}
Included doc IDs: {included_doc_ids}
Excluded doc IDs: {excluded_doc_ids}

Interaction trace and evidence:
{scratchpad}

Next action:"""

DEFAULT_INTERACT_RAG_FINAL_PROMPT = """Answer the question using the INTERACT-RAG interaction trace and gathered evidence.

Question:
{query}

Interaction trace and evidence:
{scratchpad}

Return only the final answer."""

_ACTION_PATTERNS: tuple[tuple[InteractActionKind, re.Pattern[str]], ...] = (
    ("answer", re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)),
    ("semantic_search", re.compile(r"<semantic_search>\s*(.*?)\s*</semantic_search>", re.IGNORECASE | re.DOTALL)),
    ("exact_search", re.compile(r"<exact_search>\s*(.*?)\s*</exact_search>", re.IGNORECASE | re.DOTALL)),
    ("weighted_fusion", re.compile(r"<weighted_fusion(?:\s+[^>]*)?>\s*(.*?)\s*</weighted_fusion>", re.IGNORECASE | re.DOTALL)),
    ("entity_match", re.compile(r"<entity_match>\s*(.*?)\s*</entity_match>", re.IGNORECASE | re.DOTALL)),
    ("include_docs", re.compile(r"<include_docs>\s*(.*?)\s*</include_docs>", re.IGNORECASE | re.DOTALL)),
    ("exclude_docs", re.compile(r"<exclude_docs>\s*(.*?)\s*</exclude_docs>", re.IGNORECASE | re.DOTALL)),
    ("adjust_scale", re.compile(r"<adjust_scale>\s*(.*?)\s*</adjust_scale>", re.IGNORECASE | re.DOTALL)),
)
_WEIGHTED_FUSION_ATTR_RE = re.compile(
    r"<weighted_fusion\s+semantic=['\"]?([0-9.]+)['\"]?\s+exact=['\"]?([0-9.]+)['\"]?",
    re.IGNORECASE,
)
_DOC_ID_RE = re.compile(r"-?\d+")


@dataclass(frozen=True)
class InteractRAGAction:
    """Parsed INTERACT-RAG action."""

    kind: InteractActionKind
    text: str
    semantic_weight: float | None = None
    exact_weight: float | None = None


def parse_interact_rag_action(response_text: str) -> InteractRAGAction:
    """Parse the first supported INTERACT-RAG action from an LLM response."""
    for action_kind, pattern in _ACTION_PATTERNS:
        match = pattern.search(response_text)
        if match is None:
            continue
        semantic_weight = None
        exact_weight = None
        if action_kind == "weighted_fusion":
            weight_match = _WEIGHTED_FUSION_ATTR_RE.search(response_text)
            if weight_match is not None:
                semantic_weight = float(weight_match.group(1))
                exact_weight = float(weight_match.group(2))
        return InteractRAGAction(
            kind=action_kind,
            text=match.group(1).strip(),
            semantic_weight=semantic_weight,
            exact_weight=exact_weight,
        )

    stripped_response = response_text.strip()
    if stripped_response.lower().startswith("answer:"):
        return InteractRAGAction(kind="answer", text=stripped_response.split(":", 1)[1].strip())
    return InteractRAGAction(kind="answer", text=stripped_response)


def parse_doc_ids(text: str) -> list[int]:
    """Parse document IDs from comma/space/free-form text."""
    return [int(match.group(0)) for match in _DOC_ID_RE.finditer(text)]


@dataclass
class InteractRAGState:
    """Mutable retrieval interaction state."""

    included_doc_ids: list[int | str] = field(default_factory=list)
    excluded_doc_ids: list[int | str] = field(default_factory=list)
    current_scale: int = 5
    semantic_weight: float = 0.5
    exact_weight: float = 0.5


@dataclass(kw_only=True)
class InteractRAGPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for the INTERACT-RAG inference pipeline."""

    step_prompt_template: str = field(default=DEFAULT_INTERACT_RAG_STEP_PROMPT)
    final_prompt_template: str = field(default=DEFAULT_INTERACT_RAG_FINAL_PROMPT)
    max_steps: int = 6
    initial_scale: int = 5
    max_scale: int = 20
    evidence_budget: int = 12
    fallback_to_final_prompt: bool = True

    def get_pipeline_class(self) -> type[InteractRAGPipeline]:
        """Return the InteractRAGPipeline class."""
        return InteractRAGPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for InteractRAGPipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)
        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "step_prompt_template": self.step_prompt_template,
            "final_prompt_template": self.final_prompt_template,
            "max_steps": self.max_steps,
            "initial_scale": self.initial_scale,
            "max_scale": self.max_scale,
            "evidence_budget": self.evidence_budget,
            "fallback_to_final_prompt": self.fallback_to_final_prompt,
        }


class InteractRAGPipeline(BaseGenerationPipeline):
    """Training-free INTERACT-RAG generation pipeline."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        step_prompt_template: str = DEFAULT_INTERACT_RAG_STEP_PROMPT,
        final_prompt_template: str = DEFAULT_INTERACT_RAG_FINAL_PROMPT,
        max_steps: int = 6,
        initial_scale: int = 5,
        max_scale: int = 20,
        evidence_budget: int = 12,
        fallback_to_final_prompt: bool = True,
        schema: Any | None = None,
    ):
        """Initialize the INTERACT-RAG pipeline."""
        if max_steps < 1:
            msg = "max_steps must be >= 1"
            raise ValueError(msg)
        if initial_scale < 1 or max_scale < 1:
            msg = "initial_scale and max_scale must be >= 1"
            raise ValueError(msg)
        if evidence_budget < 1:
            msg = "evidence_budget must be >= 1"
            raise ValueError(msg)

        self.step_prompt_template = step_prompt_template
        self.final_prompt_template = final_prompt_template
        self.max_steps = max_steps
        self.initial_scale = min(initial_scale, max_scale)
        self.max_scale = max_scale
        self.evidence_budget = evidence_budget
        self.fallback_to_final_prompt = fallback_to_final_prompt

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return INTERACT-RAG pipeline configuration."""
        model_name = getattr(self._llm, "model_name", None)
        if model_name is None or not isinstance(model_name, str):
            model_name = type(self._llm).__name__
        return {
            "type": "interact_rag",
            "step_prompt_template": self.step_prompt_template,
            "final_prompt_template": self.final_prompt_template,
            "max_steps": self.max_steps,
            "initial_scale": self.initial_scale,
            "max_scale": self.max_scale,
            "evidence_budget": self.evidence_budget,
            "fallback_to_final_prompt": self.fallback_to_final_prompt,
            "retrieval_pipeline_id": self._retrieval_pipeline.pipeline_id,
            "llm_model": model_name,
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text content from an LLM response."""
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def _format_doc_ids(doc_ids: list[int | str]) -> str:
        """Format doc IDs for prompts."""
        return ", ".join(str(doc_id) for doc_id in doc_ids) if doc_ids else "(none)"

    @staticmethod
    def _format_scratchpad(trace: list[str], evidence: list[str]) -> str:
        """Format trace and evidence for prompting."""
        sections: list[str] = []
        if trace:
            sections.append("Trace:\n" + "\n".join(trace))
        if evidence:
            sections.append("Evidence:\n" + "\n\n".join(f"[{i + 1}] {item}" for i, item in enumerate(evidence)))
        return "\n\n".join(sections) if sections else "(empty)"

    def _build_step_prompt(
        self,
        query: str,
        state: InteractRAGState,
        trace: list[str],
        evidence: list[str],
        steps_used: int,
    ) -> str:
        """Build one interaction prompt."""
        return self.step_prompt_template.format(
            query=query,
            scratchpad=self._format_scratchpad(trace, evidence),
            steps_used=steps_used,
            max_steps=self.max_steps,
            current_scale=state.current_scale,
            included_doc_ids=self._format_doc_ids(state.included_doc_ids),
            excluded_doc_ids=self._format_doc_ids(state.excluded_doc_ids),
        )

    def _build_final_prompt(self, query: str, trace: list[str], evidence: list[str]) -> str:
        """Build fallback answer prompt."""
        return self.final_prompt_template.format(query=query, scratchpad=self._format_scratchpad(trace, evidence))

    def _contents_from_results(self, retrieval_results: list[dict[str, Any]]) -> tuple[list[int | str], list[str]]:
        """Extract IDs and contents from retrieval results, backfilling when needed."""
        doc_ids: list[int | str] = []
        contents: list[str | None] = []
        missing_positions: list[int] = []
        missing_ids: list[int | str] = []
        for result in retrieval_results:
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

    def _apply_state_action(self, action: InteractRAGAction, state: InteractRAGState, trace: list[str]) -> bool:
        """Apply non-search context-shaping actions. Return True if handled."""
        if action.kind == "include_docs":
            for doc_id in parse_doc_ids(action.text):
                if doc_id not in state.included_doc_ids:
                    state.included_doc_ids.append(doc_id)
            trace.append(f"include_docs: {self._format_doc_ids(state.included_doc_ids)}")
            return True
        if action.kind == "exclude_docs":
            for doc_id in parse_doc_ids(action.text):
                if doc_id not in state.excluded_doc_ids:
                    state.excluded_doc_ids.append(doc_id)
            trace.append(f"exclude_docs: {self._format_doc_ids(state.excluded_doc_ids)}")
            return True
        if action.kind == "adjust_scale":
            parsed_values = parse_doc_ids(action.text)
            if parsed_values:
                state.current_scale = min(max(parsed_values[0], 1), self.max_scale)
            trace.append(f"adjust_scale: {state.current_scale}")
            return True
        if action.kind == "weighted_fusion":
            if action.semantic_weight is not None and action.exact_weight is not None:
                state.semantic_weight = action.semantic_weight
                state.exact_weight = action.exact_weight
            return False
        return False

    def _build_retrieval_query(self, action: InteractRAGAction) -> str:
        """Translate an interaction primitive into a retrieval query string."""
        if action.kind == "entity_match":
            return f"entity: {action.text}"
        if action.kind == "exact_search":
            return f"exact: {action.text}"
        if action.kind == "weighted_fusion":
            semantic_weight = action.semantic_weight if action.semantic_weight is not None else 0.5
            exact_weight = action.exact_weight if action.exact_weight is not None else 0.5
            return f"semantic weight {semantic_weight:.2f}; exact weight {exact_weight:.2f}; query: {action.text}"
        return action.text

    def _filter_results(
        self,
        retrieval_results: list[dict[str, Any]],
        state: InteractRAGState,
    ) -> list[dict[str, Any]]:
        """Apply include/exclude state to retrieval results."""
        excluded = set(state.excluded_doc_ids)
        filtered_results = [result for result in retrieval_results if result.get("doc_id") not in excluded]
        included = [
            {"doc_id": doc_id, "score": float("inf")}
            for doc_id in state.included_doc_ids
            if doc_id not in excluded and all(result.get("doc_id") != doc_id for result in filtered_results)
        ]
        return [*included, *filtered_results]

    def _append_evidence(self, evidence: list[str], new_contents: list[str]) -> list[str]:
        """Append deduplicated evidence while respecting evidence_budget."""
        updated = list(evidence)
        for content in new_contents:
            if content and content not in updated:
                updated.append(content)
        return updated[-self.evidence_budget :]

    async def _execute_retrieval_action(
        self,
        action: InteractRAGAction,
        state: InteractRAGState,
        trace: list[str],
        evidence: list[str],
        retrieved_doc_ids: list[int | str],
    ) -> list[str]:
        """Run one retrieval-like action and update evidence."""
        retrieval_query = self._build_retrieval_query(action)
        retrieval_results = await self._retrieval_pipeline.retrieve(retrieval_query, state.current_scale)
        filtered_results = self._filter_results(retrieval_results, state)
        doc_ids, contents = self._contents_from_results(filtered_results)
        for doc_id in doc_ids:
            if doc_id not in retrieved_doc_ids:
                retrieved_doc_ids.append(doc_id)
        trace.append(f"{action.kind}: {action.text}")
        return self._append_evidence(evidence, contents)

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate an answer through INTERACT-RAG corpus interaction primitives."""
        query_text = self._service.get_query_text(query_id)
        state = InteractRAGState(current_scale=min(max(top_k, self.initial_scale), self.max_scale))
        tracker = TokenUsageTracker()
        trace: list[str] = []
        evidence: list[str] = []
        retrieved_doc_ids: list[int | str] = []
        final_answer = ""
        terminated_by = "max_steps"

        for step_index in range(self.max_steps):
            prompt = self._build_step_prompt(query_text, state, trace, evidence, step_index)
            response = await self._llm.ainvoke(prompt)
            tracker.record(response)
            action = parse_interact_rag_action(self._extract_text(response))
            if action.kind == "answer":
                final_answer = action.text
                trace.append(f"answer: {final_answer}")
                terminated_by = "answer"
                break
            if self._apply_state_action(action, state, trace):
                continue
            evidence = await self._execute_retrieval_action(action, state, trace, evidence, retrieved_doc_ids)

        if not final_answer and self.fallback_to_final_prompt:
            final_prompt = self._build_final_prompt(query_text, trace, evidence)
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
                "included_doc_ids": state.included_doc_ids,
                "excluded_doc_ids": state.excluded_doc_ids,
                "current_scale": state.current_scale,
                "semantic_weight": state.semantic_weight,
                "exact_weight": state.exact_weight,
                "terminated_by": terminated_by,
            },
        )


__all__ = [
    "DEFAULT_INTERACT_RAG_FINAL_PROMPT",
    "DEFAULT_INTERACT_RAG_STEP_PROMPT",
    "InteractRAGAction",
    "InteractRAGPipeline",
    "InteractRAGPipelineConfig",
    "InteractRAGState",
    "parse_doc_ids",
    "parse_interact_rag_action",
]
