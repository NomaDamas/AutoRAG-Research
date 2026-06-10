"""INTERACT-RAG-style inference pipeline for AutoRAG-Research.

The ICLR 2026 INTERACT-RAG paper equips an agent with real corpus-interaction
primitives beyond black-box query issuing: semantic dense search, exact sparse
search, score-normalized weighted fusion, entity matching, include/exclude
document controls, and retrieval-scale adjustment. This module implements the
training-free inference slice that fits AutoRAG-Research by routing exact
primitives through an optional sparse retrieval engine while preserving rich
interaction traces for evaluation.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseGenerationPipelineConfig
from autorag_research.orm.service.generation_pipeline import GenerationResult
from autorag_research.pipelines.generation.base import BaseGenerationPipeline
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.pipelines.retrieval.hybrid import HybridRetrievalPipeline
from autorag_research.util import TokenUsageTracker, normalize_minmax

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
You may combine several XML-like actions in one step; they will be executed in order after state controls are applied:
- <semantic_search>query</semantic_search>: dense semantic search over the primary retrieval engine
- <exact_search>keywords</exact_search>: exact/sparse search over the exact retrieval engine when available
- <weighted_fusion semantic="0.6" exact="0.4">query</weighted_fusion>: run semantic and exact engines, normalize scores, and fuse by weighted sum when available
- <entity_match>entity name</entity_match>: exact engine entity lookup returning the three most query-related snippets
- <include_docs>chunk_id</include_docs>: force known useful chunk IDs from displayed evidence into context
- <exclude_docs>chunk_id</exclude_docs>: remove noisy chunk IDs from displayed evidence and future context before retrieval
- <adjust_scale>8</adjust_scale>: change retrieval scale for this and later searches
- <answer>final answer</answer>: finish

Question:
{query}

Interaction budget: {steps_used}/{max_steps} steps used.
Current retrieval scale: {current_scale}
Included doc IDs: {included_doc_ids}
Excluded doc IDs: {excluded_doc_ids}

Interaction trace and evidence:
{scratchpad}

Next action(s):"""

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
    (
        "weighted_fusion",
        re.compile(r"<weighted_fusion(?:\s+[^>]*)?>\s*(.*?)\s*</weighted_fusion>", re.IGNORECASE | re.DOTALL),
    ),
    ("entity_match", re.compile(r"<entity_match>\s*(.*?)\s*</entity_match>", re.IGNORECASE | re.DOTALL)),
    ("include_docs", re.compile(r"<include_docs>\s*(.*?)\s*</include_docs>", re.IGNORECASE | re.DOTALL)),
    ("exclude_docs", re.compile(r"<exclude_docs>\s*(.*?)\s*</exclude_docs>", re.IGNORECASE | re.DOTALL)),
    ("adjust_scale", re.compile(r"<adjust_scale>\s*(.*?)\s*</adjust_scale>", re.IGNORECASE | re.DOTALL)),
)
_WEIGHTED_FUSION_ATTR_RE = re.compile(
    r"<weighted_fusion\s+semantic=['\"]?([0-9.]+)['\"]?\s+exact=['\"]?([0-9.]+)['\"]?",
    re.IGNORECASE,
)
_DOC_ID_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")

# Paper Appendix C.2: weighted fusion normalizes the top-20 chunks from each strategy before the weighted sum.
_FUSION_FETCH_K = 20


@dataclass(frozen=True)
class InteractRAGAction:
    """Parsed INTERACT-RAG action."""

    kind: InteractActionKind
    text: str
    semantic_weight: float | None = None
    exact_weight: float | None = None


def parse_interact_rag_actions(response_text: str) -> list[InteractRAGAction]:
    """Parse all supported INTERACT-RAG actions in order of appearance."""
    matches: list[tuple[int, InteractRAGAction]] = []
    for action_kind, pattern in _ACTION_PATTERNS:
        for match in pattern.finditer(response_text):
            semantic_weight = None
            exact_weight = None
            if action_kind == "weighted_fusion":
                weight_match = _WEIGHTED_FUSION_ATTR_RE.search(match.group(0))
                if weight_match is not None:
                    try:
                        semantic_weight = float(weight_match.group(1))
                        exact_weight = float(weight_match.group(2))
                    except ValueError:
                        semantic_weight = None
                        exact_weight = None
            matches.append((
                match.start(),
                InteractRAGAction(
                    kind=action_kind,
                    text=match.group(1).strip(),
                    semantic_weight=semantic_weight,
                    exact_weight=exact_weight,
                ),
            ))

    if matches:
        return [action for _, action in sorted(matches, key=lambda item: item[0])]

    stripped_response = response_text.strip()
    if stripped_response.lower().startswith("answer:"):
        return [InteractRAGAction(kind="answer", text=stripped_response.split(":", 1)[1].strip())]
    return [InteractRAGAction(kind="answer", text=stripped_response)]


def parse_interact_rag_action(response_text: str) -> InteractRAGAction:
    """Parse the first supported INTERACT-RAG action from an LLM response."""
    actions = parse_interact_rag_actions(response_text)
    answer_action = next((action for action in actions if action.kind == "answer"), None)
    return answer_action or actions[0]


def parse_doc_ids(text: str) -> list[int | str]:
    """Parse document IDs from comma/space/free-form text without splitting string IDs."""
    doc_ids: list[int | str] = []
    for token_match in _DOC_ID_TOKEN_RE.finditer(text):
        token = token_match.group(0).strip(".,;:")
        if not token or token.lower() in {"id", "ids", "and"}:
            continue
        if re.fullmatch(r"-?\d+", token):
            doc_ids.append(int(token))
        else:
            doc_ids.append(token)
    return doc_ids


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
    exact_retrieval_pipeline_name: str | None = None

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
            "exact_retrieval_pipeline": self.exact_retrieval_pipeline_name,
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
        exact_retrieval_pipeline: BaseRetrievalPipeline | str | None = None,
        step_prompt_template: str = DEFAULT_INTERACT_RAG_STEP_PROMPT,
        final_prompt_template: str = DEFAULT_INTERACT_RAG_FINAL_PROMPT,
        max_steps: int = 6,
        initial_scale: int = 5,
        max_scale: int = 20,
        evidence_budget: int = 12,
        fallback_to_final_prompt: bool = True,
        schema: Any | None = None,
        config_dir: Path | None = None,
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
        exact_retrieval_pipeline_name = exact_retrieval_pipeline if isinstance(exact_retrieval_pipeline, str) else None
        if isinstance(exact_retrieval_pipeline, str):
            exact_retrieval_pipeline = HybridRetrievalPipeline._load_pipeline(
                exact_retrieval_pipeline,
                session_factory,
                schema,
                config_dir,
            )
        self._exact_retrieval_pipeline = exact_retrieval_pipeline
        self._exact_retrieval_pipeline_name = exact_retrieval_pipeline_name

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
            "exact_retrieval_pipeline_id": getattr(self._exact_retrieval_pipeline, "pipeline_id", None),
            "exact_retrieval_pipeline_name": self._exact_retrieval_pipeline_name,
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
    def _format_evidence_item(item: str | dict[str, Any]) -> str:
        """Format one evidence item with exposed chunk IDs when available."""
        if not isinstance(item, dict):
            return item

        doc_id = item.get("doc_id")
        content = str(item.get("content", ""))
        if doc_id is None:
            return content

        score = item.get("score")
        score_label = "" if score is None else f" score={score}"
        return f"[chunk_id={doc_id}{score_label}] {content}"

    @classmethod
    def _format_scratchpad(cls, trace: list[str], evidence: Sequence[str | dict[str, Any]]) -> str:
        """Format trace and evidence for prompting."""
        sections: list[str] = []
        if trace:
            sections.append("Trace:\n" + "\n".join(trace))
        if evidence:
            sections.append("Evidence:\n" + "\n\n".join(cls._format_evidence_item(item) for item in evidence))
        return "\n\n".join(sections) if sections else "(empty)"

    def _build_step_prompt(
        self,
        query: str,
        state: InteractRAGState,
        trace: list[str],
        evidence: list[dict[str, Any]],
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

    def _build_final_prompt(self, query: str, trace: list[str], evidence: list[dict[str, Any]]) -> str:
        """Build fallback answer prompt."""
        return self.final_prompt_template.format(query=query, scratchpad=self._format_scratchpad(trace, evidence))

    def _evidence_from_results(self, retrieval_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract displayable evidence snippets from retrieval results, backfilling missing content when needed."""
        doc_ids: list[int | str] = []
        contents: list[str | None] = []
        missing_positions: list[int] = []
        missing_ids: list[int | str] = []
        scores: list[Any] = []
        for result in retrieval_results:
            doc_id = result.get("doc_id")
            if doc_id is None:
                continue
            doc_ids.append(doc_id)
            scores.append(result.get("score"))
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
        return [
            {"doc_id": doc_id, "score": score, "content": content or ""}
            for doc_id, score, content in zip(doc_ids, scores, contents, strict=False)
        ]

    @staticmethod
    def _known_doc_ids(exposed_doc_ids: set[int | str]) -> set[int | str]:
        """Return exposed IDs plus their string forms for tolerant matching."""
        return exposed_doc_ids | {str(doc_id) for doc_id in exposed_doc_ids}

    @staticmethod
    def _resolve_requested_doc_ids(
        requested_doc_ids: list[int | str],
        exposed_doc_ids: set[int | str],
    ) -> tuple[list[int | str], list[int | str]]:
        """Resolve requested document IDs to exposed IDs while preserving their stored type."""
        exposed_by_label = {str(doc_id): doc_id for doc_id in exposed_doc_ids}
        accepted_doc_ids: list[int | str] = []
        ignored_doc_ids: list[int | str] = []
        for requested_doc_id in requested_doc_ids:
            resolved_doc_id = exposed_by_label.get(str(requested_doc_id))
            if resolved_doc_id is None:
                ignored_doc_ids.append(requested_doc_id)
            else:
                accepted_doc_ids.append(resolved_doc_id)
        return accepted_doc_ids, ignored_doc_ids

    def _apply_doc_id_action(
        self,
        action: InteractRAGAction,
        state: InteractRAGState,
        trace: list[str],
        exposed_doc_ids: set[int | str],
    ) -> bool:
        """Apply include/exclude actions only to IDs shown to the model."""
        target_list = state.included_doc_ids if action.kind == "include_docs" else state.excluded_doc_ids
        requested_doc_ids = parse_doc_ids(action.text)
        accepted_doc_ids, ignored_doc_ids = self._resolve_requested_doc_ids(requested_doc_ids, exposed_doc_ids)

        for doc_id in accepted_doc_ids:
            if doc_id not in target_list:
                target_list.append(doc_id)

        if accepted_doc_ids:
            trace.append(f"{action.kind}: {self._format_doc_ids(target_list)}")
        if ignored_doc_ids:
            trace.append(f"{action.kind} ignored unknown IDs: {self._format_doc_ids(ignored_doc_ids)}")
        if not accepted_doc_ids and not ignored_doc_ids:
            trace.append(f"{action.kind}: (none)")
        return True

    def _apply_scale_action(self, action: InteractRAGAction, state: InteractRAGState, trace: list[str]) -> bool:
        """Apply retrieval scale changes."""
        parsed_values = parse_doc_ids(action.text)
        parsed_scale = next((value for value in parsed_values if isinstance(value, int)), None)
        if parsed_scale is not None:
            state.current_scale = min(max(parsed_scale, 1), self.max_scale)
        trace.append(f"adjust_scale: {state.current_scale}")
        return True

    @staticmethod
    def _apply_weight_action(action: InteractRAGAction, state: InteractRAGState, trace: list[str]) -> bool:
        """Store weighted-fusion hint values for retrieval and metadata."""
        if action.semantic_weight is not None and action.exact_weight is not None:
            state.semantic_weight = action.semantic_weight
            state.exact_weight = action.exact_weight
            trace.append(
                f"weighted_fusion weights: semantic={state.semantic_weight:.2f} exact={state.exact_weight:.2f}"
            )
        return True

    def _apply_state_action(
        self,
        action: InteractRAGAction,
        state: InteractRAGState,
        trace: list[str],
        exposed_doc_ids: set[int | str],
    ) -> bool:
        """Apply non-search context-shaping actions. Return True if handled."""
        if action.kind in {"include_docs", "exclude_docs"}:
            return self._apply_doc_id_action(action, state, trace, exposed_doc_ids)
        if action.kind == "adjust_scale":
            return self._apply_scale_action(action, state, trace)
        if action.kind == "weighted_fusion" and self._exact_retrieval_pipeline is not None:
            return self._apply_weight_action(action, state, trace)
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

    def _append_evidence(
        self,
        evidence: list[dict[str, Any]],
        new_evidence: list[dict[str, Any]],
        state: InteractRAGState,
    ) -> list[dict[str, Any]]:
        """Append deduplicated evidence while respecting evidence_budget."""
        updated = list(evidence)
        existing_contents = {item.get("content") for item in updated}
        for item in new_evidence:
            content = item.get("content")
            if content and content not in existing_contents:
                updated.append(item)
                existing_contents.add(content)
        return self._trim_evidence_to_budget(updated, state)

    def _trim_evidence_to_budget(
        self,
        evidence: list[dict[str, Any]],
        state: InteractRAGState,
    ) -> list[dict[str, Any]]:
        """Trim evidence while preserving explicitly included chunks."""
        included_doc_ids = self._known_doc_ids(set(state.included_doc_ids))
        included_evidence = [item for item in evidence if item.get("doc_id") in included_doc_ids]
        if not included_evidence:
            return evidence[-self.evidence_budget :]

        remaining_budget = self.evidence_budget - len(included_evidence)
        if remaining_budget <= 0:
            return included_evidence

        non_included_evidence = [item for item in evidence if item.get("doc_id") not in included_doc_ids]
        return [*included_evidence, *non_included_evidence[-remaining_budget:]]

    @staticmethod
    def _is_prompt_simulated_action(action: InteractRAGAction) -> bool:
        """Return True when an action is represented as a query hint over the base retrieval boundary."""
        return action.kind in {"exact_search", "weighted_fusion", "entity_match"}

    @staticmethod
    def _is_retrieval_action(action: InteractRAGAction) -> bool:
        """Return True when an action triggers a retrieval engine call."""
        return action.kind in {"semantic_search", "exact_search", "weighted_fusion", "entity_match"}

    @staticmethod
    def _drop_excluded_docs(
        evidence: list[dict[str, Any]],
        retrieved_doc_ids: list[int | str],
        state: InteractRAGState,
    ) -> list[dict[str, Any]]:
        """Remove excluded chunks from the active evidence context and metadata."""
        excluded_doc_ids = set(state.excluded_doc_ids)
        retrieved_doc_ids[:] = [doc_id for doc_id in retrieved_doc_ids if doc_id not in excluded_doc_ids]
        return [item for item in evidence if item.get("doc_id") not in excluded_doc_ids]

    @staticmethod
    def _normalize_result_scores(results: list[dict[str, Any]]) -> dict[int | str, float]:
        """Return min-max normalized scores keyed by document ID."""
        normalized_scores = normalize_minmax([float(result.get("score") or 0.0) for result in results])
        return {
            result["doc_id"]: float(normalized_score or 0.0)
            for result, normalized_score in zip(results, normalized_scores, strict=False)
            if result.get("doc_id") is not None
        }

    async def _retrieve_for_action(self, action: InteractRAGAction, state: InteractRAGState) -> list[dict[str, Any]]:
        """Run the paper-correct retrieval primitive when an exact engine is configured."""
        if self._exact_retrieval_pipeline is None:
            retrieval_query = self._build_retrieval_query(action)
            return await self._retrieval_pipeline.retrieve(retrieval_query, state.current_scale)
        if action.kind == "exact_search":
            return await self._exact_retrieval_pipeline.retrieve(action.text, state.current_scale)
        if action.kind == "entity_match":
            entity_results = await self._exact_retrieval_pipeline.retrieve(action.text, state.current_scale)
            return entity_results[: min(3, state.current_scale)]
        if action.kind == "weighted_fusion":
            semantic_weight = action.semantic_weight if action.semantic_weight is not None else state.semantic_weight
            exact_weight = action.exact_weight if action.exact_weight is not None else state.exact_weight
            state.semantic_weight = semantic_weight
            state.exact_weight = exact_weight
            fusion_fetch_k = max(_FUSION_FETCH_K, state.current_scale)
            semantic_results = await self._retrieval_pipeline.retrieve(action.text, fusion_fetch_k)
            exact_results = await self._exact_retrieval_pipeline.retrieve(action.text, fusion_fetch_k)
            semantic_scores = self._normalize_result_scores(semantic_results)
            exact_scores = self._normalize_result_scores(exact_results)
            by_doc_id: dict[int | str, dict[str, Any]] = {}
            for result in [*semantic_results, *exact_results]:
                doc_id = result.get("doc_id")
                if doc_id is not None and doc_id not in by_doc_id:
                    by_doc_id[doc_id] = dict(result)
            fused_results: list[dict[str, Any]] = []
            for doc_id, result in by_doc_id.items():
                fused_result = dict(result)
                fused_result["score"] = semantic_weight * semantic_scores.get(
                    doc_id, 0.0
                ) + exact_weight * exact_scores.get(doc_id, 0.0)
                fused_results.append(fused_result)
            return sorted(fused_results, key=lambda result: float(result.get("score") or 0.0), reverse=True)[
                : state.current_scale
            ]
        return await self._retrieval_pipeline.retrieve(action.text, state.current_scale)

    async def _execute_retrieval_action(
        self,
        action: InteractRAGAction,
        state: InteractRAGState,
        trace: list[str],
        evidence: list[dict[str, Any]],
        retrieved_doc_ids: list[int | str],
        degraded_actions: list[str],
    ) -> list[dict[str, Any]]:
        """Run one retrieval-like action and update evidence."""
        if (
            self._exact_retrieval_pipeline is None
            and self._is_prompt_simulated_action(action)
            and action.kind not in degraded_actions
        ):
            degraded_actions.append(action.kind)
            trace.append(f"degraded {action.kind}: prompt-simulated via retrieval query")
        retrieval_results = await self._retrieve_for_action(action, state)
        filtered_results = self._filter_results(retrieval_results, state)
        new_evidence = self._evidence_from_results(filtered_results)
        doc_ids = [item["doc_id"] for item in new_evidence]
        for doc_id in doc_ids:
            if doc_id not in retrieved_doc_ids:
                retrieved_doc_ids.append(doc_id)
        trace.append(f"{action.kind}: {action.text}")
        return self._append_evidence(evidence, new_evidence, state)

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate an answer through INTERACT-RAG corpus interaction primitives."""
        query_text = self._service.get_query_text(query_id)
        state = InteractRAGState(current_scale=min(max(top_k, self.initial_scale), self.max_scale))
        tracker = TokenUsageTracker()
        trace: list[str] = []
        evidence: list[dict[str, Any]] = []
        retrieved_doc_ids: list[int | str] = []
        degraded_actions: list[str] = []
        final_answer = ""
        terminated_by = "max_steps"

        for step_index in range(self.max_steps):
            prompt = self._build_step_prompt(query_text, state, trace, evidence, step_index)
            response = await self._llm.ainvoke(prompt)
            tracker.record(response)
            actions = parse_interact_rag_actions(self._extract_text(response))
            answer_action = next((action for action in actions if action.kind == "answer"), None)
            for action in actions:
                if (
                    self._apply_state_action(action, state, trace, set(retrieved_doc_ids))
                    and action.kind == "exclude_docs"
                ):
                    evidence = self._drop_excluded_docs(evidence, retrieved_doc_ids, state)
            if answer_action is not None:
                skipped_retrievals = [action.kind for action in actions if self._is_retrieval_action(action)]
                if skipped_retrievals:
                    trace.append(
                        f"answer present; skipped same-step retrieval actions: {', '.join(skipped_retrievals)}"
                    )
                final_answer = answer_action.text
                trace.append(f"answer: {final_answer}")
                terminated_by = "answer"
                break
            for action in actions:
                if self._is_retrieval_action(action):
                    evidence = await self._execute_retrieval_action(
                        action,
                        state,
                        trace,
                        evidence,
                        retrieved_doc_ids,
                        degraded_actions,
                    )

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
                "evidence": [item.get("content", "") for item in evidence],
                "retrieved_chunk_ids": retrieved_doc_ids,
                "included_doc_ids": state.included_doc_ids,
                "excluded_doc_ids": state.excluded_doc_ids,
                "current_scale": state.current_scale,
                "semantic_weight": state.semantic_weight,
                "exact_weight": state.exact_weight,
                "retrieval_action_mode": "engine" if self._exact_retrieval_pipeline is not None else "prompt_simulated",
                "exact_engine": self._exact_retrieval_pipeline is not None,
                "degraded_actions": degraded_actions,
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
    "parse_interact_rag_actions",
]
