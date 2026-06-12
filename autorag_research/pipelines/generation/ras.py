"""RAS (Retrieval-And-Structuring) generation pipeline.

This module implements the paper-faithful inference loop for RAS: plan before any
retrieval, optionally answer from parametric knowledge with [NO_RETRIEVAL],
iteratively retrieve planner-selected subqueries, extract triples into an evolving
question-specific graph, preserve per-subquery graph history, and answer from the
serialized graph plus history. Heavy trained GraphLLM/GNN components from the
paper are out of scope for this pipeline implementation.
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

RASPlanKind = Literal["retrieve", "sufficient", "no_retrieval", "invalid"]

DEFAULT_RAS_PLAN_PROMPT = """You are a Retrieval-And-Structuring (RAS) planner.
Decide the next paper-protocol action for answering the question from an evolving graph.
Return exactly one action token:
- [NO_RETRIEVAL] when the question should be answered directly from parametric knowledge.
- [SUBQ] standalone retrieval subquery when more evidence is needed.
- [SUFFICIENT] when the graph and history are sufficient to answer.

Question:
{query}

Step: {step}/{max_steps}

Current graph G_Q:
{graph}

Iteration history:
{history}

Next action:"""

DEFAULT_RAS_TRIPLE_PROMPT = """Extract question-relevant factual triples from the passage.
Return each triple as <triple>subject | predicate | object</triple>.
If no useful triple exists, return <none/>.

Question:
{query}

Passage:
{passage}

Triples:"""

DEFAULT_RAS_ANSWER_PROMPT = """Answer the question using only the structured graph and subquery history.

Question:
{query}

Structured graph G_Q:
{graph}

Subquery-to-triples history:
{history}

Answer:"""

DEFAULT_RAS_NO_RETRIEVAL_PROMPT = """Answer the question directly from parametric knowledge. Do not cite retrieved evidence.

Question:
{query}

Answer:"""

_SUBQ_TOKEN_RE = re.compile(r"\[\s*SUBQ\s*\]\s*(.*)", re.IGNORECASE | re.DOTALL)
_NO_RETRIEVAL_TOKEN_RE = re.compile(r"\[\s*NO_RETRIEVAL\s*\]", re.IGNORECASE)
_SUFFICIENT_TOKEN_RE = re.compile(r"\[\s*SUFFICIENT\s*\]", re.IGNORECASE)
_TRIPLE_RE = re.compile(r"<triple>\s*(.*?)\s*</triple>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class RASPlanAction:
    """Parsed RAS planning action."""

    kind: RASPlanKind
    text: str


@dataclass(frozen=True)
class RASTriple:
    """Serializable question-specific graph triple."""

    subject: str
    predicate: str
    object: str

    def as_tuple(self) -> tuple[str, str, str]:
        """Return a tuple representation."""
        return (self.subject, self.predicate, self.object)

    def as_text(self) -> str:
        """Return a prompt-friendly text representation."""
        return f"({self.subject}) -[{self.predicate}]-> ({self.object})"


@dataclass(frozen=True)
class RASIterationRecord:
    """Serializable per-subquery RAS history record."""

    subquery: str
    triples: tuple[RASTriple, ...]
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        payload: dict[str, Any] = {
            "subquery": self.subquery,
            "triples": [triple.as_tuple() for triple in self.triples],
        }
        if self.note is not None:
            payload["note"] = self.note
        return payload


def parse_ras_plan_action(response_text: str) -> RASPlanAction:
    """Parse paper-token RAS planner output."""
    stripped_response = response_text.strip()

    subquery_match = _SUBQ_TOKEN_RE.search(response_text)
    if subquery_match is not None:
        text = subquery_match.group(1).strip()
        if text:
            return RASPlanAction(kind="retrieve", text=text)
        return RASPlanAction(kind="invalid", text=stripped_response)

    if _NO_RETRIEVAL_TOKEN_RE.search(response_text) is not None:
        return RASPlanAction(kind="no_retrieval", text="")

    if _SUFFICIENT_TOKEN_RE.search(response_text) is not None:
        return RASPlanAction(kind="sufficient", text="")

    return RASPlanAction(kind="invalid", text=stripped_response)


def parse_ras_triples(response_text: str) -> list[RASTriple]:
    """Parse triples from extractor output."""
    triples: list[RASTriple] = []
    for match in _TRIPLE_RE.finditer(response_text):
        parts = [part.strip() for part in match.group(1).split("|")]
        if len(parts) != 3 or not all(parts):
            continue
        triple = RASTriple(subject=parts[0], predicate=parts[1], object=parts[2])
        if triple not in triples:
            triples.append(triple)
    return triples


@dataclass(kw_only=True)
class RASGenerationPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for RASGenerationPipeline."""

    plan_prompt_template: str = field(default=DEFAULT_RAS_PLAN_PROMPT)
    triple_prompt_template: str = field(default=DEFAULT_RAS_TRIPLE_PROMPT)
    answer_prompt_template: str = field(default=DEFAULT_RAS_ANSWER_PROMPT)
    no_retrieval_prompt_template: str = field(default=DEFAULT_RAS_NO_RETRIEVAL_PROMPT)
    max_steps: int = 3
    k_per_step: int = 4
    evidence_budget: int = 12
    triple_budget: int = 40

    def get_pipeline_class(self) -> type[RASGenerationPipeline]:
        """Return the RASGenerationPipeline class."""
        return RASGenerationPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for RASGenerationPipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)
        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "plan_prompt_template": self.plan_prompt_template,
            "triple_prompt_template": self.triple_prompt_template,
            "answer_prompt_template": self.answer_prompt_template,
            "no_retrieval_prompt_template": self.no_retrieval_prompt_template,
            "max_steps": self.max_steps,
            "k_per_step": self.k_per_step,
            "evidence_budget": self.evidence_budget,
            "triple_budget": self.triple_budget,
        }


class RASGenerationPipeline(BaseGenerationPipeline):
    """Retrieval-And-Structuring generation pipeline."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        plan_prompt_template: str = DEFAULT_RAS_PLAN_PROMPT,
        triple_prompt_template: str = DEFAULT_RAS_TRIPLE_PROMPT,
        answer_prompt_template: str = DEFAULT_RAS_ANSWER_PROMPT,
        no_retrieval_prompt_template: str = DEFAULT_RAS_NO_RETRIEVAL_PROMPT,
        max_steps: int = 3,
        k_per_step: int = 4,
        evidence_budget: int = 12,
        triple_budget: int = 40,
        schema: Any | None = None,
    ):
        """Initialize RASGenerationPipeline."""
        if max_steps < 1:
            msg = "max_steps must be >= 1"
            raise ValueError(msg)
        if k_per_step < 1:
            msg = "k_per_step must be >= 1"
            raise ValueError(msg)
        if evidence_budget < 1 or triple_budget < 1:
            msg = "evidence_budget and triple_budget must be >= 1"
            raise ValueError(msg)

        self.plan_prompt_template = plan_prompt_template
        self.triple_prompt_template = triple_prompt_template
        self.answer_prompt_template = answer_prompt_template
        self.no_retrieval_prompt_template = no_retrieval_prompt_template
        self.max_steps = max_steps
        self.k_per_step = k_per_step
        self.evidence_budget = evidence_budget
        self.triple_budget = triple_budget

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return RAS configuration for storage."""
        model_name = getattr(self._llm, "model_name", None)
        if model_name is None or not isinstance(model_name, str):
            model_name = type(self._llm).__name__
        return {
            "type": "ras",
            "plan_prompt_template": self.plan_prompt_template,
            "triple_prompt_template": self.triple_prompt_template,
            "answer_prompt_template": self.answer_prompt_template,
            "no_retrieval_prompt_template": self.no_retrieval_prompt_template,
            "max_steps": self.max_steps,
            "k_per_step": self.k_per_step,
            "evidence_budget": self.evidence_budget,
            "triple_budget": self.triple_budget,
            "retrieval_pipeline_id": self._retrieval_pipeline.pipeline_id,
            "llm_model": model_name,
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text from a LangChain response."""
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def _format_graph(triples: list[RASTriple]) -> str:
        """Format graph triples for prompts."""
        if not triples:
            return "(empty)"
        return "\n".join(f"[{index + 1}] {triple.as_text()}" for index, triple in enumerate(triples))

    @staticmethod
    def _format_iteration_history(iteration_history: list[RASIterationRecord]) -> str:
        """Format per-subquery graph history for prompts."""
        if not iteration_history:
            return "(none)"
        blocks: list[str] = []
        for index, record in enumerate(iteration_history, start=1):
            if record.triples:
                triple_lines = "\n".join(f"  - {triple.as_text()}" for triple in record.triples)
            else:
                triple_lines = "  - (no new triples)"
            note = f"\n  note: {record.note}" if record.note else ""
            blocks.append(f"[{index}] subquery: {record.subquery}\n{triple_lines}{note}")
        return "\n".join(blocks)

    def _build_plan_prompt(
        self,
        query: str,
        graph: list[RASTriple],
        iteration_history: list[RASIterationRecord],
        step: int,
        corrective_suffix: str = "",
    ) -> str:
        """Build a retrieval planning prompt."""
        prompt = self.plan_prompt_template.format(
            query=query,
            graph=self._format_graph(graph),
            history=self._format_iteration_history(iteration_history),
            step=step,
            max_steps=self.max_steps,
        )
        if corrective_suffix:
            prompt = f"{prompt}\n\n{corrective_suffix}"
        return prompt

    def _build_triple_prompt(self, query: str, passage: str) -> str:
        """Build a triple extraction prompt."""
        return self.triple_prompt_template.format(query=query, passage=passage)

    def _build_answer_prompt(
        self, query: str, graph: list[RASTriple], iteration_history: list[RASIterationRecord]
    ) -> str:
        """Build final answer prompt."""
        return self.answer_prompt_template.format(
            query=query,
            graph=self._format_graph(graph),
            history=self._format_iteration_history(iteration_history),
        )

    def _build_no_retrieval_answer_prompt(self, query: str) -> str:
        """Build no-retrieval answer prompt."""
        return self.no_retrieval_prompt_template.format(query=query)

    def _contents_from_results(self, results: list[dict[str, Any]]) -> tuple[list[int | str], list[str]]:
        """Extract IDs and passage text from retrieval results."""
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

    def _append_evidence(self, evidence: list[str], passages: list[str]) -> list[str]:
        """Append deduplicated passages within evidence budget."""
        updated = list(evidence)
        for passage in passages:
            if passage and passage not in updated:
                updated.append(passage)
        return updated[-self.evidence_budget :]

    def _merge_triples(self, triples: list[RASTriple], new_triples: list[RASTriple]) -> list[RASTriple]:
        """Merge deduplicated triples within triple budget."""
        merged = list(triples)
        for triple in new_triples:
            if triple not in merged:
                merged.append(triple)
        return merged[-self.triple_budget :]

    def _split_seen_and_new_passages(
        self,
        results: list[dict[str, Any]],
        retrieved_chunk_ids: list[int | str],
        extracted_chunk_ids: set[int | str],
    ) -> tuple[list[str], list[str]]:
        """Record retrieved chunks and return all passages plus extraction-ready new passages."""
        doc_ids, passages = self._contents_from_results(results)
        new_passages: list[str] = []
        for doc_id, passage in zip(doc_ids, passages, strict=False):
            if doc_id not in retrieved_chunk_ids:
                retrieved_chunk_ids.append(doc_id)
            if doc_id not in extracted_chunk_ids:
                extracted_chunk_ids.add(doc_id)
                new_passages.append(passage)
        return passages, new_passages

    async def _extract_triples_for_passages(
        self,
        query: str,
        passages: list[str],
        tracker: TokenUsageTracker,
    ) -> list[RASTriple]:
        """Extract triples from retrieved passages."""
        extracted: list[RASTriple] = []
        for passage in passages:
            prompt = self._build_triple_prompt(query, passage)
            response = await self._llm.ainvoke(prompt)
            tracker.record(response)
            extracted = self._merge_triples(extracted, parse_ras_triples(self._extract_text(response)))
        return extracted

    async def _plan_with_retry(
        self,
        query_text: str,
        graph: list[RASTriple],
        iteration_history: list[RASIterationRecord],
        step: int,
        tracker: TokenUsageTracker,
        plan_trace: list[str],
        malformed_plans: list[str],
    ) -> RASPlanAction:
        """Invoke planner with one bounded retry for malformed output."""
        plan_prompt = self._build_plan_prompt(query_text, graph, iteration_history, step)
        plan_response = await self._llm.ainvoke(plan_prompt)
        tracker.record(plan_response)
        response_text = self._extract_text(plan_response)
        action = parse_ras_plan_action(response_text)
        plan_trace.append(f"{action.kind}: {action.text}".rstrip())
        if action.kind != "invalid":
            return action

        malformed_plans.append(action.text)
        retry_prompt = self._build_plan_prompt(
            query_text,
            graph,
            iteration_history,
            step,
            "Your previous action was malformed. Return exactly one of [NO_RETRIEVAL], [SUBQ] <query>, or "
            "[SUFFICIENT]. [SUBQ] must include a non-empty query.",
        )
        retry_response = await self._llm.ainvoke(retry_prompt)
        tracker.record(retry_response)
        retry_text = self._extract_text(retry_response)
        retry_action = parse_ras_plan_action(retry_text)
        plan_trace.append(f"retry {retry_action.kind}: {retry_action.text}".rstrip())
        if retry_action.kind == "invalid":
            malformed_plans.append(retry_action.text)
            return RASPlanAction(kind="sufficient", text="malformed planner output")
        return retry_action

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate an answer using plan-first retrieval-and-structuring."""
        query_text = self._service.get_query_text(query_id)
        retrieval_k = max(1, min(top_k, self.k_per_step)) if top_k > 0 else self.k_per_step
        tracker = TokenUsageTracker()
        evidence: list[str] = []
        graph: list[RASTriple] = []
        iteration_history: list[RASIterationRecord] = []
        subqueries: list[str] = []
        retrieved_chunk_ids: list[int | str] = []
        plan_trace: list[str] = []
        malformed_plans: list[str] = []
        route = "graph"
        extracted_chunk_ids: set[int | str] = set()

        for step in range(1, self.max_steps + 1):
            action = await self._plan_with_retry(
                query_text,
                graph,
                iteration_history,
                step,
                tracker,
                plan_trace,
                malformed_plans,
            )
            if action.kind == "no_retrieval" and step == 1:
                route = "no_retrieval"
                answer_prompt = self._build_no_retrieval_answer_prompt(query_text)
                answer_response = await self._llm.ainvoke(answer_prompt)
                tracker.record(answer_response)
                return GenerationResult(
                    text=self._extract_text(answer_response),
                    token_usage=tracker.total,
                    metadata={
                        "route": route,
                        "subqueries": subqueries,
                        "retrieved_chunk_ids": retrieved_chunk_ids,
                        "triples": [triple.as_tuple() for triple in graph],
                        "triple_texts": [triple.as_text() for triple in graph],
                        "iteration_history": [record.as_dict() for record in iteration_history],
                        "plan_trace": plan_trace,
                        "malformed_plans": malformed_plans,
                        "evidence": evidence,
                    },
                )
            if action.kind in {"sufficient", "no_retrieval"}:
                break

            subqueries.append(action.text)
            results = await self._retrieval_pipeline.retrieve(action.text, retrieval_k)
            passages, new_passages = self._split_seen_and_new_passages(
                results,
                retrieved_chunk_ids,
                extracted_chunk_ids,
            )
            evidence = self._append_evidence(evidence, passages)
            if not new_passages:
                iteration_history.append(RASIterationRecord(subquery=action.text, triples=(), note="no new passages"))
                plan_trace.append(f"note: no new passages for {action.text}")
                continue

            new_triples = await self._extract_triples_for_passages(query_text, new_passages, tracker)
            graph = self._merge_triples(graph, new_triples)
            iteration_history.append(RASIterationRecord(subquery=action.text, triples=tuple(new_triples)))

        answer_prompt = self._build_answer_prompt(query_text, graph, iteration_history)
        answer_response = await self._llm.ainvoke(answer_prompt)
        tracker.record(answer_response)

        return GenerationResult(
            text=self._extract_text(answer_response),
            token_usage=tracker.total,
            metadata={
                "route": route,
                "subqueries": subqueries,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "triples": [triple.as_tuple() for triple in graph],
                "triple_texts": [triple.as_text() for triple in graph],
                "iteration_history": [record.as_dict() for record in iteration_history],
                "plan_trace": plan_trace,
                "malformed_plans": malformed_plans,
                "evidence": evidence,
            },
        )


__all__ = [
    "DEFAULT_RAS_ANSWER_PROMPT",
    "DEFAULT_RAS_NO_RETRIEVAL_PROMPT",
    "DEFAULT_RAS_PLAN_PROMPT",
    "DEFAULT_RAS_TRIPLE_PROMPT",
    "RASGenerationPipeline",
    "RASGenerationPipelineConfig",
    "RASPlanAction",
    "RASTriple",
    "parse_ras_plan_action",
    "parse_ras_triples",
]
