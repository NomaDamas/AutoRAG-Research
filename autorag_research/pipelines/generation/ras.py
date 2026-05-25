"""RAS (Retrieval-And-Structuring) generation pipeline.

This module implements an AutoRAG-compatible, inference-time RAS baseline: plan
retrieval needs, retrieve passages, extract triples into a question-specific
structured graph, and answer from that graph plus supporting evidence. Heavy
GraphLLM/GNN training components from the paper are out of scope for this
pipeline implementation.
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

RASPlanKind = Literal["retrieve", "sufficient"]

DEFAULT_RAS_PLAN_PROMPT = """You are a Retrieval-And-Structuring (RAS) planner.
Decide whether more retrieval is needed before building the structured answer graph.
Return exactly one action:
- <subquery>standalone retrieval query</subquery>
- <sufficient>brief reason evidence is sufficient</sufficient>

Question:
{query}

Step: {step}/{max_steps}
Current triples:
{triples}

Evidence:
{evidence}

Next action:"""

DEFAULT_RAS_TRIPLE_PROMPT = """Extract question-relevant factual triples from the passage.
Return each triple as <triple>subject | predicate | object</triple>.
If no useful triple exists, return <none/>.

Question:
{query}

Passage:
{passage}

Triples:"""

DEFAULT_RAS_ANSWER_PROMPT = """Answer the question using the structured triples and supporting evidence.

Question:
{query}

Structured triples:
{triples}

Supporting evidence:
{evidence}

Answer:"""

_SUBQUERY_RE = re.compile(r"<subquery>\s*(.*?)\s*</subquery>", re.IGNORECASE | re.DOTALL)
_SUFFICIENT_RE = re.compile(r"<sufficient>\s*(.*?)\s*</sufficient>", re.IGNORECASE | re.DOTALL)
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


def parse_ras_plan_action(response_text: str) -> RASPlanAction:
    """Parse RAS planner output."""
    sufficient_match = _SUFFICIENT_RE.search(response_text)
    if sufficient_match is not None:
        return RASPlanAction(kind="sufficient", text=sufficient_match.group(1).strip())

    subquery_match = _SUBQUERY_RE.search(response_text)
    if subquery_match is not None:
        return RASPlanAction(kind="retrieve", text=subquery_match.group(1).strip())

    stripped_response = response_text.strip()
    if stripped_response.lower().startswith("subquery:"):
        return RASPlanAction(kind="retrieve", text=stripped_response.split(":", 1)[1].strip())
    return RASPlanAction(kind="sufficient", text=stripped_response)


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
    def _format_evidence(evidence: list[str]) -> str:
        """Format evidence for prompts."""
        if not evidence:
            return "(none)"
        return "\n\n".join(f"[{index + 1}] {item}" for index, item in enumerate(evidence))

    @staticmethod
    def _format_triples(triples: list[RASTriple]) -> str:
        """Format triples for prompts."""
        if not triples:
            return "(none)"
        return "\n".join(f"[{index + 1}] {triple.as_text()}" for index, triple in enumerate(triples))

    def _build_plan_prompt(self, query: str, evidence: list[str], triples: list[RASTriple], step: int) -> str:
        """Build a retrieval planning prompt."""
        return self.plan_prompt_template.format(
            query=query,
            evidence=self._format_evidence(evidence),
            triples=self._format_triples(triples),
            step=step,
            max_steps=self.max_steps,
        )

    def _build_triple_prompt(self, query: str, passage: str) -> str:
        """Build a triple extraction prompt."""
        return self.triple_prompt_template.format(query=query, passage=passage)

    def _build_answer_prompt(self, query: str, evidence: list[str], triples: list[RASTriple]) -> str:
        """Build final answer prompt."""
        return self.answer_prompt_template.format(
            query=query,
            evidence=self._format_evidence(evidence),
            triples=self._format_triples(triples),
        )

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

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate an answer using retrieval-and-structuring."""
        query_text = self._service.get_query_text(query_id)
        retrieval_k = max(1, min(top_k, self.k_per_step)) if top_k > 0 else self.k_per_step
        tracker = TokenUsageTracker()
        evidence: list[str] = []
        triples: list[RASTriple] = []
        subqueries: list[str] = []
        retrieved_chunk_ids: list[int | str] = []
        plan_trace: list[str] = []

        for step in range(1, self.max_steps + 1):
            plan_prompt = self._build_plan_prompt(query_text, evidence, triples, step)
            plan_response = await self._llm.ainvoke(plan_prompt)
            tracker.record(plan_response)
            action = parse_ras_plan_action(self._extract_text(plan_response))
            plan_trace.append(f"{action.kind}: {action.text}")
            if action.kind == "sufficient":
                break

            subqueries.append(action.text)
            results = await self._retrieval_pipeline.retrieve(action.text, retrieval_k)
            doc_ids, passages = self._contents_from_results(results)
            for doc_id in doc_ids:
                if doc_id not in retrieved_chunk_ids:
                    retrieved_chunk_ids.append(doc_id)
            evidence = self._append_evidence(evidence, passages)
            triples = self._merge_triples(
                triples,
                await self._extract_triples_for_passages(query_text, passages, tracker),
            )

        answer_prompt = self._build_answer_prompt(query_text, evidence, triples)
        answer_response = await self._llm.ainvoke(answer_prompt)
        tracker.record(answer_response)
        answer_text = self._extract_text(answer_response)

        return GenerationResult(
            text=answer_text,
            token_usage=tracker.total,
            metadata={
                "subqueries": subqueries,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "triples": [triple.as_tuple() for triple in triples],
                "triple_texts": [triple.as_text() for triple in triples],
                "evidence": evidence,
                "plan_trace": plan_trace,
            },
        )


__all__ = [
    "DEFAULT_RAS_ANSWER_PROMPT",
    "DEFAULT_RAS_PLAN_PROMPT",
    "DEFAULT_RAS_TRIPLE_PROMPT",
    "RASGenerationPipeline",
    "RASGenerationPipelineConfig",
    "RASPlanAction",
    "RASTriple",
    "parse_ras_plan_action",
    "parse_ras_triples",
]
