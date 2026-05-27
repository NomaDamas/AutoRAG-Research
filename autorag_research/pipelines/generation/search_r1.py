"""Search-R1-style inference pipeline for AutoRAG-Research.

Search-R1 is a training framework for reasoning-and-search interleaved agents.
This module implements the inference/evaluation slice that fits AutoRAG: an LLM
iteratively decides when to search, observes retrieved evidence from an existing
retrieval pipeline, and eventually emits an answer. RL training from the
upstream project is intentionally out of scope.
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
from autorag_research.util import TokenUsageTracker

logger = logging.getLogger("AutoRAG-Research")

DEFAULT_SEARCH_R1_STEP_PROMPT = """You are a Search-R1-style question answering agent.
You may interleave reasoning with retrieval. At each step, choose exactly one action:
- Search for more evidence by writing <search>your search query</search>
- Finish with an answer by writing <answer>your final answer</answer>

Question:
{query}

Search budget: {searches_used}/{max_searches} used, {remaining_searches} remaining.

Scratchpad and observations:
{scratchpad}

Next action:"""

DEFAULT_SEARCH_R1_FINAL_PROMPT = """Answer the question using the gathered Search-R1 evidence.

Question:
{query}

Scratchpad and observations:
{scratchpad}

Return only the final answer."""

_SEARCH_PATTERN = re.compile(r"<search>\s*(.*?)\s*</search>", re.IGNORECASE | re.DOTALL)
_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class SearchR1Action:
    """Parsed Search-R1 agent action."""

    kind: str
    text: str


def parse_search_r1_action(response_text: str) -> SearchR1Action:
    """Parse a Search-R1 action from an LLM response.

    ``<answer>`` is preferred over ``<search>`` when both are present because an
    explicit final answer should stop the loop.
    """
    answer_match = _ANSWER_PATTERN.search(response_text)
    if answer_match is not None:
        return SearchR1Action(kind="answer", text=answer_match.group(1).strip())

    search_match = _SEARCH_PATTERN.search(response_text)
    if search_match is not None:
        return SearchR1Action(kind="search", text=search_match.group(1).strip())

    stripped_response = response_text.strip()
    if stripped_response.lower().startswith("answer:"):
        return SearchR1Action(kind="answer", text=stripped_response.split(":", 1)[1].strip())

    return SearchR1Action(kind="answer", text=stripped_response)


@dataclass(kw_only=True)
class SearchR1GenerationPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for the inference-only Search-R1 generation pipeline."""

    step_prompt_template: str = field(default=DEFAULT_SEARCH_R1_STEP_PROMPT)
    final_prompt_template: str = field(default=DEFAULT_SEARCH_R1_FINAL_PROMPT)
    max_searches: int = 3
    k_per_search: int = 5
    observation_budget: int = 12
    fallback_to_final_prompt: bool = True

    def get_pipeline_class(self) -> type[SearchR1GenerationPipeline]:
        """Return the SearchR1GenerationPipeline class."""
        return SearchR1GenerationPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for SearchR1GenerationPipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)

        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "step_prompt_template": self.step_prompt_template,
            "final_prompt_template": self.final_prompt_template,
            "max_searches": self.max_searches,
            "k_per_search": self.k_per_search,
            "observation_budget": self.observation_budget,
            "fallback_to_final_prompt": self.fallback_to_final_prompt,
        }


class SearchR1GenerationPipeline(BaseGenerationPipeline):
    """Inference-time Search-R1-style reason/search/answer generation pipeline."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        step_prompt_template: str = DEFAULT_SEARCH_R1_STEP_PROMPT,
        final_prompt_template: str = DEFAULT_SEARCH_R1_FINAL_PROMPT,
        max_searches: int = 3,
        k_per_search: int = 5,
        observation_budget: int = 12,
        fallback_to_final_prompt: bool = True,
        schema: Any | None = None,
    ):
        """Initialize the Search-R1 inference pipeline."""
        self._validate_prompt_templates(step_prompt_template, final_prompt_template)
        if max_searches < 1:
            msg = "max_searches must be >= 1"
            raise ValueError(msg)
        if k_per_search < 1:
            msg = "k_per_search must be >= 1"
            raise ValueError(msg)
        if observation_budget < 1:
            msg = "observation_budget must be >= 1"
            raise ValueError(msg)

        self.step_prompt_template = step_prompt_template
        self.final_prompt_template = final_prompt_template
        self.max_searches = max_searches
        self.k_per_search = k_per_search
        self.observation_budget = observation_budget
        self.fallback_to_final_prompt = fallback_to_final_prompt

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    @staticmethod
    def _validate_prompt_templates(step_prompt_template: str, final_prompt_template: str) -> None:
        """Validate required prompt placeholders."""
        for placeholder in ("{query}", "{scratchpad}", "{max_searches}", "{remaining_searches}", "{searches_used}"):
            if placeholder not in step_prompt_template:
                msg = f"step_prompt_template must contain {placeholder}"
                raise ValueError(msg)
        for placeholder in ("{query}", "{scratchpad}"):
            if placeholder not in final_prompt_template:
                msg = f"final_prompt_template must contain {placeholder}"
                raise ValueError(msg)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return Search-R1 pipeline configuration."""
        model_name = getattr(self._llm, "model_name", None)
        if model_name is None or not isinstance(model_name, str):
            model_name = type(self._llm).__name__

        return {
            "type": "search_r1",
            "step_prompt_template": self.step_prompt_template,
            "final_prompt_template": self.final_prompt_template,
            "max_searches": self.max_searches,
            "k_per_search": self.k_per_search,
            "observation_budget": self.observation_budget,
            "fallback_to_final_prompt": self.fallback_to_final_prompt,
            "retrieval_pipeline_id": self._retrieval_pipeline.pipeline_id,
            "llm_model": model_name,
        }

    @staticmethod
    def _extract_response_content(response: Any) -> str:
        """Extract string content from a LangChain response."""
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)

    @staticmethod
    def _format_scratchpad(steps: list[str], observations: list[str]) -> str:
        """Format reasoning/search steps and observations for the next prompt."""
        sections: list[str] = []
        if steps:
            sections.append("Previous actions:\n" + "\n".join(steps))
        if observations:
            numbered_observations = "\n\n".join(
                f"[Observation {index + 1}] {observation}" for index, observation in enumerate(observations)
            )
            sections.append("Retrieved evidence:\n" + numbered_observations)
        return "\n\n".join(sections) if sections else "(empty)"

    def _build_step_prompt(
        self,
        query: str,
        steps: list[str],
        observations: list[str],
        searches_used: int,
    ) -> str:
        """Build one Search-R1 agent-step prompt."""
        remaining_searches = max(self.max_searches - searches_used, 0)
        return self.step_prompt_template.format(
            query=query,
            scratchpad=self._format_scratchpad(steps, observations),
            max_searches=self.max_searches,
            searches_used=searches_used,
            remaining_searches=remaining_searches,
        )

    def _build_final_prompt(self, query: str, steps: list[str], observations: list[str]) -> str:
        """Build fallback final-answer prompt after search budget exhaustion."""
        return self.final_prompt_template.format(
            query=query,
            scratchpad=self._format_scratchpad(steps, observations),
        )

    def _contents_from_results(self, retrieval_results: list[dict[str, Any]]) -> tuple[list[int | str], list[str]]:
        """Extract contents from retrieval results, backfilling from DB when needed."""
        chunk_ids: list[int | str] = []
        contents: list[str | None] = []
        missing_positions: list[int] = []
        missing_ids: list[int | str] = []

        for result in retrieval_results:
            doc_id = result.get("doc_id")
            if doc_id is None:
                continue
            chunk_ids.append(doc_id)
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

        return chunk_ids, [content or "" for content in contents]

    def _append_observations(
        self,
        observations: list[str],
        new_contents: list[str],
    ) -> list[str]:
        """Append non-empty, deduplicated observations within the observation budget."""
        updated_observations = list(observations)
        for content in new_contents:
            if content and content not in updated_observations:
                updated_observations.append(content)
        return updated_observations[-self.observation_budget :]

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate an answer with interleaved Search-R1-style search actions."""
        query_text = self._service.get_query_text(query_id)
        search_k = min(top_k, self.k_per_search) if top_k > 0 else self.k_per_search
        search_k = max(search_k, 1)
        tracker = TokenUsageTracker()
        steps: list[str] = []
        observations: list[str] = []
        retrieved_chunk_ids: list[int | str] = []
        search_queries: list[str] = []
        final_answer = ""
        terminated_by = "max_searches"

        for search_index in range(self.max_searches):
            prompt = self._build_step_prompt(query_text, steps, observations, search_index)
            response = await self._llm.ainvoke(prompt)
            tracker.record(response)
            response_text = self._extract_response_content(response)
            action = parse_search_r1_action(response_text)

            if action.kind == "answer":
                final_answer = action.text
                steps.append(f"Answer: {final_answer}")
                terminated_by = "answer"
                break

            if not action.text:
                logger.debug("Search-R1 emitted an empty search action; stopping with fallback answer")
                terminated_by = "empty_search"
                break

            search_queries.append(action.text)
            steps.append(f"Search: {action.text}")
            retrieval_results = await self._retrieval_pipeline.retrieve(action.text, search_k)
            chunk_ids, contents = self._contents_from_results(retrieval_results)
            for chunk_id in chunk_ids:
                if chunk_id not in retrieved_chunk_ids:
                    retrieved_chunk_ids.append(chunk_id)
            observations = self._append_observations(observations, contents)

        if not final_answer and self.fallback_to_final_prompt:
            final_prompt = self._build_final_prompt(query_text, steps, observations)
            final_response = await self._llm.ainvoke(final_prompt)
            tracker.record(final_response)
            final_answer = self._extract_response_content(final_response)
            terminated_by = f"{terminated_by}_fallback"

        return GenerationResult(
            text=final_answer,
            token_usage=tracker.total,
            metadata={
                "search_queries": search_queries,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "observations": observations,
                "steps": steps,
                "terminated_by": terminated_by,
            },
        )


__all__ = [
    "DEFAULT_SEARCH_R1_FINAL_PROMPT",
    "DEFAULT_SEARCH_R1_STEP_PROMPT",
    "SearchR1Action",
    "SearchR1GenerationPipeline",
    "SearchR1GenerationPipelineConfig",
    "parse_search_r1_action",
]
