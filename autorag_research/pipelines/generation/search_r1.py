"""Search-R1 paper-faithful inference pipeline for AutoRAG-Research.

Search-R1 rolls out one string ``y`` after a paper prompt that requires
``<think>`` before each action. Every LLM segment is appended verbatim only
through the earliest completed ``<search>`` or ``<answer>`` action. Search
results are inserted by the environment inside ``<information>`` tags, and
malformed segments append the paper rethink message before consuming another
action budget step. RL training, PPO/GRPO, and retrieved-token masking from
the upstream project are intentionally out of scope. ``fallback_to_final_prompt``
is a non-paper compatibility option and defaults to disabled.
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

DEFAULT_SEARCH_R1_STEP_PROMPT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get "
    "new information. After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between <information> and "
    "</information>. You can search as many times as you want. If you find no further external knowledge needed, "
    "you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. Question: "
    "{query}"
)

DEFAULT_SEARCH_R1_FINAL_PROMPT = """Answer the question from the Search-R1 rollout.

Question:
{query}

Rollout:
{rollout}

Return only the final answer."""

_SEARCH_PATTERN = re.compile(r"<search>\s*(.*?)\s*</search>", re.IGNORECASE | re.DOTALL)
_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class SearchR1Action:
    """Parsed Search-R1 agent action."""

    kind: str
    text: str
    end_index: int | None = None


def parse_search_r1_action(response_text: str) -> SearchR1Action:
    """Parse the earliest completed Search-R1 action from an LLM segment."""
    search_match = _SEARCH_PATTERN.search(response_text)
    answer_match = _ANSWER_PATTERN.search(response_text)
    matches = [match for match in (search_match, answer_match) if match is not None]
    if not matches:
        return SearchR1Action(kind="malformed", text="", end_index=None)

    first_match = min(matches, key=lambda match: match.end())
    kind = "search" if first_match.re is _SEARCH_PATTERN else "answer"
    return SearchR1Action(kind=kind, text=first_match.group(1).strip(), end_index=first_match.end())


@dataclass(kw_only=True)
class SearchR1GenerationPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for the inference-only Search-R1 generation pipeline."""

    step_prompt_template: str = field(default=DEFAULT_SEARCH_R1_STEP_PROMPT)
    final_prompt_template: str = field(default=DEFAULT_SEARCH_R1_FINAL_PROMPT)
    max_actions: int = 4
    k_per_search: int = 3
    fallback_to_final_prompt: bool = False

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
            "max_actions": self.max_actions,
            "k_per_search": self.k_per_search,
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
        max_actions: int = 4,
        k_per_search: int = 3,
        fallback_to_final_prompt: bool = False,
        schema: Any | None = None,
    ):
        """Initialize the Search-R1 inference pipeline."""
        self._validate_prompt_templates(step_prompt_template, final_prompt_template)
        if max_actions < 1:
            msg = "max_actions must be >= 1"
            raise ValueError(msg)
        if k_per_search < 1:
            msg = "k_per_search must be >= 1"
            raise ValueError(msg)

        self.step_prompt_template = step_prompt_template
        self.final_prompt_template = final_prompt_template
        self.max_actions = max_actions
        self.k_per_search = k_per_search
        self.fallback_to_final_prompt = fallback_to_final_prompt

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    @staticmethod
    def _validate_prompt_templates(step_prompt_template: str, final_prompt_template: str) -> None:
        """Validate required prompt placeholders."""
        if "{query}" not in step_prompt_template:
            msg = "step_prompt_template must contain {query}"
            raise ValueError(msg)
        for placeholder in ("{query}", "{rollout}"):
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
            "max_actions": self.max_actions,
            "k_per_search": self.k_per_search,
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

    def _build_step_prompt(self, query: str, rollout: str) -> str:
        """Build one Search-R1 rollout prompt."""
        return self.step_prompt_template.format(query=query) + rollout

    def _build_final_prompt(self, query: str, rollout: str) -> str:
        """Build non-paper fallback final-answer prompt after action budget exhaustion."""
        return self.final_prompt_template.format(query=query, rollout=rollout)

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

    @staticmethod
    def _format_information(contents: list[str]) -> str:
        """Format retrieved contents as a Search-R1 environment insertion."""
        joined_contents = "\n".join(content for content in contents if content)
        return f"<information>{joined_contents}</information>"

    @staticmethod
    def _extract_last_answer(rollout: str) -> str:
        """Extract the last completed answer in a rollout, if any."""
        matches = list(_ANSWER_PATTERN.finditer(rollout))
        if not matches:
            return ""
        return matches[-1].group(1).strip()

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate an answer with interleaved Search-R1-style search actions."""
        query_text = self._service.get_query_text(query_id)
        search_k = min(top_k, self.k_per_search) if top_k > 0 else self.k_per_search
        search_k = max(search_k, 1)
        tracker = TokenUsageTracker()
        rollout = ""
        retrieved_chunk_ids: list[int | str] = []
        search_queries: list[str] = []
        final_answer = ""
        terminated_by = "budget_exhausted"
        actions_used = 0

        for _ in range(self.max_actions):
            prompt = self._build_step_prompt(query_text, rollout)
            response = await self._llm.ainvoke(prompt)
            tracker.record(response)
            response_text = self._extract_response_content(response)
            action = parse_search_r1_action(response_text)
            actions_used += 1

            if action.kind == "malformed" or action.end_index is None:
                rollout += response_text + "My action is not correct. Let me rethink."
                continue

            segment = response_text[: action.end_index]
            rollout += segment

            if action.kind == "answer":
                final_answer = action.text
                terminated_by = "answer"
                break

            if not action.text:
                logger.debug("Search-R1 emitted an empty search action; treating it as malformed")
                rollout += "My action is not correct. Let me rethink."
                continue

            search_queries.append(action.text)
            retrieval_results = await self._retrieval_pipeline.retrieve(action.text, search_k)
            chunk_ids, contents = self._contents_from_results(retrieval_results)
            for chunk_id in chunk_ids:
                if chunk_id not in retrieved_chunk_ids:
                    retrieved_chunk_ids.append(chunk_id)
            rollout += self._format_information(contents)

        if not final_answer:
            final_answer = self._extract_last_answer(rollout)
        if not final_answer and self.fallback_to_final_prompt:
            final_prompt = self._build_final_prompt(query_text, rollout)
            final_response = await self._llm.ainvoke(final_prompt)
            tracker.record(final_response)
            final_answer = self._extract_response_content(final_response)
            terminated_by = "budget_exhausted_fallback"

        return GenerationResult(
            text=final_answer,
            token_usage=tracker.total,
            metadata={
                "rollout": rollout,
                "search_queries": search_queries,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "actions_used": actions_used,
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
