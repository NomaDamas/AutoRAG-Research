"""Self-RAG generation pipeline for AutoRAG-Research.

This implementation is a prompt-based approximation of Self-RAG. It does not
require a fine-tuned reflection-token model. Instead, it uses iterative
self-reflection prompts to decide whether retrieval is necessary, revise the
current answer with retrieved evidence, and stop once the answer appears
supported.
"""

import json
import logging
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

DEFAULT_INITIAL_PROMPT = """You are answering a question without external evidence.

Question: {query}

Provide your best concise answer. If you are uncertain, answer briefly and leave room for later revision.

Answer:"""

DEFAULT_REFLECTION_PROMPT = """You are a Self-RAG controller deciding whether an answer needs retrieval or revision.

Question: {query}

Current Answer:
{answer}

Retrieved Context:
{context}

Critique History:
{critique_history}

Respond as JSON with this schema:
{{
  "should_retrieve": true or false,
  "is_supported": true or false,
  "follow_up_query": "optional refined retrieval query",
  "critique": "short critique of the current answer"
}}
"""

DEFAULT_REVISION_PROMPT = """Revise the answer so it is grounded in the retrieved context.

Question: {query}

Current Answer:
{answer}

Retrieved Context:
{context}

Critique:
{critique}

Return only the revised answer.
"""


class SelfRAGPipeline(BaseGenerationPipeline):
    """Prompt-based Self-RAG generation pipeline.

    The pipeline follows a lightweight self-reflection loop:
    1. Draft an initial answer without retrieval.
    2. Reflect on whether retrieval or revision is needed.
    3. Retrieve evidence only when requested by reflection.
    4. Revise the answer with the current evidence.
    5. Stop once reflection marks the answer as supported or the step budget is exhausted.
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        initial_prompt_template: str = DEFAULT_INITIAL_PROMPT,
        reflection_prompt_template: str = DEFAULT_REFLECTION_PROMPT,
        revision_prompt_template: str = DEFAULT_REVISION_PROMPT,
        max_reflection_steps: int = 3,
        schema: Any | None = None,
    ) -> None:
        self._initial_prompt_template = initial_prompt_template
        self._reflection_prompt_template = reflection_prompt_template
        self._revision_prompt_template = revision_prompt_template
        self.max_reflection_steps = max_reflection_steps
        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        model_name = getattr(self._llm, "model_name", None)
        if model_name is None or not isinstance(model_name, str):
            model_name = type(self._llm).__name__

        return {
            "type": "self_rag",
            "initial_prompt_template": self._initial_prompt_template,
            "reflection_prompt_template": self._reflection_prompt_template,
            "revision_prompt_template": self._revision_prompt_template,
            "max_reflection_steps": self.max_reflection_steps,
            "retrieval_pipeline_id": self._retrieval_pipeline.pipeline_id,
            "llm_model": model_name,
        }

    def _format_context(self, chunk_contents: list[str]) -> str:
        if not chunk_contents:
            return "(No retrieved context yet)"
        return "\n\n".join(f"[{index}] {content}" for index, content in enumerate(chunk_contents, start=1))

    def _format_critique_history(self, critiques: list[str]) -> str:
        if not critiques:
            return "(No previous critiques)"
        return "\n".join(f"[{index}] {critique}" for index, critique in enumerate(critiques, start=1))

    def _build_initial_prompt(self, query: str) -> str:
        return self._initial_prompt_template.format(query=query)

    def _build_reflection_prompt(
        self,
        query: str,
        answer: str,
        chunk_contents: list[str],
        critiques: list[str],
    ) -> str:
        return self._reflection_prompt_template.format(
            query=query,
            answer=answer,
            context=self._format_context(chunk_contents),
            critique_history=self._format_critique_history(critiques),
        )

    def _build_revision_prompt(
        self,
        query: str,
        answer: str,
        chunk_contents: list[str],
        critique: str,
    ) -> str:
        return self._revision_prompt_template.format(
            query=query,
            answer=answer,
            context=self._format_context(chunk_contents),
            critique=critique or "Improve grounding and specificity.",
        )

    def _coerce_reflection_flag(self, value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1", "supported"}:
                return True
            if normalized in {"false", "no", "0", "unsupported", ""}:
                return False
            return default
        if value is None:
            return default
        return bool(value)

    def _parse_key_value_reflection(self, response_text: str) -> dict[str, Any]:
        parsed: dict[str, Any] = {
            "action": "FINISH",
            "supported": False,
            "search_query": "",
            "critique": response_text.strip(),
        }

        for line in response_text.splitlines():
            if ":" not in line:
                continue
            raw_key, raw_value = line.split(":", 1)
            key = raw_key.strip().upper()
            value = raw_value.strip()

            if key == "ACTION":
                normalized_action = value.upper()
                if normalized_action in {"FINISH", "RETRIEVE", "REVISE"}:
                    parsed["action"] = normalized_action
            elif key == "SUPPORTED":
                parsed["supported"] = self._coerce_reflection_flag(value)
            elif key in {"SEARCH_QUERY", "FOLLOW_UP_QUERY"}:
                parsed["search_query"] = value
            elif key == "CRITIQUE" and value:
                parsed["critique"] = value

        return parsed

    def _parse_reflection(self, response_text: str) -> dict[str, Any]:
        stripped = response_text.strip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
                should_retrieve = self._coerce_reflection_flag(payload.get("should_retrieve", False))
                supported = self._coerce_reflection_flag(payload.get("is_supported", False))
                if should_retrieve:
                    action = "RETRIEVE"
                elif supported:
                    action = "FINISH"
                else:
                    action = "REVISE"
                return {
                    "action": action,
                    "supported": supported,
                    "search_query": payload.get("follow_up_query", "") or payload.get("search_query", ""),
                    "critique": payload.get("critique", stripped),
                }
            except json.JSONDecodeError:
                logger.warning("Failed to parse Self-RAG reflection JSON; falling back to key-value parsing")

        return self._parse_key_value_reflection(stripped)

    async def _ainvoke_text(self, prompt: str, tracker: TokenUsageTracker) -> str:
        response = await self._llm.ainvoke(prompt)
        tracker.record(response)
        return response.content if hasattr(response, "content") else str(response)

    def _extend_context(
        self,
        retrieved_results: list[dict[str, Any]],
        retrieved_chunk_ids: list[int | str],
        retrieved_scores: list[float | int],
        chunk_contents: list[str],
    ) -> None:
        new_chunk_ids: list[int | str] = []
        for result in retrieved_results:
            doc_id = result.get("doc_id")
            if doc_id is None or doc_id in retrieved_chunk_ids:
                continue
            new_chunk_ids.append(doc_id)
            retrieved_chunk_ids.append(doc_id)
            retrieved_scores.append(result.get("score", 0.0))

        if new_chunk_ids:
            chunk_contents.extend(self._service.get_chunk_contents(new_chunk_ids))

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        query = self._service.get_query_text(query_id)
        tracker = TokenUsageTracker()

        current_answer = await self._ainvoke_text(self._build_initial_prompt(query), tracker)

        chunk_contents: list[str] = []
        critiques: list[str] = []
        reflection_actions: list[str] = []
        retrieval_queries: list[str] = []
        retrieved_chunk_ids: list[int | str] = []
        retrieved_scores: list[float | int] = []
        final_supported = False

        for _ in range(self.max_reflection_steps):
            reflection_text = await self._ainvoke_text(
                self._build_reflection_prompt(query, current_answer, chunk_contents, critiques),
                tracker,
            )
            reflection = self._parse_reflection(reflection_text)
            action = reflection["action"]
            critique = reflection["critique"]
            final_supported = bool(reflection["supported"])

            reflection_actions.append(action)
            critiques.append(critique)

            if action == "FINISH":
                break

            if action == "RETRIEVE":
                search_query = reflection["search_query"] or query
                retrieval_queries.append(search_query)
                retrieved = await self._retrieval_pipeline.retrieve(search_query, top_k)
                self._extend_context(retrieved, retrieved_chunk_ids, retrieved_scores, chunk_contents)

            current_answer = await self._ainvoke_text(
                self._build_revision_prompt(query, current_answer, chunk_contents, critique),
                tracker,
            )

        reflection_iterations = sum(1 for action in reflection_actions if action != "FINISH")

        return GenerationResult(
            text=current_answer,
            token_usage=tracker.total,
            metadata={
                "pipeline_type": "self_rag",
                "reflection_actions": reflection_actions,
                "reflection_iterations": reflection_iterations,
                "retrieval_queries": retrieval_queries,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieved_scores": retrieved_scores,
                "critiques": critiques,
                "final_supported": final_supported,
                "used_retrieval": bool(retrieval_queries),
                "support_passed": final_supported,
            },
        )


@dataclass(kw_only=True)
class SelfRAGPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for the prompt-based Self-RAG pipeline."""

    initial_prompt_template: str = field(default=DEFAULT_INITIAL_PROMPT)
    reflection_prompt_template: str = field(default=DEFAULT_REFLECTION_PROMPT)
    revision_prompt_template: str = field(default=DEFAULT_REVISION_PROMPT)
    max_reflection_steps: int = 3

    def get_pipeline_class(self) -> type["SelfRAGPipeline"]:
        return SelfRAGPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)

        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "initial_prompt_template": self.initial_prompt_template,
            "reflection_prompt_template": self.reflection_prompt_template,
            "revision_prompt_template": self.revision_prompt_template,
            "max_reflection_steps": self.max_reflection_steps,
        }
