"""SPD-RAG (Sub-Agent Per Document) Pipeline for AutoRAG-Research."""

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

DEFAULT_SUB_AGENT_SYSTEM_PROMPT = """You are a focused document analyst. Your task is to answer the given question using ONLY the provided document. If the document does not contain relevant information, state that clearly. Provide a concise, specific answer based solely on the document's content."""

DEFAULT_SUB_AGENT_USER_PROMPT = """Document:
{document}

Question: {query}

Based solely on the above document, provide your answer to the question. If the document does not contain relevant information, state 'No relevant information found in this document.'"""

DEFAULT_COORDINATOR_SYSTEM_PROMPT = """You are a relevance evaluator for a multi-document question answering system. Your task is to determine whether a partial answer generated from a specific document is relevant and useful for answering the given question. Answer with "Yes" if the partial answer provides useful information, or "No" if it does not contribute meaningfully."""

DEFAULT_COORDINATOR_USER_PROMPT = """Question: {query}

Document:
{document}

Partial Answer:
{partial_answer}

Does this partial answer provide relevant and useful information for answering the question? Respond with only "Yes" or "No"."""

DEFAULT_SYNTHESIS_SYSTEM_PROMPT = """You are a synthesis agent for a multi-document question answering system. Your task is to merge multiple partial answers into a single coherent and comprehensive answer. Resolve any conflicts between partial answers by preferring more specific and well-supported information. Do not simply concatenate - synthesize into a unified response."""

DEFAULT_SYNTHESIS_USER_PROMPT = """Question: {query}

The following partial answers were generated from different documents:

{partial_answers}

Synthesize these partial answers into a single, coherent, and comprehensive answer to the question. Resolve any conflicts and remove redundancy."""


@dataclass(kw_only=True)
class SPDRAGPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for SPD-RAG pipeline."""

    sub_agent_system_prompt: str = field(default=DEFAULT_SUB_AGENT_SYSTEM_PROMPT)
    sub_agent_user_prompt: str = field(default=DEFAULT_SUB_AGENT_USER_PROMPT)
    coordinator_system_prompt: str = field(default=DEFAULT_COORDINATOR_SYSTEM_PROMPT)
    coordinator_user_prompt: str = field(default=DEFAULT_COORDINATOR_USER_PROMPT)
    synthesis_system_prompt: str = field(default=DEFAULT_SYNTHESIS_SYSTEM_PROMPT)
    synthesis_user_prompt: str = field(default=DEFAULT_SYNTHESIS_USER_PROMPT)
    max_synthesis_tokens: int = 4000
    synthesis_batch_size: int = 3

    def get_pipeline_class(self) -> type["SPDRAGPipeline"]:
        """Return the SPD-RAG pipeline class."""
        return SPDRAGPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for the SPD-RAG pipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)

        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "sub_agent_system_prompt": self.sub_agent_system_prompt,
            "sub_agent_user_prompt": self.sub_agent_user_prompt,
            "coordinator_system_prompt": self.coordinator_system_prompt,
            "coordinator_user_prompt": self.coordinator_user_prompt,
            "synthesis_system_prompt": self.synthesis_system_prompt,
            "synthesis_user_prompt": self.synthesis_user_prompt,
            "max_synthesis_tokens": self.max_synthesis_tokens,
            "synthesis_batch_size": self.synthesis_batch_size,
        }


class SPDRAGPipeline(BaseGenerationPipeline):
    """Sub-Agent Per Document RAG pipeline."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        sub_agent_system_prompt: str = DEFAULT_SUB_AGENT_SYSTEM_PROMPT,
        sub_agent_user_prompt: str = DEFAULT_SUB_AGENT_USER_PROMPT,
        coordinator_system_prompt: str = DEFAULT_COORDINATOR_SYSTEM_PROMPT,
        coordinator_user_prompt: str = DEFAULT_COORDINATOR_USER_PROMPT,
        synthesis_system_prompt: str = DEFAULT_SYNTHESIS_SYSTEM_PROMPT,
        synthesis_user_prompt: str = DEFAULT_SYNTHESIS_USER_PROMPT,
        max_synthesis_tokens: int = 4000,
        synthesis_batch_size: int = 3,
        schema: Any | None = None,
    ) -> None:
        """Initialize SPD-RAG pipeline."""
        self._sub_agent_system_prompt = sub_agent_system_prompt
        self._sub_agent_user_prompt = sub_agent_user_prompt
        self._coordinator_system_prompt = coordinator_system_prompt
        self._coordinator_user_prompt = coordinator_user_prompt
        self._synthesis_system_prompt = synthesis_system_prompt
        self._synthesis_user_prompt = synthesis_user_prompt
        self._max_synthesis_tokens = max_synthesis_tokens
        self._synthesis_batch_size = synthesis_batch_size

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return SPD-RAG pipeline configuration."""
        return {
            "type": "spd_rag",
            "sub_agent_system_prompt": self._sub_agent_system_prompt,
            "sub_agent_user_prompt": self._sub_agent_user_prompt,
            "coordinator_system_prompt": self._coordinator_system_prompt,
            "coordinator_user_prompt": self._coordinator_user_prompt,
            "synthesis_system_prompt": self._synthesis_system_prompt,
            "synthesis_user_prompt": self._synthesis_user_prompt,
            "max_synthesis_tokens": self._max_synthesis_tokens,
            "synthesis_batch_size": self._synthesis_batch_size,
        }

    async def _ainvoke_llm(self, system_prompt: str, user_prompt: str, **format_kwargs: Any) -> Any:
        """Invoke the LLM with system and user messages."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt.format(**format_kwargs)),
        ]
        return await self._llm.ainvoke(messages)

    async def _sub_agent_generate(self, query: str, document: str) -> tuple[str, Any]:
        """Generate a partial answer from a single document."""
        response = await self._ainvoke_llm(
            self._sub_agent_system_prompt,
            self._sub_agent_user_prompt,
            query=query,
            document=document,
        )
        text = response.content if hasattr(response, "content") else str(response)
        return text, response

    async def _coordinator_evaluate(self, query: str, partial_answer: str, document: str) -> tuple[bool, Any]:
        """Evaluate whether a partial answer should contribute to synthesis."""
        response = await self._ainvoke_llm(
            self._coordinator_system_prompt,
            self._coordinator_user_prompt,
            query=query,
            partial_answer=partial_answer,
            document=document,
        )
        text = response.content if hasattr(response, "content") else str(response)
        return text.strip().lower().startswith("yes"), response

    async def _synthesize_answers(
        self,
        query: str,
        relevant_answers: list[dict[str, Any]],
        tracker: TokenUsageTracker,
    ) -> str:
        """Recursively synthesize partial answers with map-reduce batching."""
        current_answers = [answer["partial_answer"] for answer in relevant_answers]

        while len(current_answers) > 1:
            next_level: list[str] = []

            for i in range(0, len(current_answers), self._synthesis_batch_size):
                batch = current_answers[i : i + self._synthesis_batch_size]
                if len(batch) == 1:
                    next_level.append(batch[0])
                    continue

                formatted_answers = "\n\n".join(f"[Partial Answer {j + 1}]\n{answer}" for j, answer in enumerate(batch))
                response = await self._ainvoke_llm(
                    self._synthesis_system_prompt,
                    self._synthesis_user_prompt,
                    query=query,
                    partial_answers=formatted_answers,
                )
                text = response.content if hasattr(response, "content") else str(response)
                next_level.append(text)
                tracker.record(response)

            current_answers = next_level

        return current_answers[0]

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate an answer using the SPD-RAG algorithm."""
        query = self._service.get_query_text(query_id)
        tracker = TokenUsageTracker()

        retrieved = await self._retrieval_pipeline._retrieve_by_id(query_id, top_k)
        if not retrieved:
            return GenerationResult(
                text="",
                token_usage=tracker.total,
                metadata={
                    "pipeline_type": "spd_rag",
                    "error": "No documents retrieved",
                    "original_doc_count": 0,
                    "relevant_answer_count": 0,
                },
            )

        chunk_ids = [result["doc_id"] for result in retrieved]
        retrieval_scores = [result["score"] for result in retrieved]
        chunk_contents = self._service.get_chunk_contents(chunk_ids)

        partial_answers: list[dict[str, Any]] = []
        for chunk_id, document in zip(chunk_ids, chunk_contents, strict=True):
            partial_answer, response = await self._sub_agent_generate(query, document)
            partial_answers.append({
                "doc_id": chunk_id,
                "content": document,
                "partial_answer": partial_answer,
            })
            tracker.record(response)

        relevant_answers: list[dict[str, Any]] = []
        for partial_answer in partial_answers:
            is_relevant, response = await self._coordinator_evaluate(
                query,
                partial_answer["partial_answer"],
                partial_answer["content"],
            )
            tracker.record(response)
            if is_relevant:
                relevant_answers.append(partial_answer)

        used_fallback = False
        if not relevant_answers:
            logger.warning("Coordinator filtered all partial answers, using all as fallback")
            relevant_answers = partial_answers
            used_fallback = True

        if len(relevant_answers) == 1:
            final_text = relevant_answers[0]["partial_answer"]
        else:
            final_text = await self._synthesize_answers(query, relevant_answers, tracker)

        return GenerationResult(
            text=final_text,
            token_usage=tracker.total,
            metadata={
                "pipeline_type": "spd_rag",
                "original_doc_count": len(chunk_contents),
                "relevant_answer_count": len(relevant_answers),
                "retrieved_chunk_ids": chunk_ids,
                "retrieval_scores": retrieval_scores,
                "synthesis_batch_size": self._synthesis_batch_size,
                "max_synthesis_tokens": self._max_synthesis_tokens,
                "used_coordinator_fallback": used_fallback,
            },
        )
