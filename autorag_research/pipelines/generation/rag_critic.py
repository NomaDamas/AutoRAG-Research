"""RAG-Critic generation pipeline for AutoRAG-Research.

Implements a critic-guided corrective loop inspired by the ACL 2025
RAG-Critic workflow while staying compatible with this repository's
generation-pipeline abstraction:

1. Retrieve documents for the original query
2. Generate an initial answer
3. Ask a critic to judge whether the answer needs revision
4. If revision is needed, ask a planner to choose corrective actions
5. Execute supported actions (retrieval / rewrite / decomposition /
   context refinement / answer regeneration)
6. Re-run the critic until approval or max_iterations is reached
"""

from __future__ import annotations

import json
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

DEFAULT_ANSWER_PROMPT = """You are answering a retrieval-augmented generation question.

Question: {query}

Retrieved context:
{context}

Instruction: {instruction}

Write the best grounded answer you can using the retrieved context. If the context is insufficient, say so explicitly."""

DEFAULT_CRITIC_PROMPT = """You are a critic for a retrieval-augmented generation workflow.

Question: {query}

Retrieved context:
{context}

Current answer:
{answer}

Return JSON with:
- verdict: "approved" or "revise"
- feedback: a concise explanation
- recommended_actions: list chosen from ["retrieval", "rewrite_query", "decompose_query", "refine_documents", "generate_answer"]"""

DEFAULT_PLANNER_PROMPT = """You are a critic-guided planner for a retrieval-augmented generation workflow.

Question: {query}

Current answer:
{answer}

Critic feedback JSON:
{critique}

Return JSON with an actions array. Each action item must contain:
- action: one of ["retrieval", "rewrite_query", "decompose_query", "refine_documents", "generate_answer"]

Optional fields:
- instruction: short text instruction
- query_source: one of ["original", "rewritten_query", "sub_questions"] for retrieval. If omitted, retrieval uses the current working query.
- top_k: integer for retrieval
- strategy: "replace" or "append" for retrieval"""

DEFAULT_REWRITE_PROMPT = """You are rewriting a query for better retrieval.

Original query: {query}
Critic feedback: {feedback}
Planner instruction: {instruction}

Return only a rewritten query."""

DEFAULT_DECOMPOSITION_PROMPT = """You are decomposing a query into retrieval-friendly sub-questions.

Current query: {query}
Critic feedback: {feedback}
Planner instruction: {instruction}

Return JSON with a sub_questions array."""

DEFAULT_REFINE_PROMPT = """You are refining retrieved evidence before answer generation.

Question: {query}
Critic feedback: {feedback}
Planner instruction: {instruction}

Retrieved context:
{context}

Return only the refined document."""

SUPPORTED_ACTIONS = {
    "retrieval",
    "rewrite_query",
    "decompose_query",
    "refine_documents",
    "generate_answer",
}


@dataclass(kw_only=True)
class RAGCriticPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for the RAG-Critic pipeline."""

    answer_prompt_template: str = field(default=DEFAULT_ANSWER_PROMPT)
    critic_prompt_template: str = field(default=DEFAULT_CRITIC_PROMPT)
    planner_prompt_template: str = field(default=DEFAULT_PLANNER_PROMPT)
    rewrite_prompt_template: str = field(default=DEFAULT_REWRITE_PROMPT)
    decomposition_prompt_template: str = field(default=DEFAULT_DECOMPOSITION_PROMPT)
    refine_prompt_template: str = field(default=DEFAULT_REFINE_PROMPT)
    max_iterations: int = 2
    max_actions_per_iteration: int = 4

    def get_pipeline_class(self) -> type[RAGCriticPipeline]:
        """Return the RAG-Critic pipeline class."""
        return RAGCriticPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for the RAG-Critic constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)

        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "answer_prompt_template": self.answer_prompt_template,
            "critic_prompt_template": self.critic_prompt_template,
            "planner_prompt_template": self.planner_prompt_template,
            "rewrite_prompt_template": self.rewrite_prompt_template,
            "decomposition_prompt_template": self.decomposition_prompt_template,
            "refine_prompt_template": self.refine_prompt_template,
            "max_iterations": self.max_iterations,
            "max_actions_per_iteration": self.max_actions_per_iteration,
        }


class RAGCriticPipeline(BaseGenerationPipeline):
    """Critic-guided corrective generation pipeline."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        answer_prompt_template: str = DEFAULT_ANSWER_PROMPT,
        critic_prompt_template: str = DEFAULT_CRITIC_PROMPT,
        planner_prompt_template: str = DEFAULT_PLANNER_PROMPT,
        rewrite_prompt_template: str = DEFAULT_REWRITE_PROMPT,
        decomposition_prompt_template: str = DEFAULT_DECOMPOSITION_PROMPT,
        refine_prompt_template: str = DEFAULT_REFINE_PROMPT,
        max_iterations: int = 2,
        max_actions_per_iteration: int = 4,
        schema: Any | None = None,
    ) -> None:
        """Initialize the RAG-Critic pipeline."""
        self._answer_prompt_template = answer_prompt_template
        self._critic_prompt_template = critic_prompt_template
        self._planner_prompt_template = planner_prompt_template
        self._rewrite_prompt_template = rewrite_prompt_template
        self._decomposition_prompt_template = decomposition_prompt_template
        self._refine_prompt_template = refine_prompt_template
        self._max_iterations = max_iterations
        self._max_actions_per_iteration = max_actions_per_iteration

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return RAG-Critic pipeline configuration."""
        return {
            "type": "rag_critic",
            "answer_prompt_template": self._answer_prompt_template,
            "critic_prompt_template": self._critic_prompt_template,
            "planner_prompt_template": self._planner_prompt_template,
            "rewrite_prompt_template": self._rewrite_prompt_template,
            "decomposition_prompt_template": self._decomposition_prompt_template,
            "refine_prompt_template": self._refine_prompt_template,
            "max_iterations": self._max_iterations,
            "max_actions_per_iteration": self._max_actions_per_iteration,
            "retrieval_pipeline_id": self._retrieval_pipeline.pipeline_id,
        }

    @staticmethod
    def _parse_json_payload(text: str) -> dict[str, Any]:
        """Parse structured JSON responses, tolerating markdown fences."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        json_match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
        if json_match:
            cleaned = json_match.group(1)
        return json.loads(cleaned)

    @staticmethod
    def _normalize_action_name(action: str) -> str:
        """Normalize action names from planner output."""
        normalized = action.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "retrieve": "retrieval",
            "refine_document": "refine_documents",
            "refine_docs": "refine_documents",
            "generate": "generate_answer",
        }
        return aliases.get(normalized, normalized)

    @staticmethod
    def _is_approved(critique: dict[str, Any]) -> bool:
        """Return whether the critic approved the answer."""
        verdict = str(critique.get("verdict", "")).strip().lower()
        return verdict in {"approved", "approve", "pass", "correct"}

    @staticmethod
    def _format_documents(documents: list[dict[str, Any]]) -> str:
        """Format hydrated documents for prompt context."""
        if not documents:
            return "(No documents available)"
        return "\n\n".join(
            (f"[Document {index} | doc_id={document['doc_id']} | score={document['score']}]\n{document['content']}")
            for index, document in enumerate(documents, start=1)
        )

    def _hydrate_documents(self, retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach chunk contents to retrieval results."""
        if not retrieved:
            return []
        chunk_ids = [item["doc_id"] for item in retrieved]
        chunk_contents = self._service.get_chunk_contents(chunk_ids)
        return [
            {
                "doc_id": item["doc_id"],
                "score": item["score"],
                "content": content,
            }
            for item, content in zip(retrieved, chunk_contents, strict=True)
        ]

    @staticmethod
    def _merge_documents(
        current: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
        strategy: str,
    ) -> list[dict[str, Any]]:
        """Merge documents while preserving existing order."""
        if strategy == "replace":
            return [dict(item) for item in incoming]

        merged = [dict(item) for item in current]
        index_by_id = {item["doc_id"]: idx for idx, item in enumerate(merged)}
        for item in incoming:
            doc_id = item["doc_id"]
            if doc_id in index_by_id:
                existing_index = index_by_id[doc_id]
                if item["score"] > merged[existing_index]["score"]:
                    merged[existing_index] = dict(item)
            else:
                index_by_id[doc_id] = len(merged)
                merged.append(dict(item))
        return merged

    async def _invoke_and_record(self, prompt: str, tracker: TokenUsageTracker) -> str:
        """Invoke the LLM and record token usage."""
        response = await self._llm.ainvoke(prompt)
        tracker.record(response)
        return response.content if hasattr(response, "content") else str(response)

    async def _generate_answer(
        self,
        query: str,
        context: str,
        instruction: str,
        tracker: TokenUsageTracker,
    ) -> str:
        """Generate an answer from the current context."""
        prompt = self._answer_prompt_template.format(
            query=query,
            context=context,
            instruction=instruction or "Answer directly and stay grounded in the documents.",
        )
        return await self._invoke_and_record(prompt, tracker)

    async def _run_critic(
        self,
        query: str,
        context: str,
        answer: str,
        tracker: TokenUsageTracker,
    ) -> dict[str, Any]:
        """Run the critic and parse its JSON output."""
        prompt = self._critic_prompt_template.format(query=query, context=context, answer=answer)
        critique_text = await self._invoke_and_record(prompt, tracker)
        try:
            critique = self._parse_json_payload(critique_text)
        except json.JSONDecodeError:
            logger.warning("Critic output was not valid JSON; falling back to revise verdict")
            critique = {"verdict": "revise", "feedback": critique_text, "recommended_actions": ["generate_answer"]}
        critique.setdefault("feedback", "")
        critique.setdefault("recommended_actions", [])
        return critique

    async def _plan_actions(
        self,
        query: str,
        answer: str,
        critique: dict[str, Any],
        tracker: TokenUsageTracker,
    ) -> list[dict[str, Any]]:
        """Plan corrective actions from the critic feedback."""
        prompt = self._planner_prompt_template.format(
            query=query,
            answer=answer,
            critique=json.dumps(critique, ensure_ascii=False),
        )
        plan_text = await self._invoke_and_record(prompt, tracker)
        try:
            payload = self._parse_json_payload(plan_text)
        except json.JSONDecodeError:
            payload = {"actions": critique.get("recommended_actions", [])}

        raw_actions = payload.get("actions", [])
        actions: list[dict[str, Any]] = []
        for item in raw_actions:
            if isinstance(item, str):
                actions.append({"action": item})
            elif isinstance(item, dict):
                actions.append(item)
        if not actions:
            actions = [{"action": action} for action in critique.get("recommended_actions", [])]
        return actions[: self._max_actions_per_iteration]

    async def _rewrite_query(
        self,
        query: str,
        feedback: str,
        instruction: str,
        tracker: TokenUsageTracker,
    ) -> str:
        """Rewrite the current retrieval query."""
        prompt = self._rewrite_prompt_template.format(query=query, feedback=feedback, instruction=instruction)
        rewritten = await self._invoke_and_record(prompt, tracker)
        return rewritten.strip()

    async def _decompose_query(
        self,
        query: str,
        feedback: str,
        instruction: str,
        tracker: TokenUsageTracker,
    ) -> list[str]:
        """Generate retrieval sub-questions."""
        prompt = self._decomposition_prompt_template.format(query=query, feedback=feedback, instruction=instruction)
        decomposition_text = await self._invoke_and_record(prompt, tracker)
        try:
            payload = self._parse_json_payload(decomposition_text)
        except json.JSONDecodeError:
            payload = {"sub_questions": [part.strip() for part in decomposition_text.split("\n") if part.strip()]}
        sub_questions = payload.get("sub_questions", [])
        return [question.strip() for question in sub_questions if isinstance(question, str) and question.strip()]

    async def _refine_context(
        self,
        query: str,
        context: str,
        feedback: str,
        instruction: str,
        tracker: TokenUsageTracker,
    ) -> str:
        """Refine the working context before answer regeneration."""
        prompt = self._refine_prompt_template.format(
            query=query,
            feedback=feedback,
            instruction=instruction,
            context=context,
        )
        return await self._invoke_and_record(prompt, tracker)

    async def _handle_rewrite_action(
        self,
        raw_action: dict[str, Any],
        *,
        working_query: str,
        feedback: str,
        rewritten_queries: list[str],
        executed_actions: list[dict[str, Any]],
        tracker: TokenUsageTracker,
    ) -> str:
        """Execute a rewrite_query action."""
        rewritten_query = await self._rewrite_query(
            working_query,
            feedback,
            str(raw_action.get("instruction", "")),
            tracker,
        )
        rewritten_queries.append(rewritten_query)
        executed_actions.append({"action": "rewrite_query", "query": rewritten_query})
        return rewritten_query

    async def _handle_decompose_action(
        self,
        raw_action: dict[str, Any],
        *,
        working_query: str,
        feedback: str,
        sub_questions: list[str],
        executed_actions: list[dict[str, Any]],
        tracker: TokenUsageTracker,
    ) -> None:
        """Execute a decompose_query action."""
        for question in await self._decompose_query(
            working_query,
            feedback,
            str(raw_action.get("instruction", "")),
            tracker,
        ):
            if question not in sub_questions:
                sub_questions.append(question)
        executed_actions.append({"action": "decompose_query", "sub_questions": list(sub_questions)})

    async def _handle_retrieval_action(
        self,
        raw_action: dict[str, Any],
        *,
        original_query: str,
        working_query: str,
        documents: list[dict[str, Any]],
        rewritten_queries: list[str],
        sub_questions: list[str],
        retrieval_history: list[dict[str, Any]],
        executed_actions: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], str]:
        """Execute a retrieval action and return updated documents/context."""
        query_source = str(raw_action.get("query_source", "working")).strip().lower()
        strategy = str(raw_action.get("strategy", "append")).strip().lower()
        action_top_k = int(raw_action.get("top_k", top_k))
        retrieval_queries = self._select_retrieval_queries(
            query_source=query_source,
            original_query=original_query,
            rewritten_queries=rewritten_queries,
            sub_questions=sub_questions,
            working_query=working_query,
        )

        new_documents: list[dict[str, Any]] = []
        for retrieval_query in retrieval_queries:
            hydrated_documents = self._hydrate_documents(
                await self._retrieval_pipeline.retrieve(retrieval_query, action_top_k)
            )
            new_documents = self._merge_documents(new_documents, hydrated_documents, "append")
            retrieval_history.append({
                "query": retrieval_query,
                "query_source": query_source,
                "strategy": strategy,
                "returned_doc_ids": [doc["doc_id"] for doc in hydrated_documents],
            })

        merged_documents = self._merge_documents(documents, new_documents, strategy)
        executed_actions.append({
            "action": "retrieval",
            "query_source": query_source,
            "queries": retrieval_queries,
            "strategy": strategy,
        })
        return merged_documents, self._format_documents(merged_documents)

    async def _handle_refine_action(
        self,
        raw_action: dict[str, Any],
        *,
        original_query: str,
        working_context: str,
        feedback: str,
        executed_actions: list[dict[str, Any]],
        tracker: TokenUsageTracker,
    ) -> str:
        """Execute a refine_documents action."""
        instruction = str(raw_action.get("instruction", ""))
        refined_context = await self._refine_context(
            original_query,
            working_context,
            feedback,
            instruction,
            tracker,
        )
        executed_actions.append({"action": "refine_documents", "instruction": instruction})
        return refined_context

    async def _handle_generate_action(
        self,
        raw_action: dict[str, Any],
        *,
        original_query: str,
        working_context: str,
        executed_actions: list[dict[str, Any]],
        tracker: TokenUsageTracker,
    ) -> str:
        """Execute a generate_answer action."""
        instruction = str(raw_action.get("instruction", ""))
        answer = await self._generate_answer(original_query, working_context, instruction, tracker)
        executed_actions.append({"action": "generate_answer", "instruction": instruction})
        return answer

    async def _dispatch_action(
        self,
        action_name: str,
        raw_action: dict[str, Any],
        *,
        original_query: str,
        working_query: str,
        documents: list[dict[str, Any]],
        working_context: str,
        feedback: str,
        rewritten_queries: list[str],
        sub_questions: list[str],
        executed_actions: list[dict[str, Any]],
        retrieval_history: list[dict[str, Any]],
        top_k: int,
        tracker: TokenUsageTracker,
    ) -> dict[str, Any]:
        """Dispatch a single planner action and return changed state fields."""
        if action_name == "rewrite_query":
            return {
                "working_query": await self._handle_rewrite_action(
                    raw_action,
                    working_query=working_query,
                    feedback=feedback,
                    rewritten_queries=rewritten_queries,
                    executed_actions=executed_actions,
                    tracker=tracker,
                )
            }

        if action_name == "decompose_query":
            await self._handle_decompose_action(
                raw_action,
                working_query=working_query,
                feedback=feedback,
                sub_questions=sub_questions,
                executed_actions=executed_actions,
                tracker=tracker,
            )
            return {}

        if action_name == "retrieval":
            documents, working_context = await self._handle_retrieval_action(
                raw_action,
                original_query=original_query,
                working_query=working_query,
                documents=documents,
                rewritten_queries=rewritten_queries,
                sub_questions=sub_questions,
                retrieval_history=retrieval_history,
                executed_actions=executed_actions,
                top_k=top_k,
            )
            return {
                "documents": documents,
                "working_context": working_context,
            }

        if action_name == "refine_documents":
            return {
                "working_context": await self._handle_refine_action(
                    raw_action,
                    original_query=original_query,
                    working_context=working_context,
                    feedback=feedback,
                    executed_actions=executed_actions,
                    tracker=tracker,
                )
            }

        if action_name == "generate_answer":
            return {
                "answer": await self._handle_generate_action(
                    raw_action,
                    original_query=original_query,
                    working_context=working_context,
                    executed_actions=executed_actions,
                    tracker=tracker,
                ),
                "answer_updated": True,
            }

        logger.warning("Skipping unsupported RAG-Critic action: %s", action_name)
        return {}

    async def _execute_action_plan(
        self,
        actions: list[dict[str, Any]],
        *,
        original_query: str,
        working_query: str,
        documents: list[dict[str, Any]],
        working_context: str,
        feedback: str,
        rewritten_queries: list[str],
        sub_questions: list[str],
        executed_actions: list[dict[str, Any]],
        retrieval_history: list[dict[str, Any]],
        answer: str,
        top_k: int,
        tracker: TokenUsageTracker,
    ) -> tuple[str, list[dict[str, Any]], str, str, bool]:
        """Execute the planner's actions and return the updated working state."""
        answer_updated = False

        for raw_action in actions:
            action_name = self._normalize_action_name(str(raw_action.get("action", raw_action.get("name", ""))))
            updates = await self._dispatch_action(
                action_name,
                raw_action,
                original_query=original_query,
                working_query=working_query,
                documents=documents,
                working_context=working_context,
                feedback=feedback,
                rewritten_queries=rewritten_queries,
                sub_questions=sub_questions,
                executed_actions=executed_actions,
                retrieval_history=retrieval_history,
                top_k=top_k,
                tracker=tracker,
            )
            working_query = updates.get("working_query", working_query)
            documents = updates.get("documents", documents)
            working_context = updates.get("working_context", working_context)
            answer = updates.get("answer", answer)
            answer_updated = updates.get("answer_updated", answer_updated)

        return answer, documents, working_context, working_query, answer_updated

    @staticmethod
    def _select_retrieval_queries(
        query_source: str,
        original_query: str,
        rewritten_queries: list[str],
        sub_questions: list[str],
        working_query: str,
    ) -> list[str]:
        """Resolve which queries a retrieval action should execute."""
        if query_source == "sub_questions" and sub_questions:
            return sub_questions
        if query_source == "rewritten_query" and rewritten_queries:
            return [rewritten_queries[-1]]
        if query_source == "original":
            return [original_query]
        return [working_query]

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate an answer with critic-guided correction."""
        original_query = self._service.get_query_text(query_id)
        working_query = original_query
        tracker = TokenUsageTracker()

        retrieved = await self._retrieval_pipeline._retrieve_by_id(query_id, top_k)
        if not retrieved:
            return GenerationResult(
                text="",
                token_usage=tracker.total,
                metadata={
                    "pipeline_type": "rag_critic",
                    "error": "No documents retrieved",
                    "iteration_count": 0,
                    "executed_actions": [],
                    "critique_history": [],
                    "retrieved_chunk_ids": [],
                    "retrieval_scores": [],
                },
            )

        documents = self._hydrate_documents(retrieved)
        working_context = self._format_documents(documents)
        retrieval_history = [
            {
                "query": original_query,
                "query_source": "original",
                "strategy": "replace",
                "returned_doc_ids": [doc["doc_id"] for doc in documents],
            }
        ]
        executed_actions: list[dict[str, Any]] = []
        critique_history: list[dict[str, Any]] = []
        rewritten_queries: list[str] = []
        sub_questions: list[str] = []

        answer = await self._generate_answer(original_query, working_context, "", tracker)

        for _ in range(self._max_iterations):
            critique = await self._run_critic(original_query, working_context, answer, tracker)
            critique_history.append(critique)
            if self._is_approved(critique):
                break

            feedback = critique["feedback"]
            actions = await self._plan_actions(original_query, answer, critique, tracker)
            answer, documents, working_context, working_query, answer_updated = await self._execute_action_plan(
                actions,
                original_query=original_query,
                working_query=working_query,
                documents=documents,
                working_context=working_context,
                feedback=feedback,
                rewritten_queries=rewritten_queries,
                sub_questions=sub_questions,
                executed_actions=executed_actions,
                retrieval_history=retrieval_history,
                answer=answer,
                top_k=top_k,
                tracker=tracker,
            )

            if not answer_updated:
                answer = await self._generate_answer(original_query, working_context, feedback, tracker)

        return GenerationResult(
            text=answer,
            token_usage=tracker.total,
            metadata={
                "pipeline_type": "rag_critic",
                "iteration_count": len(critique_history),
                "executed_actions": executed_actions,
                "critique_history": critique_history,
                "rewritten_queries": rewritten_queries,
                "sub_questions": sub_questions,
                "retrieved_chunk_ids": [document["doc_id"] for document in documents],
                "retrieval_scores": [document["score"] for document in documents],
                "retrieval_history": retrieval_history,
            },
        )
