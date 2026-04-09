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

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, cast

from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseGenerationPipelineConfig
from autorag_research.orm.service.generation_pipeline import GenerationResult
from autorag_research.pipelines.generation.base import BaseGenerationPipeline
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.util import TokenUsageTracker

logger = logging.getLogger("AutoRAG-Research")

JsonPayload = dict[str, Any] | list[Any]

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

DEFAULT_TRAINED_CRITIC_PROMPT = """You are the RAG-Critic error-analysis model.

Below are the [Question], the retrieved [Passages], and the [Model's Prediction] for a RAG task.

Question: {query}

Retrieved Passages:
{context}

Model's Prediction:
{answer}

Please first determine whether the model's prediction is correct.

If it is correct, output exactly:
{{"Judgement": "Correct"}}

If it is incorrect, analyze the errors and identify tags at three levels.
Use the following tag taxonomy where tag1 and tag2 correspond semantically:

tag1 = [
    "Incomplete Information",
    "Irrelevant Information",
    "Erroneous Information",
    "Incomplete or Missing Response",
    "Inaccurate or Misunderstood Response",
    "Irrelevant or Off-Topic Response",
    "Overly Verbose Response"
]

tag2 = [
    "Insufficient or Incomplete Information Retrieval",
    "Data Insufficiency in Retrieval",
    "Relevance Gaps in Retrieval",
    "Irrelevant Information Retrieval",
    "Erroneous Information Retrieval",
    "Omission of Key Information",
    "Lack of Specificity",
    "Specificity and Precision Errors",
    "Partial Coverage and Temporal Issues",
    "Lack of Practicality",
    "Contextual Understanding Errors",
    "Factual Inaccuracies",
    "Incorrect and Incomplete Answers",
    "Golden Answer Misalignment",
    "Misinterpretation of Queries and Information",
    "Entity and Concept Confusion",
    "Irrelevant Content and Topic Drift",
    "Off-Topic and Redundant Responses",
    "Content and Context Misalignment",
    "Overly Complex and Redundant Response"
]

Adhere strictly to this JSON format:
{{
    "Judgement": "Error",
    "Error_analysis": "",
    "tag1": [],
    "tag2": [],
    "tag3": []
}}"""

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
- strategy: "replace" or "append" for retrieval

Strict output rules:
- Return valid JSON only. No markdown fences.
- If you include top_k, emit a JSON integer literal such as 1, 3, or 5.
- Do not emit top_k as a string, float, null, boolean, or word.
- Omit top_k entirely if you are unsure."""

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

DEFAULT_AGENT_SYSTEM_PROMPT = """
You are an agent tasked with optimizing a Retrieval-Augmented Generation process. The goal is to improve the model's
predictions by addressing issues flagged in the error_type. You are given the results from an initial RAG process,
including a query, a list of retrieved documents, a prediction, and the identified error type. Your task is to
optimize the current RAG process by selecting the appropriate functions and generating the corresponding Python code to
fix the problem.

### Available Functions

1. `Retrieval(query: str, topk: int) -> List[str]`
2. `RewriteQuery(query: str, instruction: str) -> List[str]`
3. `DecomposeQuery(query: str) -> List[str]`
4. `RefineDoc(query: str, doc: str, instruction: str) -> str`
5. `GenerateAnswer(query: str, docs: List[str], additional_instruction: str = None) -> str`

You can directly use the variables I provide as the input of the functions. You can freely combine the functions to
improve the performance. Ensure that each function execution is necessary and can improve the result. Only give code.
You must use `final_answer = GenerateAnswer(...)` in the end.
"""

DEFAULT_AGENT_USER_PROMPT = """Given the following information:

question = "{question}"
doc_list = {doc_list}
previous_pred = "{previous_pred}"

Error type of previous pred: {error_type}

Please carefully read the provided question, doc list, previous answer, and the error type of the previous prediction
given by a teacher model. Your task is to generate Python code that calls the relevant functions to optimize the
current RAG process and solve the previous error. The generated code should only include function calls and variable
assignments. Do not write function implementations or any explanation.
"""

DEFAULT_AGENT_GENERATE_ANSWER_PROMPT = """Find the useful content from the provided documents, then answer the
question directly. Your response should be very concise. Please use 'So the final answer is:' as a prefix for the
final answer.

Additional instruction:
{instruction}

Question: {query}

Documents:
{context}

Response:"""

DEFAULT_AGENT_REWRITE_CLARIFY_PROMPT = """Please rewrite the given query to make it more specific, clear, and focused.
Output only valid JSON with the query under the "query" key.

Original query: {query}"""

DEFAULT_AGENT_REWRITE_EXPAND_PROMPT = """Please rewrite the given query by expanding it with additional relevant
questions or variations that address the same topic. Output only a valid JSON array of query strings.

Original query: {query}"""

DEFAULT_AGENT_REWRITE_CUSTOM_PROMPT = """Please rewrite the given query based on the following instruction:
{instruction}. Output only valid JSON with the query under the "query" key.

Original query: {query}"""

DEFAULT_AGENT_DECOMPOSE_PROMPT = """Please split the given query into multiple smaller, more specific subqueries.
Output only a valid JSON array of subquery strings.

Original query: {query}"""

DEFAULT_AGENT_REFINE_SUMMARIZE_PROMPT = """Please refine the given document to retain only the information helpful for
answering the provided question. Output only valid JSON with the refined content under the "refined_document" key.

Document: {document}
Question: {question}"""

DEFAULT_AGENT_REFINE_EXPLAIN_PROMPT = """Please read the given document carefully and provide a detailed explanation
that answers the question. Output only valid JSON with the explanation under the "explanation" key.

Original document: {document}
Question: {question}"""

SUPPORTED_ACTIONS = {
    "retrieval",
    "rewrite_query",
    "decompose_query",
    "refine_documents",
    "generate_answer",
}
SUPPORTED_PLANNER_OUTPUT_FORMATS = {"json_actions", "python_agent"}
SUPPORTED_CRITIC_OUTPUT_FORMATS = {"json_actions", "rag_critic_tags"}

RAG_CRITIC_3B_TAG2_ACTIONS = {
    "insufficient or incomplete information retrieval": ["retrieval"],
    "data insufficiency in retrieval": ["retrieval"],
    "relevance gaps in retrieval": ["retrieval"],
    "irrelevant information retrieval": ["retrieval", "refine_documents"],
    "erroneous information retrieval": ["retrieval", "refine_documents"],
    "omission of key information": ["generate_answer"],
    "lack of specificity": ["generate_answer"],
    "specificity and precision errors": ["generate_answer"],
    "partial coverage and temporal issues": ["retrieval", "generate_answer"],
    "lack of practicality": ["generate_answer"],
    "contextual understanding errors": ["rewrite_query", "generate_answer"],
    "factual inaccuracies": ["generate_answer"],
    "incorrect and incomplete answers": ["generate_answer"],
    "golden answer misalignment": ["generate_answer"],
    "misinterpretation of queries and information": ["rewrite_query", "decompose_query"],
    "entity and concept confusion": ["rewrite_query", "decompose_query"],
    "irrelevant content and topic drift": ["refine_documents", "generate_answer"],
    "off-topic and redundant responses": ["refine_documents", "generate_answer"],
    "content and context misalignment": ["refine_documents", "generate_answer"],
    "overly complex and redundant response": ["refine_documents", "generate_answer"],
}


@dataclass(kw_only=True)
class RAGCriticPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for the RAG-Critic pipeline."""

    critic_llm: str | BaseLanguageModel | None = None
    answer_prompt_template: str = field(default=DEFAULT_ANSWER_PROMPT)
    critic_prompt_template: str = field(default=DEFAULT_CRITIC_PROMPT)
    trained_critic_prompt_template: str = field(default=DEFAULT_TRAINED_CRITIC_PROMPT)
    planner_prompt_template: str = field(default=DEFAULT_PLANNER_PROMPT)
    agent_system_prompt_template: str = field(default=DEFAULT_AGENT_SYSTEM_PROMPT)
    agent_user_prompt_template: str = field(default=DEFAULT_AGENT_USER_PROMPT)
    rewrite_prompt_template: str = field(default=DEFAULT_REWRITE_PROMPT)
    decomposition_prompt_template: str = field(default=DEFAULT_DECOMPOSITION_PROMPT)
    refine_prompt_template: str = field(default=DEFAULT_REFINE_PROMPT)
    agent_generate_answer_prompt_template: str = field(default=DEFAULT_AGENT_GENERATE_ANSWER_PROMPT)
    agent_rewrite_clarify_prompt_template: str = field(default=DEFAULT_AGENT_REWRITE_CLARIFY_PROMPT)
    agent_rewrite_expand_prompt_template: str = field(default=DEFAULT_AGENT_REWRITE_EXPAND_PROMPT)
    agent_rewrite_custom_prompt_template: str = field(default=DEFAULT_AGENT_REWRITE_CUSTOM_PROMPT)
    agent_decompose_prompt_template: str = field(default=DEFAULT_AGENT_DECOMPOSE_PROMPT)
    agent_refine_summarize_prompt_template: str = field(default=DEFAULT_AGENT_REFINE_SUMMARIZE_PROMPT)
    agent_refine_explain_prompt_template: str = field(default=DEFAULT_AGENT_REFINE_EXPLAIN_PROMPT)
    critic_output_format: str = "json_actions"
    planner_output_format: str = "json_actions"
    max_iterations: int = 2
    max_actions_per_iteration: int = 4

    def __setattr__(self, name: str, value: Any) -> None:
        """Auto-convert optional critic LLM config names to model instances."""
        if name == "critic_llm" and isinstance(value, str):
            from autorag_research.injection import load_llm

            value = load_llm(value)
        super().__setattr__(name, value)

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
            "critic_llm": self.critic_llm,
            "answer_prompt_template": self.answer_prompt_template,
            "critic_prompt_template": self.critic_prompt_template,
            "trained_critic_prompt_template": self.trained_critic_prompt_template,
            "planner_prompt_template": self.planner_prompt_template,
            "agent_system_prompt_template": self.agent_system_prompt_template,
            "agent_user_prompt_template": self.agent_user_prompt_template,
            "rewrite_prompt_template": self.rewrite_prompt_template,
            "decomposition_prompt_template": self.decomposition_prompt_template,
            "refine_prompt_template": self.refine_prompt_template,
            "agent_generate_answer_prompt_template": self.agent_generate_answer_prompt_template,
            "agent_rewrite_clarify_prompt_template": self.agent_rewrite_clarify_prompt_template,
            "agent_rewrite_expand_prompt_template": self.agent_rewrite_expand_prompt_template,
            "agent_rewrite_custom_prompt_template": self.agent_rewrite_custom_prompt_template,
            "agent_decompose_prompt_template": self.agent_decompose_prompt_template,
            "agent_refine_summarize_prompt_template": self.agent_refine_summarize_prompt_template,
            "agent_refine_explain_prompt_template": self.agent_refine_explain_prompt_template,
            "critic_output_format": self.critic_output_format,
            "planner_output_format": self.planner_output_format,
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
        critic_llm: BaseLanguageModel | None = None,
        answer_prompt_template: str = DEFAULT_ANSWER_PROMPT,
        critic_prompt_template: str = DEFAULT_CRITIC_PROMPT,
        trained_critic_prompt_template: str = DEFAULT_TRAINED_CRITIC_PROMPT,
        planner_prompt_template: str = DEFAULT_PLANNER_PROMPT,
        agent_system_prompt_template: str = DEFAULT_AGENT_SYSTEM_PROMPT,
        agent_user_prompt_template: str = DEFAULT_AGENT_USER_PROMPT,
        rewrite_prompt_template: str = DEFAULT_REWRITE_PROMPT,
        decomposition_prompt_template: str = DEFAULT_DECOMPOSITION_PROMPT,
        refine_prompt_template: str = DEFAULT_REFINE_PROMPT,
        agent_generate_answer_prompt_template: str = DEFAULT_AGENT_GENERATE_ANSWER_PROMPT,
        agent_rewrite_clarify_prompt_template: str = DEFAULT_AGENT_REWRITE_CLARIFY_PROMPT,
        agent_rewrite_expand_prompt_template: str = DEFAULT_AGENT_REWRITE_EXPAND_PROMPT,
        agent_rewrite_custom_prompt_template: str = DEFAULT_AGENT_REWRITE_CUSTOM_PROMPT,
        agent_decompose_prompt_template: str = DEFAULT_AGENT_DECOMPOSE_PROMPT,
        agent_refine_summarize_prompt_template: str = DEFAULT_AGENT_REFINE_SUMMARIZE_PROMPT,
        agent_refine_explain_prompt_template: str = DEFAULT_AGENT_REFINE_EXPLAIN_PROMPT,
        critic_output_format: str = "json_actions",
        planner_output_format: str = "json_actions",
        max_iterations: int = 2,
        max_actions_per_iteration: int = 4,
        schema: Any | None = None,
    ) -> None:
        """Initialize the RAG-Critic pipeline."""
        self._critic_llm = critic_llm or llm
        self._answer_prompt_template = answer_prompt_template
        self._critic_prompt_template = critic_prompt_template
        self._trained_critic_prompt_template = trained_critic_prompt_template
        self._planner_prompt_template = planner_prompt_template
        self._agent_system_prompt_template = agent_system_prompt_template
        self._agent_user_prompt_template = agent_user_prompt_template
        self._rewrite_prompt_template = rewrite_prompt_template
        self._decomposition_prompt_template = decomposition_prompt_template
        self._refine_prompt_template = refine_prompt_template
        self._agent_generate_answer_prompt_template = agent_generate_answer_prompt_template
        self._agent_rewrite_clarify_prompt_template = agent_rewrite_clarify_prompt_template
        self._agent_rewrite_expand_prompt_template = agent_rewrite_expand_prompt_template
        self._agent_rewrite_custom_prompt_template = agent_rewrite_custom_prompt_template
        self._agent_decompose_prompt_template = agent_decompose_prompt_template
        self._agent_refine_summarize_prompt_template = agent_refine_summarize_prompt_template
        self._agent_refine_explain_prompt_template = agent_refine_explain_prompt_template
        self._critic_output_format = critic_output_format
        self._planner_output_format = planner_output_format
        self._max_iterations = max_iterations
        self._max_actions_per_iteration = max_actions_per_iteration

        if self._critic_output_format not in SUPPORTED_CRITIC_OUTPUT_FORMATS:
            msg = f"Unsupported critic_output_format: {self._critic_output_format}"
            raise ValueError(msg)
        if self._planner_output_format not in SUPPORTED_PLANNER_OUTPUT_FORMATS:
            msg = f"Unsupported planner_output_format: {self._planner_output_format}"
            raise ValueError(msg)

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return RAG-Critic pipeline configuration."""
        return {
            "type": "rag_critic",
            "answer_prompt_template": self._answer_prompt_template,
            "critic_prompt_template": self._critic_prompt_template,
            "trained_critic_prompt_template": self._trained_critic_prompt_template,
            "planner_prompt_template": self._planner_prompt_template,
            "agent_system_prompt_template": self._agent_system_prompt_template,
            "agent_user_prompt_template": self._agent_user_prompt_template,
            "rewrite_prompt_template": self._rewrite_prompt_template,
            "decomposition_prompt_template": self._decomposition_prompt_template,
            "refine_prompt_template": self._refine_prompt_template,
            "agent_generate_answer_prompt_template": self._agent_generate_answer_prompt_template,
            "agent_rewrite_clarify_prompt_template": self._agent_rewrite_clarify_prompt_template,
            "agent_rewrite_expand_prompt_template": self._agent_rewrite_expand_prompt_template,
            "agent_rewrite_custom_prompt_template": self._agent_rewrite_custom_prompt_template,
            "agent_decompose_prompt_template": self._agent_decompose_prompt_template,
            "agent_refine_summarize_prompt_template": self._agent_refine_summarize_prompt_template,
            "agent_refine_explain_prompt_template": self._agent_refine_explain_prompt_template,
            "critic_output_format": self._critic_output_format,
            "planner_output_format": self._planner_output_format,
            "max_iterations": self._max_iterations,
            "max_actions_per_iteration": self._max_actions_per_iteration,
            "retrieval_pipeline_id": self._retrieval_pipeline.pipeline_id,
        }

    @staticmethod
    def _parse_json_payload(text: str) -> JsonPayload:
        """Parse structured JSON responses as either objects or arrays."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        json_match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
        if json_match:
            cleaned = json_match.group(1)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            payload = ast.literal_eval(cleaned)
            if isinstance(payload, dict | list):
                return payload
            raise

    @staticmethod
    def _extract_payload_list(payload: Any, key: str) -> list[Any]:
        """Extract a list from either a top-level array or a keyed JSON object."""
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            value = payload.get(key, [])
            if isinstance(value, list):
                return value
        return []

    @staticmethod
    def _normalize_string_list(value: Any) -> list[str]:
        """Normalize a string-or-list payload into a list of non-empty strings."""
        if isinstance(value, str):
            normalized = value.strip()
            return [normalized] if normalized else []
        if isinstance(value, list):
            return [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return []

    @staticmethod
    def _deduplicate_actions(actions: list[str]) -> list[str]:
        """Deduplicate actions while preserving order."""
        seen: set[str] = set()
        result: list[str] = []
        for action in actions:
            if action in seen or action not in SUPPORTED_ACTIONS:
                continue
            seen.add(action)
            result.append(action)
        return result

    @classmethod
    def _map_trained_critic_tags_to_actions(cls, tag2_values: list[str]) -> list[str]:
        """Map published RAG-Critic error tags to local corrective actions."""
        actions: list[str] = []
        for value in tag2_values:
            actions.extend(RAG_CRITIC_3B_TAG2_ACTIONS.get(value.strip().lower(), []))
        if not actions:
            actions = ["generate_answer"]
        return cls._deduplicate_actions(actions)

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
    def _coerce_positive_int(value: Any, default: int) -> int:
        """Coerce planner-provided values to a positive integer."""
        if isinstance(value, bool):
            return default
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return default
        return max(coerced, 1)

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

    @staticmethod
    def _format_passage_list(passages: list[str]) -> str:
        """Format plain passage strings for prompting."""
        if not passages:
            return "(No documents available)"
        return "\n".join(f"Passage: {passage}" for passage in passages)

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

    async def _invoke_model_and_record(
        self,
        model: BaseLanguageModel,
        prompt: str,
        tracker: TokenUsageTracker,
    ) -> str:
        """Invoke the provided model and record token usage."""
        response = await model.ainvoke(prompt)
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
        if self._critic_output_format == "rag_critic_tags":
            return await self._run_trained_critic(query, context, answer, tracker)

        prompt = self._critic_prompt_template.format(query=query, context=context, answer=answer)
        critique_text = await self._invoke_and_record(prompt, tracker)
        try:
            critique = self._parse_json_payload(critique_text)
        except (SyntaxError, ValueError, json.JSONDecodeError):
            logger.warning("Critic output was not valid JSON; falling back to revise verdict")
            critique = {"verdict": "revise", "feedback": critique_text, "recommended_actions": ["generate_answer"]}
        if not isinstance(critique, dict):
            logger.warning("Critic output must be a JSON object; falling back to revise verdict")
            critique = {"verdict": "revise", "feedback": critique_text, "recommended_actions": ["generate_answer"]}
        critique.setdefault("feedback", "")
        critique["recommended_actions"] = self._normalize_string_list(critique.get("recommended_actions", []))
        return critique

    async def _run_trained_critic(
        self,
        query: str,
        context: str,
        answer: str,
        tracker: TokenUsageTracker,
    ) -> dict[str, Any]:
        """Run the published RAG-Critic model and adapt its tags to local actions."""
        prompt = self._trained_critic_prompt_template.format(query=query, context=context, answer=answer)
        response = await self._critic_llm.ainvoke(prompt)
        tracker.record(response)
        critique_text = response.content if hasattr(response, "content") else str(response)
        try:
            payload = self._parse_json_payload(critique_text)
        except (SyntaxError, ValueError, json.JSONDecodeError):
            logger.warning("Trained critic output was not parseable; falling back to revise verdict")
            return {"verdict": "revise", "feedback": critique_text, "recommended_actions": ["generate_answer"]}

        if not isinstance(payload, dict):
            logger.warning("Trained critic output must be a JSON object; falling back to revise verdict")
            return {"verdict": "revise", "feedback": critique_text, "recommended_actions": ["generate_answer"]}

        judgement = str(payload.get("Judgement", payload.get("judgement", ""))).strip().lower()
        if judgement == "correct":
            return {
                "verdict": "approved",
                "feedback": str(payload.get("Error_analysis", payload.get("error_analysis", ""))).strip(),
                "recommended_actions": [],
                "tag1": self._normalize_string_list(payload.get("tag1", [])),
                "tag2": self._normalize_string_list(payload.get("tag2", [])),
                "tag3": self._normalize_string_list(payload.get("tag3", [])),
            }

        tag1 = self._normalize_string_list(payload.get("tag1", []))
        tag2 = self._normalize_string_list(payload.get("tag2", []))
        tag3 = self._normalize_string_list(payload.get("tag3", []))
        feedback = str(payload.get("Error_analysis", payload.get("error_analysis", critique_text))).strip()
        return {
            "verdict": "revise",
            "feedback": feedback,
            "recommended_actions": self._map_trained_critic_tags_to_actions(tag2),
            "tag1": tag1,
            "tag2": tag2,
            "tag3": tag3,
        }

    @staticmethod
    def _extract_python_code_block(text: str) -> str:
        """Extract Python code from fenced or unfenced planner output."""
        cleaned = text.strip()
        if "```python" in cleaned:
            return cleaned.split("```python", maxsplit=1)[1].split("```", maxsplit=1)[0].strip()
        if "```" in cleaned:
            return cleaned.split("```", maxsplit=1)[1].split("```", maxsplit=1)[0].strip()
        return cleaned

    async def _agent_rewrite_query(
        self,
        query: str,
        instruction: str,
        tracker: TokenUsageTracker,
    ) -> list[str]:
        """Rewrite a query using the official-style agent prompt family."""
        normalized_instruction = instruction.strip().lower()
        if normalized_instruction == "clarify":
            prompt = self._agent_rewrite_clarify_prompt_template.format(query=query)
        elif normalized_instruction == "expand":
            prompt = self._agent_rewrite_expand_prompt_template.format(query=query)
        else:
            prompt = self._agent_rewrite_custom_prompt_template.format(query=query, instruction=instruction)

        text = await self._invoke_and_record(prompt, tracker)
        try:
            payload = self._parse_json_payload(text)
        except (SyntaxError, ValueError, json.JSONDecodeError):
            return [query]

        if isinstance(payload, dict):
            queries = self._normalize_string_list(payload.get("query", []))
        else:
            queries = self._normalize_string_list(payload)
        return queries or [query]

    async def _agent_decompose_query(self, query: str, tracker: TokenUsageTracker) -> list[str]:
        """Decompose a query using the official-style agent prompt."""
        text = await self._invoke_and_record(self._agent_decompose_prompt_template.format(query=query), tracker)
        try:
            payload = self._parse_json_payload(text)
        except (SyntaxError, ValueError, json.JSONDecodeError):
            return [query]
        return self._normalize_string_list(payload)

    async def _agent_refine_doc(
        self,
        query: str,
        document: str,
        instruction: str,
        tracker: TokenUsageTracker,
    ) -> str:
        """Refine a document with the official-style summarize/explain prompts."""
        normalized_instruction = instruction.strip().lower()
        if normalized_instruction == "explain":
            prompt = self._agent_refine_explain_prompt_template.format(document=document, question=query)
        elif normalized_instruction == "summarize":
            prompt = self._agent_refine_summarize_prompt_template.format(document=document, question=query)
        else:
            return document

        text = await self._invoke_and_record(prompt, tracker)
        try:
            payload = self._parse_json_payload(text)
        except (SyntaxError, ValueError, json.JSONDecodeError):
            return document

        if isinstance(payload, dict):
            return str(payload.get("refined_document", payload.get("explanation", document))).strip() or document
        return document

    async def _agent_generate_answer(
        self,
        query: str,
        passages: list[str],
        additional_instruction: str,
        tracker: TokenUsageTracker,
    ) -> str:
        """Generate an answer with the official execute-agent response style."""
        prompt = self._agent_generate_answer_prompt_template.format(
            instruction=additional_instruction or "(none)",
            query=query,
            context=self._format_passage_list(passages),
        )
        return await self._invoke_and_record(prompt, tracker)

    async def _plan_agent_code(
        self,
        query: str,
        doc_list: list[str],
        previous_answer: str,
        critique: dict[str, Any],
        tracker: TokenUsageTracker,
    ) -> str:
        """Generate official-style Python function-call code for corrective execution."""
        error_type = critique.get("tag2") or critique.get("recommended_actions") or critique.get("feedback", "")
        if isinstance(error_type, list):
            formatted_error_type = ", ".join(str(item) for item in error_type)
        else:
            formatted_error_type = str(error_type)
        prompt = (
            self._agent_system_prompt_template.strip()
            + "\n\n"
            + self._agent_user_prompt_template.format(
                question=query,
                doc_list=repr(doc_list),
                previous_pred=previous_answer,
                error_type=formatted_error_type or "Unknown Error",
            )
        )
        return self._extract_python_code_block(await self._invoke_and_record(prompt, tracker))

    @staticmethod
    def _validate_agent_code(tree: ast.Module) -> None:
        """Validate that generated planner code stays within a narrow safe subset."""
        allowed_calls = {"Retrieval", "RewriteQuery", "DecomposeQuery", "RefineDoc", "GenerateAnswer"}
        allowed_nodes = (
            ast.Module,
            ast.Assign,
            ast.Expr,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Store,
            ast.Constant,
            ast.List,
            ast.Tuple,
            ast.Subscript,
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                msg = f"Unsupported node in agent plan: {type(node).__name__}"
                raise TypeError(msg)
            if isinstance(node, ast.Call) and (
                not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls
            ):
                msg = "Agent plan may only call Retrieval, RewriteQuery, DecomposeQuery, RefineDoc, GenerateAnswer"
                raise ValueError(msg)
            if isinstance(node, ast.Assign) and (len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name)):
                msg = "Agent plan assignments must target a single variable name"
                raise ValueError(msg)

    async def _eval_agent_expr(  # noqa: C901
        self,
        node: ast.AST,
        namespace: dict[str, Any],
        execution_state: dict[str, Any],
        tracker: TokenUsageTracker,
        default_top_k: int,
    ) -> Any:
        """Evaluate a restricted AST expression for official-style agent plans."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in namespace:
                return namespace[node.id]
            msg = f"Unknown variable in agent plan: {node.id}"
            raise ValueError(msg)
        if isinstance(node, ast.List):
            return [
                await self._eval_agent_expr(element, namespace, execution_state, tracker, default_top_k)
                for element in node.elts
            ]
        if isinstance(node, ast.Tuple):
            return tuple([
                await self._eval_agent_expr(element, namespace, execution_state, tracker, default_top_k)
                for element in node.elts
            ])
        if isinstance(node, ast.Subscript):
            value = await self._eval_agent_expr(node.value, namespace, execution_state, tracker, default_top_k)
            index = await self._eval_agent_expr(node.slice, namespace, execution_state, tracker, default_top_k)
            return value[index]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = [
                await self._eval_agent_expr(arg, namespace, execution_state, tracker, default_top_k)
                for arg in node.args
            ]
            kwargs = {
                keyword.arg: await self._eval_agent_expr(
                    keyword.value, namespace, execution_state, tracker, default_top_k
                )
                for keyword in node.keywords
                if keyword.arg is not None
            }
            func_name = node.func.id
            if func_name == "Retrieval":
                query = str(args[0])
                topk = self._coerce_positive_int(
                    args[1] if len(args) > 1 else kwargs.get("topk", default_top_k), default_top_k
                )
                hydrated_documents = self._hydrate_documents(await self._retrieval_pipeline.retrieve(query, topk))
                execution_state["retrieval_history"].append({
                    "query": query,
                    "query_source": "agent_plan",
                    "strategy": "replace",
                    "returned_doc_ids": [doc["doc_id"] for doc in hydrated_documents],
                })
                execution_state["documents"] = hydrated_documents
                execution_state["working_context"] = self._format_documents(hydrated_documents)
                execution_state["executed_actions"].append({"action": "retrieval", "query": query, "top_k": topk})
                return [doc["content"] for doc in hydrated_documents]
            if func_name == "RewriteQuery":
                rewritten_queries = await self._agent_rewrite_query(str(args[0]), str(args[1]), tracker)
                execution_state["rewritten_queries"].extend([
                    query for query in rewritten_queries if query not in execution_state["rewritten_queries"]
                ])
                execution_state["executed_actions"].append({
                    "action": "rewrite_query",
                    "instruction": str(args[1]),
                    "queries": rewritten_queries,
                })
                return rewritten_queries
            if func_name == "DecomposeQuery":
                sub_questions = await self._agent_decompose_query(str(args[0]), tracker)
                execution_state["sub_questions"].extend([
                    question for question in sub_questions if question not in execution_state["sub_questions"]
                ])
                execution_state["executed_actions"].append({
                    "action": "decompose_query",
                    "sub_questions": sub_questions,
                })
                return sub_questions
            if func_name == "RefineDoc":
                refined_document = await self._agent_refine_doc(str(args[0]), str(args[1]), str(args[2]), tracker)
                execution_state["executed_actions"].append({"action": "refine_documents", "instruction": str(args[2])})
                return refined_document
            if func_name == "GenerateAnswer":
                docs = [str(doc) for doc in args[1]]
                instruction = str(args[2]) if len(args) > 2 else str(kwargs.get("additional_instruction", ""))
                answer = await self._agent_generate_answer(str(args[0]), docs, instruction, tracker)
                execution_state["executed_actions"].append({"action": "generate_answer", "instruction": instruction})
                execution_state["answer_generated"] = True
                return answer
        msg = f"Unsupported expression in agent plan: {ast.dump(node, include_attributes=False)}"
        raise ValueError(msg)

    async def _execute_agent_code_plan(
        self,
        code_snippet: str,
        *,
        original_query: str,
        documents: list[dict[str, Any]],
        working_context: str,
        answer: str,
        rewritten_queries: list[str],
        sub_questions: list[str],
        executed_actions: list[dict[str, Any]],
        retrieval_history: list[dict[str, Any]],
        top_k: int,
        tracker: TokenUsageTracker,
    ) -> tuple[str, list[dict[str, Any]], str, bool]:
        """Execute official-style planner code via a restricted AST evaluator."""
        tree = ast.parse(code_snippet, mode="exec")
        self._validate_agent_code(tree)
        execution_state = {
            "documents": documents,
            "working_context": working_context,
            "executed_actions": executed_actions,
            "retrieval_history": retrieval_history,
            "rewritten_queries": rewritten_queries,
            "sub_questions": sub_questions,
            "answer_generated": False,
        }
        namespace: dict[str, Any] = {
            "question": original_query,
            "doc_list": [document["content"] for document in documents],
            "previous_pred": answer,
        }

        for statement in tree.body:
            if isinstance(statement, ast.Assign):
                target = cast(ast.Name, statement.targets[0])
                namespace[target.id] = await self._eval_agent_expr(
                    statement.value,
                    namespace,
                    execution_state,
                    tracker,
                    top_k,
                )
            elif isinstance(statement, ast.Expr):
                await self._eval_agent_expr(statement.value, namespace, execution_state, tracker, top_k)

        final_answer = namespace.get("final_answer")
        final_documents = cast(list[dict[str, Any]], execution_state["documents"])
        final_working_context = cast(str, execution_state["working_context"])
        answer_generated = cast(bool, execution_state["answer_generated"])
        if not isinstance(final_answer, str) or not final_answer.strip():
            return answer, final_documents, final_working_context, False
        return (
            final_answer,
            final_documents,
            final_working_context,
            answer_generated,
        )

    async def _plan_actions(
        self,
        query: str,
        doc_list: list[str],
        answer: str,
        critique: dict[str, Any],
        tracker: TokenUsageTracker,
    ) -> list[dict[str, Any]]:
        """Plan corrective actions from the critic feedback."""
        if self._planner_output_format == "python_agent":
            return [
                {
                    "action": "agent_code",
                    "code": await self._plan_agent_code(query, doc_list, answer, critique, tracker),
                }
            ]

        prompt = self._planner_prompt_template.format(
            query=query,
            answer=answer,
            critique=json.dumps(critique, ensure_ascii=False),
        )
        plan_text = await self._invoke_and_record(prompt, tracker)
        try:
            payload = self._parse_json_payload(plan_text)
        except (SyntaxError, ValueError, json.JSONDecodeError):
            payload = {"actions": self._normalize_string_list(critique.get("recommended_actions", []))}
        raw_actions = self._extract_payload_list(payload, "actions")
        actions: list[dict[str, Any]] = []
        for item in raw_actions:
            if isinstance(item, str):
                actions.append({"action": item})
            elif isinstance(item, dict):
                actions.append(item)
        if not actions:
            actions = [
                {"action": action} for action in self._normalize_string_list(critique.get("recommended_actions", []))
            ]
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
        return rewritten.strip() or query

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
        except (SyntaxError, ValueError, json.JSONDecodeError):
            payload = {"sub_questions": [part.strip() for part in decomposition_text.split("\n") if part.strip()]}
        sub_questions = self._extract_payload_list(payload, "sub_questions")
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
        action_top_k = self._coerce_positive_int(raw_action.get("top_k", top_k), top_k)
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
        answer: str,
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

        if action_name == "agent_code":
            executed_actions.append({"action": "agent_code", "code": str(raw_action.get("code", ""))})
            answer, documents, working_context, answer_updated = await self._execute_agent_code_plan(
                str(raw_action.get("code", "")),
                original_query=original_query,
                documents=documents,
                working_context=working_context,
                answer=answer,
                rewritten_queries=rewritten_queries,
                sub_questions=sub_questions,
                executed_actions=executed_actions,
                retrieval_history=retrieval_history,
                top_k=top_k,
                tracker=tracker,
            )
            return {
                "answer": answer,
                "documents": documents,
                "working_context": working_context,
                "answer_updated": answer_updated,
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
                answer=answer,
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
            actions = await self._plan_actions(
                original_query,
                [document["content"] for document in documents],
                answer,
                critique,
                tracker,
            )
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
                "critic_output_format": self._critic_output_format,
                "planner_output_format": self._planner_output_format,
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
