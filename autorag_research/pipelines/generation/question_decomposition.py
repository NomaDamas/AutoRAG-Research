"""Question Decomposition generation pipeline for AutoRAG-Research.

Implements a decomposition-first RAG strategy:
1. Decompose the original query into sub-questions
2. Retrieve for the original query and each sub-question
3. Merge and deduplicate retrieved documents by doc_id using the highest score
4. Keep the final top-k documents
5. Generate the answer from the merged context
"""

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

DEFAULT_DECOMPOSITION_PROMPT = """You are decomposing a question for retrieval-augmented generation.

Question: {query}

Write up to {max_subquestions} short, standalone sub-questions that would help retrieve evidence.
Return one sub-question per line and no other text.
If decomposition is unnecessary, repeat the original question once.

Sub-questions:"""

DEFAULT_QA_PROMPT = """Answer the following question using the provided paragraphs.

Question: {query}

Paragraphs:
{paragraphs}

Provide a concise, direct answer to the question based on the information in the paragraphs.

Answer:"""

_SUBQUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:sub-?question\s*\d*\s*:|question\s*\d*\s*:|[-*•]|\d+[.)]|[A-Za-z][.)])\s*",
    re.IGNORECASE,
)


class QuestionDecompositionPipeline(BaseGenerationPipeline):
    """Generation pipeline that retrieves over decomposed sub-questions."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: "BaseLanguageModel",
        retrieval_pipeline: "BaseRetrievalPipeline",
        decomposition_prompt_template: str = DEFAULT_DECOMPOSITION_PROMPT,
        qa_prompt_template: str = DEFAULT_QA_PROMPT,
        max_subquestions: int = 3,
        schema: Any | None = None,
    ):
        """Initialize Question Decomposition pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            llm: LangChain BaseLanguageModel instance for decomposition and generation.
            retrieval_pipeline: Retrieval pipeline for fetching relevant context.
            decomposition_prompt_template: Template for decomposition prompt.
                Must contain {query}; may optionally contain {max_subquestions}.
            qa_prompt_template: Template for the final QA prompt.
                Must contain {query} and {paragraphs}.
            max_subquestions: Maximum number of unique sub-questions to retrieve for.
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        # IMPORTANT: Store parameters BEFORE calling super().__init__
        # because _get_pipeline_config() is called during base init
        self._decomposition_prompt_template = decomposition_prompt_template
        self._qa_prompt_template = qa_prompt_template
        self.max_subquestions = max_subquestions

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return Question Decomposition pipeline configuration for storage."""
        model_name = getattr(self._llm, "model_name", None)
        if model_name is None or not isinstance(model_name, str):
            model_name = type(self._llm).__name__

        return {
            "type": "question_decomposition",
            "decomposition_prompt_template": self._decomposition_prompt_template,
            "qa_prompt_template": self._qa_prompt_template,
            "max_subquestions": self.max_subquestions,
            "retrieval_pipeline_id": self._retrieval_pipeline.pipeline_id,
            "llm_model": model_name,
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text content from a LangChain response or raw string."""
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def _normalize_question(question: str) -> str:
        """Normalize question text for duplicate detection."""
        return " ".join(question.lower().split())

    @staticmethod
    def _normalize_score(score: Any) -> float:
        """Normalize retrieval score values for ranking."""
        if isinstance(score, (int, float)):
            return float(score)
        return 0.0

    def _build_decomposition_prompt(self, query: str) -> str:
        """Build the decomposition prompt."""
        return self._decomposition_prompt_template.format(
            query=query,
            max_subquestions=self.max_subquestions,
        )

    def _build_qa_prompt(self, query: str, paragraphs: list[str]) -> str:
        """Build prompt for final answer generation."""
        if paragraphs:
            numbered_paragraphs = "\n\n".join([f"[{i + 1}] {p}" for i, p in enumerate(paragraphs)])
        else:
            numbered_paragraphs = "(No paragraphs available)"

        return self._qa_prompt_template.format(
            query=query,
            paragraphs=numbered_paragraphs,
        )

    def _parse_subquestions(self, decomposition_text: str) -> list[str]:
        """Parse sub-questions from the decomposition model output."""
        text = decomposition_text.strip()
        if not text:
            return []

        candidates = [line.strip() for line in text.splitlines() if line.strip()]

        if len(candidates) == 1:
            inline_numbered = re.split(r"\s+(?=(?:\d+[.)]|[-*•])\s*)", candidates[0])
            if len(inline_numbered) > 1:
                candidates = [item.strip() for item in inline_numbered if item.strip()]
            elif candidates[0].count("?") > 1:
                candidates = [item.strip() for item in re.split(r"(?<=\?)\s+", candidates[0]) if item.strip()]

        subquestions: list[str] = []
        for candidate in candidates:
            cleaned = _SUBQUESTION_PREFIX_RE.sub("", candidate).strip()
            cleaned = cleaned.rstrip(" ;")
            if cleaned:
                subquestions.append(cleaned)

        return subquestions

    def _deduplicate_subquestions(self, query: str, subquestions: list[str]) -> list[str]:
        """Drop duplicate sub-questions and the original query."""
        seen = {self._normalize_question(query)}
        unique_subquestions: list[str] = []

        for subquestion in subquestions:
            normalized = self._normalize_question(subquestion)
            if not normalized or normalized in seen:
                continue

            seen.add(normalized)
            unique_subquestions.append(subquestion)

            if len(unique_subquestions) >= self.max_subquestions:
                break

        return unique_subquestions

    def _merge_results(self, result_sets: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Merge retrieved results by doc_id, keeping the highest score."""
        merged: dict[int | str, dict[str, Any]] = {}

        for result_set in result_sets:
            for result in result_set:
                doc_id = result.get("doc_id")
                if doc_id is None:
                    continue

                score = self._normalize_score(result.get("score"))
                existing = merged.get(doc_id)
                if existing is None or score > self._normalize_score(existing.get("score")):
                    merged[doc_id] = {
                        "doc_id": doc_id,
                        "score": score,
                    }

        return sorted(
            merged.values(),
            key=lambda item: (-self._normalize_score(item.get("score")), str(item.get("doc_id"))),
        )

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate an answer using question decomposition."""
        query_text = self._service.get_query_text(query_id)
        tracker = TokenUsageTracker()

        logger.debug(f"Question Decomposition: decomposing query ID {query_id}")
        decomposition_prompt = self._build_decomposition_prompt(query_text)
        decomposition_response = await self._llm.ainvoke(decomposition_prompt)
        decomposition_text = self._extract_text(decomposition_response)
        tracker.record(decomposition_response)

        subquestions = self._deduplicate_subquestions(
            query_text,
            self._parse_subquestions(decomposition_text),
        )

        result_sets: list[list[dict[str, Any]]] = [
            await self._retrieval_pipeline._retrieve_by_id(query_id, top_k),
        ]

        for subquestion in subquestions:
            result_sets.append(await self._retrieval_pipeline.retrieve(subquestion, top_k))

        merged_results = self._merge_results(result_sets)
        final_results = merged_results[:top_k]
        chunk_ids = [result["doc_id"] for result in final_results]
        paragraphs = self._service.get_chunk_contents(chunk_ids) if chunk_ids else []

        qa_prompt = self._build_qa_prompt(query_text, paragraphs)
        qa_response = await self._llm.ainvoke(qa_prompt)
        answer_text = self._extract_text(qa_response)
        tracker.record(qa_response)

        return GenerationResult(
            text=answer_text,
            token_usage=tracker.total,
            metadata={
                "raw_decomposition": decomposition_text,
                "sub_questions": subquestions,
                "retrieval_queries": [query_text, *subquestions],
                "retrieved_chunk_ids": chunk_ids,
                "retrieved_scores": [result["score"] for result in final_results],
            },
        )


@dataclass(kw_only=True)
class QuestionDecompositionPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for Question Decomposition generation pipeline."""

    decomposition_prompt_template: str = field(default=DEFAULT_DECOMPOSITION_PROMPT)
    qa_prompt_template: str = field(default=DEFAULT_QA_PROMPT)
    max_subquestions: int = 3

    def get_pipeline_class(self) -> type["QuestionDecompositionPipeline"]:
        """Return the QuestionDecompositionPipeline class."""
        return QuestionDecompositionPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for QuestionDecompositionPipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)

        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "decomposition_prompt_template": self.decomposition_prompt_template,
            "qa_prompt_template": self.qa_prompt_template,
            "max_subquestions": self.max_subquestions,
        }
