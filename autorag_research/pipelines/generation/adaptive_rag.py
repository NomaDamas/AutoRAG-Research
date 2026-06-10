"""AdaptiveRAG generation pipeline for AutoRAG-Research.

AdaptiveRAG routes each query to a retrieval strategy based on predicted complexity:
- zero: answer directly without retrieval
- single: single-pass retrieval then generation
- multi: IRCoT-style interleaved reasoning and retrieval
"""

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

DEFAULT_COMPLEXITY_PROMPT_TEMPLATE = """Classify the following question complexity as exactly one of: simple, moderate, complex.

- simple: can be answered directly without retrieval
- moderate: needs one retrieval pass
- complex: needs iterative multi-step retrieval

Question: {query}

Return only one word: simple, moderate, or complex."""

DEFAULT_ZERO_RETRIEVAL_PROMPT_TEMPLATE = """Answer the question directly.

Question: {query}

Answer:"""

DEFAULT_SINGLE_RETRIEVAL_PROMPT_TEMPLATE = """Answer the question with the provided context.

Context:
{context}

Question: {query}

Answer:"""

DEFAULT_MULTI_REASONING_PROMPT_TEMPLATE = """You are answering a multi-step question using chain-of-thought reasoning.

Question: {query}

Available Paragraphs:
{paragraphs}

Previous Thoughts:
{cot_history}

Generate the next reasoning step. Think step-by-step about what information you need or what conclusion you can draw.

If you have enough information to answer the question, write: "The answer is: [your answer]"

Next Thought:"""

DEFAULT_MULTI_RETRIEVAL_ANSWER_PROMPT_TEMPLATE = """Answer the question using the collected evidence.

Question: {query}

Collected Context:
{context}

Chain-of-Thought History:
{cot_history}

Answer:"""


@dataclass(kw_only=True)
class AdaptiveRAGPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for AdaptiveRAG generation pipeline."""

    complexity_prompt_template: str = field(default=DEFAULT_COMPLEXITY_PROMPT_TEMPLATE)
    zero_retrieval_prompt_template: str = field(default=DEFAULT_ZERO_RETRIEVAL_PROMPT_TEMPLATE)
    single_retrieval_prompt_template: str = field(default=DEFAULT_SINGLE_RETRIEVAL_PROMPT_TEMPLATE)
    multi_reasoning_prompt_template: str = field(default=DEFAULT_MULTI_REASONING_PROMPT_TEMPLATE)
    multi_retrieval_answer_prompt_template: str = field(default=DEFAULT_MULTI_RETRIEVAL_ANSWER_PROMPT_TEMPLATE)
    route_for_simple: Literal["zero", "single", "multi"] = "zero"
    route_for_moderate: Literal["zero", "single", "multi"] = "single"
    route_for_complex: Literal["zero", "single", "multi"] = "multi"
    max_multi_steps: int = 2
    answer_signal: str = "answer is:"
    paragraph_budget: int = 15

    def get_pipeline_class(self) -> type["AdaptiveRAGPipeline"]:
        """Return the AdaptiveRAGPipeline class."""
        return AdaptiveRAGPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for AdaptiveRAGPipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)

        route_map = AdaptiveRAGPipeline._validate_route_map(
            route_for_simple=self.route_for_simple,
            route_for_moderate=self.route_for_moderate,
            route_for_complex=self.route_for_complex,
        )

        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "complexity_prompt_template": self.complexity_prompt_template,
            "zero_retrieval_prompt_template": self.zero_retrieval_prompt_template,
            "single_retrieval_prompt_template": self.single_retrieval_prompt_template,
            "multi_reasoning_prompt_template": self.multi_reasoning_prompt_template,
            "multi_retrieval_answer_prompt_template": self.multi_retrieval_answer_prompt_template,
            "route_for_simple": route_map["simple"],
            "route_for_moderate": route_map["moderate"],
            "route_for_complex": route_map["complex"],
            "max_multi_steps": self.max_multi_steps,
            "answer_signal": self.answer_signal,
            "paragraph_budget": self.paragraph_budget,
        }


class AdaptiveRAGPipeline(BaseGenerationPipeline):
    """Adaptive routing generation pipeline with zero/single/multi retrieval modes."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        complexity_prompt_template: str = DEFAULT_COMPLEXITY_PROMPT_TEMPLATE,
        zero_retrieval_prompt_template: str = DEFAULT_ZERO_RETRIEVAL_PROMPT_TEMPLATE,
        single_retrieval_prompt_template: str = DEFAULT_SINGLE_RETRIEVAL_PROMPT_TEMPLATE,
        multi_reasoning_prompt_template: str = DEFAULT_MULTI_REASONING_PROMPT_TEMPLATE,
        multi_retrieval_answer_prompt_template: str = DEFAULT_MULTI_RETRIEVAL_ANSWER_PROMPT_TEMPLATE,
        route_for_simple: Literal["zero", "single", "multi"] = "zero",
        route_for_moderate: Literal["zero", "single", "multi"] = "single",
        route_for_complex: Literal["zero", "single", "multi"] = "multi",
        max_multi_steps: int = 2,
        answer_signal: str = "answer is:",
        paragraph_budget: int = 15,
        schema: Any | None = None,
    ):
        """Initialize AdaptiveRAG pipeline."""
        self._complexity_prompt_template = complexity_prompt_template
        self._zero_retrieval_prompt_template = zero_retrieval_prompt_template
        self._single_retrieval_prompt_template = single_retrieval_prompt_template
        self._multi_reasoning_prompt_template = multi_reasoning_prompt_template
        self._multi_retrieval_answer_prompt_template = multi_retrieval_answer_prompt_template
        route_map = self._validate_route_map(
            route_for_simple=route_for_simple,
            route_for_moderate=route_for_moderate,
            route_for_complex=route_for_complex,
        )
        self._route_for_simple = route_map["simple"]
        self._route_for_moderate = route_map["moderate"]
        self._route_for_complex = route_map["complex"]
        self._max_multi_steps = max_multi_steps
        self._answer_signal = answer_signal
        self._paragraph_budget = paragraph_budget

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return AdaptiveRAG pipeline configuration."""
        model_name = getattr(self._llm, "model_name", None)
        if model_name is None or not isinstance(model_name, str):
            model_name = type(self._llm).__name__

        return {
            "type": "adaptive_rag",
            "complexity_prompt_template": self._complexity_prompt_template,
            "zero_retrieval_prompt_template": self._zero_retrieval_prompt_template,
            "single_retrieval_prompt_template": self._single_retrieval_prompt_template,
            "multi_reasoning_prompt_template": self._multi_reasoning_prompt_template,
            "multi_retrieval_answer_prompt_template": self._multi_retrieval_answer_prompt_template,
            "route_for_simple": self._route_for_simple,
            "route_for_moderate": self._route_for_moderate,
            "route_for_complex": self._route_for_complex,
            "max_multi_steps": self._max_multi_steps,
            "answer_signal": self._answer_signal,
            "paragraph_budget": self._paragraph_budget,
            "retrieval_pipeline_id": self._retrieval_pipeline.pipeline_id,
            "llm_model": model_name,
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text content from an LLM response."""
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def _normalize_score(score: Any) -> float:
        """Normalize retrieval score to float."""
        if isinstance(score, (int, float)):
            return float(score)
        return 0.0

    @staticmethod
    def _parse_complexity_tier(response_text: str) -> Literal["simple", "moderate", "complex"]:
        """Parse complexity output, defaulting to moderate for unknown outputs.

        The classifier prompt asks for one label, but LLMs can still return explanatory text.
        Match labels as whole tokens and choose the safest mentioned tier when output is ambiguous.
        """
        normalized = response_text.strip().lower()
        tier_tokens = {
            "moderate" if label == "medium" else label
            for label in re.findall(r"\b(simple|moderate|medium|complex)\b", normalized)
        }

        if "complex" in tier_tokens:
            return "complex"
        if "moderate" in tier_tokens:
            return "moderate"
        if "simple" in tier_tokens:
            return "simple"

        return "moderate"

    @staticmethod
    def _normalize_route(route: str, field_name: str = "route") -> Literal["zero", "single", "multi"]:
        """Normalize and validate route labels to supported route names."""
        normalized = route.strip().lower()
        if normalized == "zero":
            return "zero"
        if normalized == "single":
            return "single"
        if normalized == "multi":
            return "multi"

        msg = f"{field_name} must be one of: zero, single, multi (got {route!r})"
        raise ValueError(msg)

    @classmethod
    def _validate_route_map(
        cls,
        *,
        route_for_simple: str,
        route_for_moderate: str,
        route_for_complex: str,
    ) -> dict[Literal["simple", "moderate", "complex"], Literal["zero", "single", "multi"]]:
        """Validate tier-specific route configuration and return normalized routes."""
        return {
            "simple": cls._normalize_route(route_for_simple, "route_for_simple"),
            "moderate": cls._normalize_route(route_for_moderate, "route_for_moderate"),
            "complex": cls._normalize_route(route_for_complex, "route_for_complex"),
        }

    def _select_route(
        self, complexity_tier: Literal["simple", "moderate", "complex"]
    ) -> Literal["zero", "single", "multi"]:
        """Select route name from tier-specific mapping."""
        route_map: dict[Literal["simple", "moderate", "complex"], Literal["zero", "single", "multi"]] = {
            "simple": self._route_for_simple,
            "moderate": self._route_for_moderate,
            "complex": self._route_for_complex,
        }
        return route_map[complexity_tier]

    @staticmethod
    def _extract_first_sentence(text: str) -> str:
        """Extract the first sentence from generated text."""
        text = text.strip()
        if not text:
            return text

        for delimiter in [". ", "! ", "? "]:
            if delimiter in text:
                return text.split(delimiter, 1)[0] + delimiter[0]

        return text

    @staticmethod
    def _format_numbered_paragraphs(paragraphs: list[str]) -> str:
        """Format ordered paragraphs for prompts."""
        if not paragraphs:
            return "(No paragraphs available)"
        return "\n\n".join(f"[{index + 1}] {paragraph}" for index, paragraph in enumerate(paragraphs))

    @staticmethod
    def _format_cot_history(cot_sentences: list[str]) -> str:
        """Format chain-of-thought history for prompts."""
        if not cot_sentences:
            return "(No previous thoughts)"
        return "\n".join(f"Thought {index + 1}: {sentence}" for index, sentence in enumerate(cot_sentences))

    def _build_context(self, chunk_ids: list[int | str]) -> str:
        """Build context text from chunk IDs."""
        if not chunk_ids:
            return ""
        chunk_contents = self._service.get_chunk_contents(chunk_ids)
        return "\n\n".join(content for content in chunk_contents if content)

    def _append_ordered_paragraphs(
        self,
        retrieved_results: list[dict[str, Any]],
        chunk_ids: list[int | str],
        scores: list[float],
        paragraphs: list[str],
    ) -> None:
        """Append newly retrieved paragraphs in retrieval order, deduping by chunk ID and applying FIFO budget."""
        seen_chunk_ids: set[int | str] = set(chunk_ids)
        new_results: list[dict[str, Any]] = []
        for result in retrieved_results:
            doc_id = result.get("doc_id")
            if doc_id is None or doc_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(doc_id)
            new_results.append(result)
        new_chunk_ids = [result["doc_id"] for result in new_results]
        if not new_chunk_ids:
            return

        new_contents = self._service.get_chunk_contents(new_chunk_ids)
        for result, content in zip(new_results, new_contents, strict=False):
            if not content:
                continue
            chunk_ids.append(result["doc_id"])
            scores.append(self._normalize_score(result.get("score")))
            paragraphs.append(content)

        overflow = len(paragraphs) - self._paragraph_budget
        if overflow > 0:
            del chunk_ids[:overflow]
            del scores[:overflow]
            del paragraphs[:overflow]

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate answer using adaptive route selection."""
        query_text = self._service.get_query_text(query_id)
        tracker = TokenUsageTracker()

        complexity_prompt = self._complexity_prompt_template.format(query=query_text)
        complexity_response = await self._llm.ainvoke(complexity_prompt)
        tracker.record(complexity_response)
        complexity_tier = self._parse_complexity_tier(self._extract_text(complexity_response))
        route = self._select_route(complexity_tier)

        if route == "zero":
            prompt = self._zero_retrieval_prompt_template.format(query=query_text)
            answer_response = await self._llm.ainvoke(prompt)
            tracker.record(answer_response)
            answer_text = self._extract_text(answer_response)

            return GenerationResult(
                text=answer_text,
                token_usage=tracker.total,
                metadata={
                    "complexity_tier": complexity_tier,
                    "route": route,
                    "retrieved_chunk_ids": [],
                    "retrieved_scores": [],
                    "follow_up_queries": [],
                },
            )

        if route == "single":
            retrieved_results = await self._retrieval_pipeline._retrieve_by_id(query_id, top_k)
            chunk_ids = [item["doc_id"] for item in retrieved_results if item.get("doc_id") is not None]
            context = self._build_context(chunk_ids)
            prompt = self._single_retrieval_prompt_template.format(context=context, query=query_text)
            answer_response = await self._llm.ainvoke(prompt)
            tracker.record(answer_response)
            answer_text = self._extract_text(answer_response)

            return GenerationResult(
                text=answer_text,
                token_usage=tracker.total,
                metadata={
                    "complexity_tier": complexity_tier,
                    "route": route,
                    "retrieved_chunk_ids": chunk_ids,
                    "retrieved_scores": [self._normalize_score(item.get("score")) for item in retrieved_results],
                    "follow_up_queries": [],
                },
            )

        # multi route: Adaptive-RAG route C follows IRCoT-style interleaved reasoning and retrieval.
        retrieved_results = await self._retrieval_pipeline._retrieve_by_id(query_id, top_k)
        chunk_ids: list[int | str] = []
        retrieved_scores: list[float] = []
        paragraphs: list[str] = []
        cot_sentences: list[str] = []
        steps_completed = 0
        self._append_ordered_paragraphs(retrieved_results, chunk_ids, retrieved_scores, paragraphs)

        for step in range(self._max_multi_steps):
            steps_completed = step + 1
            reasoning_prompt = self._multi_reasoning_prompt_template.format(
                query=query_text,
                paragraphs=self._format_numbered_paragraphs(paragraphs),
                cot_history=self._format_cot_history(cot_sentences),
            )
            reasoning_response = await self._llm.ainvoke(reasoning_prompt)
            tracker.record(reasoning_response)
            reasoning_text = self._extract_text(reasoning_response)
            cot_sentence = self._extract_first_sentence(reasoning_text)
            cot_sentences.append(cot_sentence)

            if self._answer_signal.lower() in reasoning_text.lower():
                break

            cot_results = await self._retrieval_pipeline.retrieve(cot_sentence, top_k)
            self._append_ordered_paragraphs(cot_results, chunk_ids, retrieved_scores, paragraphs)

        answer_prompt = self._multi_retrieval_answer_prompt_template.format(
            query=query_text,
            context="\n\n".join(paragraphs),
            cot_history=self._format_cot_history(cot_sentences),
        )
        answer_response = await self._llm.ainvoke(answer_prompt)
        tracker.record(answer_response)
        answer_text = self._extract_text(answer_response)

        return GenerationResult(
            text=answer_text,
            token_usage=tracker.total,
            metadata={
                "complexity_tier": complexity_tier,
                "route": route,
                "retrieved_chunk_ids": chunk_ids,
                "retrieved_scores": retrieved_scores,
                "cot_sentences": cot_sentences,
                "steps_completed": steps_completed,
            },
        )
