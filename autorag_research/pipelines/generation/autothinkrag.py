"""AutoThinkRAG generation pipeline for complexity-aware multimodal RAG."""

from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseGenerationPipelineConfig
from autorag_research.orm.service.generation_pipeline import GenerationResult
from autorag_research.pipelines.generation.base import BaseGenerationPipeline
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.util import TokenUsageTracker, image_chunk_to_pil_images, pil_image_to_data_uri

DEFAULT_COMPLEXITY_PROMPT_TEMPLATE = """Analyze the complexity of the following query and classify it as exactly one of: simple, moderate, complex.

- simple: Factual lookup, single-hop reasoning, direct answer from context
- moderate: Requires synthesis across multiple pieces of information or visual interpretation
- complex: Multi-hop reasoning, requires combining visual and textual evidence, or needs step-by-step deduction

Query: {query}

Classification (respond with ONLY one word - simple, moderate, or complex):"""

DEFAULT_SIMPLE_PROMPT_TEMPLATE = """Answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""

DEFAULT_COMPLEX_PROMPT_TEMPLATE = """You are solving a complex question that requires multi-step reasoning. Use the provided context, visual interpretation, and reasoning chain to synthesize a final answer.

Context:
{context}

{visual_context}

Reasoning so far:
{reasoning_chain}

Question: {query}

Final Answer:"""

DEFAULT_VISUAL_INTERPRETATION_PROMPT_TEMPLATE = """Analyze the provided document images and extract all query-relevant information as structured text.

Question: {query}

Describe the visual content relevant to answering this question:"""


@dataclass(kw_only=True)
class AutoThinkRAGPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for the AutoThinkRAG generation pipeline."""

    complexity_prompt_template: str = field(default=DEFAULT_COMPLEXITY_PROMPT_TEMPLATE)
    simple_prompt_template: str = field(default=DEFAULT_SIMPLE_PROMPT_TEMPLATE)
    complex_prompt_template: str = field(default=DEFAULT_COMPLEX_PROMPT_TEMPLATE)
    visual_interpretation_prompt_template: str = field(default=DEFAULT_VISUAL_INTERPRETATION_PROMPT_TEMPLATE)
    vlm: str | BaseChatModel | None = None
    complexity_tiers: list[str] = field(default_factory=lambda: ["simple", "moderate", "complex"])
    max_reasoning_steps: int = 3
    temperature: float = 0.0
    max_tokens: int | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        """Auto-convert VLM string configs into instantiated chat models."""
        if name == "vlm" and isinstance(value, str):
            from autorag_research.injection import load_llm

            value = load_llm(value)
        super().__setattr__(name, value)

    def get_pipeline_class(self) -> type["AutoThinkRAGPipeline"]:
        """Return the AutoThinkRAGPipeline class."""
        return AutoThinkRAGPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for AutoThinkRAGPipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)

        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "vlm": self.vlm,
            "complexity_prompt_template": self.complexity_prompt_template,
            "simple_prompt_template": self.simple_prompt_template,
            "complex_prompt_template": self.complex_prompt_template,
            "visual_interpretation_prompt_template": self.visual_interpretation_prompt_template,
            "complexity_tiers": self.complexity_tiers,
            "max_reasoning_steps": self.max_reasoning_steps,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


class AutoThinkRAGPipeline(BaseGenerationPipeline):
    """Complexity-aware generation pipeline with optional visual interpretation."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        complexity_prompt_template: str = DEFAULT_COMPLEXITY_PROMPT_TEMPLATE,
        simple_prompt_template: str = DEFAULT_SIMPLE_PROMPT_TEMPLATE,
        complex_prompt_template: str = DEFAULT_COMPLEX_PROMPT_TEMPLATE,
        visual_interpretation_prompt_template: str = DEFAULT_VISUAL_INTERPRETATION_PROMPT_TEMPLATE,
        vlm: str | BaseChatModel | None = None,
        complexity_tiers: list[str] | None = None,
        max_reasoning_steps: int = 3,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        schema: Any | None = None,
    ):
        """Initialize the AutoThinkRAG pipeline."""
        self._complexity_prompt_template = complexity_prompt_template
        self._simple_prompt_template = simple_prompt_template
        self._complex_prompt_template = complex_prompt_template
        self._visual_interpretation_prompt_template = visual_interpretation_prompt_template
        self._vlm = vlm
        self._complexity_tiers = complexity_tiers or ["simple", "moderate", "complex"]
        self._max_reasoning_steps = max_reasoning_steps
        self._temperature = temperature
        self._max_tokens = max_tokens

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return AutoThinkRAG pipeline configuration."""
        vlm_name: str | None = None
        if self._vlm is not None:
            model_name = getattr(self._vlm, "model_name", None)
            vlm_name = model_name if isinstance(model_name, str) else type(self._vlm).__name__

        return {
            "type": "autothinkrag",
            "complexity_prompt_template": self._complexity_prompt_template,
            "simple_prompt_template": self._simple_prompt_template,
            "complex_prompt_template": self._complex_prompt_template,
            "visual_interpretation_prompt_template": self._visual_interpretation_prompt_template,
            "vlm": vlm_name,
            "complexity_tiers": self._complexity_tiers,
            "max_reasoning_steps": self._max_reasoning_steps,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "retrieval_pipeline_id": self._retrieval_pipeline.pipeline_id,
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text content from an LLM response."""
        return response.content if hasattr(response, "content") else str(response)

    def _build_generation_kwargs(self) -> dict[str, Any]:
        """Build generation kwargs supported by LangChain ainvoke."""
        kwargs: dict[str, Any] = {}
        if self._temperature != 0.0:
            kwargs["temperature"] = self._temperature
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        return kwargs

    async def _ainvoke_model(self, model: BaseLanguageModel, prompt: Any) -> Any:
        """Invoke an LLM and gracefully fall back if it rejects generation kwargs."""
        kwargs = self._build_generation_kwargs()
        if not kwargs:
            return await model.ainvoke(prompt)

        try:
            return await model.ainvoke(prompt, **kwargs)
        except TypeError:
            return await model.ainvoke(prompt)

    def _parse_complexity_tier(self, response_text: str) -> Literal["simple", "moderate", "complex"]:
        """Parse the routed complexity tier, defaulting to moderate."""
        normalized = response_text.strip().lower()
        for tier in self._complexity_tiers:
            if normalized == tier.lower():
                return tier.lower()  # type: ignore[return-value]
        for tier in self._complexity_tiers:
            if tier.lower() in normalized:
                return tier.lower()  # type: ignore[return-value]
        return "moderate"

    def _format_context(self, chunk_ids: list[int | str]) -> str:
        """Fetch and join chunk contents for prompt context."""
        if not chunk_ids:
            return ""
        chunk_contents = self._service.get_chunk_contents(chunk_ids)
        return "\n\n".join(content for content in chunk_contents if content)

    async def _generate_visual_interpretation(
        self,
        query: str,
        image_chunk_ids: list[int | str],
        tracker: TokenUsageTracker,
    ) -> str | None:
        """Run VLM-based visual interpretation if images and a VLM are available."""
        if self._vlm is None or not image_chunk_ids:
            return None

        image_chunks = self._service.get_image_chunk_contents(image_chunk_ids)
        images = image_chunk_to_pil_images(image_chunks)
        if not images:
            return None

        content: list[str | dict[str, Any]] = [
            {
                "type": "text",
                "text": self._visual_interpretation_prompt_template.format(query=query),
            }
        ]
        for image in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": pil_image_to_data_uri(image)},
            })

        response = await self._ainvoke_model(self._vlm, [HumanMessage(content=content)])
        tracker.record(response)
        return self._extract_text(response)

    @staticmethod
    def _merge_retrieval_metadata(
        existing_ids: list[int | str],
        existing_scores: list[float],
        results: list[dict[str, Any]],
    ) -> tuple[list[int | str], list[float]]:
        """Merge retrieval results while preserving first-seen order."""
        score_by_id = dict(zip(existing_ids, existing_scores, strict=False))
        ordered_ids = list(existing_ids)

        for result in results:
            doc_id = result.get("doc_id")
            score = result.get("score", 0.0)
            if doc_id is None:
                continue
            if doc_id not in score_by_id:
                ordered_ids.append(doc_id)
                score_by_id[doc_id] = score

        return ordered_ids, [score_by_id[doc_id] for doc_id in ordered_ids]

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate an answer with adaptive routing based on query complexity."""
        tracker = TokenUsageTracker()
        query = self._service.get_query_text(query_id)

        complexity_prompt = self._complexity_prompt_template.format(query=query)
        complexity_response = await self._ainvoke_model(self._llm, complexity_prompt)
        tracker.record(complexity_response)
        complexity_tier = self._parse_complexity_tier(self._extract_text(complexity_response))

        retrieved = await self._retrieval_pipeline._retrieve_by_id(query_id, top_k)
        chunk_ids, retrieved_scores = self._merge_retrieval_metadata([], [], retrieved)
        context = self._format_context(chunk_ids)
        visual_interpretation = await self._generate_visual_interpretation(query, chunk_ids, tracker)

        metadata: dict[str, Any] = {
            "complexity_tier": complexity_tier,
            "retrieved_chunk_ids": chunk_ids,
            "retrieved_scores": retrieved_scores,
        }
        if visual_interpretation is not None:
            metadata["visual_interpretation"] = visual_interpretation

        if complexity_tier == "complex":
            reasoning_steps: list[str] = []
            accumulated_context = context

            for _step in range(self._max_reasoning_steps):
                reasoning_prompt = self._complex_prompt_template.format(
                    context=accumulated_context,
                    visual_context=(
                        f"Visual Interpretation:\n{visual_interpretation}" if visual_interpretation else ""
                    ),
                    reasoning_chain="\n".join(reasoning_steps) if reasoning_steps else "(No prior reasoning yet)",
                    query=query,
                )
                reasoning_prompt = (
                    f"{reasoning_prompt}\n\nProvide the next concise reasoning step only, not the final answer."
                )
                reasoning_response = await self._ainvoke_model(self._llm, reasoning_prompt)
                tracker.record(reasoning_response)
                reasoning_text = self._extract_text(reasoning_response).strip()
                reasoning_steps.append(reasoning_text)

                follow_up_results = await self._retrieval_pipeline.retrieve(reasoning_text, top_k)
                chunk_ids, retrieved_scores = self._merge_retrieval_metadata(
                    chunk_ids,
                    retrieved_scores,
                    follow_up_results,
                )
                accumulated_context = self._format_context(chunk_ids)

            final_prompt = self._complex_prompt_template.format(
                context=accumulated_context,
                visual_context=f"Visual Interpretation:\n{visual_interpretation}" if visual_interpretation else "",
                reasoning_chain="\n".join(reasoning_steps),
                query=query,
            )
            final_response = await self._ainvoke_model(self._llm, final_prompt)
            tracker.record(final_response)
            metadata["reasoning_steps"] = reasoning_steps
            metadata["retrieved_chunk_ids"] = chunk_ids
            metadata["retrieved_scores"] = retrieved_scores

            return GenerationResult(
                text=self._extract_text(final_response),
                token_usage=tracker.total,
                metadata=metadata,
            )

        answer_context = context
        if visual_interpretation:
            visual_block = f"Visual Interpretation:\n{visual_interpretation}"
            answer_context = f"{context}\n\n{visual_block}" if context else visual_block

        answer_prompt = self._simple_prompt_template.format(context=answer_context, query=query)
        answer_response = await self._ainvoke_model(self._llm, answer_prompt)
        tracker.record(answer_response)

        return GenerationResult(
            text=self._extract_text(answer_response),
            token_usage=tracker.total,
            metadata=metadata,
        )
