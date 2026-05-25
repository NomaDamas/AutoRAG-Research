"""DynamicRAG generation pipeline.

This pipeline implements the inference/evaluation slice of DynamicRAG: retrieve a
candidate pool, dynamically rerank and truncate it, and generate from only the
selected evidence. Training the original DynamicRAG policy with generator
feedback is outside this repo-native baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseGenerationPipelineConfig
from autorag_research.orm.service.generation_pipeline import GenerationResult
from autorag_research.pipelines.generation.base import BaseGenerationPipeline
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.rerankers.base import BaseReranker, RerankResult
from autorag_research.rerankers.dynamic_rag import DynamicRAGReranker
from autorag_research.util import TokenUsageTracker

DEFAULT_DYNAMIC_RAG_PROMPT = """Answer the question using only the dynamically selected evidence.

Question:
{query}

Evidence:
{context}

Answer:"""


@dataclass(kw_only=True)
class DynamicRAGPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for DynamicRAGPipeline."""

    reranker: str | BaseReranker = field(default_factory=DynamicRAGReranker)
    prompt_template: str = field(default=DEFAULT_DYNAMIC_RAG_PROMPT)
    candidate_top_k: int = 20

    def __setattr__(self, name: str, value: Any) -> None:
        """Auto-convert string reranker config names to reranker instances."""
        if name == "reranker" and isinstance(value, str):
            from autorag_research.injection import load_reranker

            value = load_reranker(value)
        super().__setattr__(name, value)

    def get_pipeline_class(self) -> type[DynamicRAGPipeline]:
        """Return the DynamicRAGPipeline class."""
        return DynamicRAGPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for DynamicRAGPipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)
        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "reranker": self.reranker,
            "prompt_template": self.prompt_template,
            "candidate_top_k": self.candidate_top_k,
        }


class DynamicRAGPipeline(BaseGenerationPipeline):
    """Generation pipeline that applies DynamicRAG reranking before answering."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        reranker: BaseReranker | None = None,
        prompt_template: str = DEFAULT_DYNAMIC_RAG_PROMPT,
        candidate_top_k: int = 20,
        schema: Any | None = None,
    ):
        """Initialize DynamicRAGPipeline."""
        if candidate_top_k < 1:
            msg = "candidate_top_k must be >= 1"
            raise ValueError(msg)
        if "{query}" not in prompt_template or "{context}" not in prompt_template:
            msg = "prompt_template must contain {query} and {context} placeholders"
            raise ValueError(msg)

        self.reranker = reranker or DynamicRAGReranker()
        self.prompt_template = prompt_template
        self.candidate_top_k = candidate_top_k

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return DynamicRAG pipeline configuration."""
        model_name = getattr(self._llm, "model_name", None)
        if model_name is None or not isinstance(model_name, str):
            model_name = type(self._llm).__name__
        return {
            "type": "dynamic_rag",
            "prompt_template": self.prompt_template,
            "candidate_top_k": self.candidate_top_k,
            "reranker_model": getattr(self.reranker, "model_name", type(self.reranker).__name__),
            "retrieval_pipeline_id": self._retrieval_pipeline.pipeline_id,
            "llm_model": model_name,
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text from a LangChain response."""
        return response.content if hasattr(response, "content") else str(response)

    def _contents_from_results(self, results: list[dict[str, Any]]) -> tuple[list[int | str], list[str]]:
        """Extract IDs and contents from retrieval results, backfilling missing text."""
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

    @staticmethod
    def _map_reranked_results(
        candidate_doc_ids: list[int | str], rerank_results: list[RerankResult]
    ) -> tuple[list[int | str], list[str], list[float]]:
        """Map reranker result indices back to candidate metadata."""
        selected_doc_ids: list[int | str] = []
        selected_contents: list[str] = []
        selected_scores: list[float] = []
        for rerank_result in rerank_results:
            if rerank_result.index < 0 or rerank_result.index >= len(candidate_doc_ids):
                msg = f"Reranker returned out-of-range candidate index: {rerank_result.index}"
                raise ValueError(msg)
            selected_doc_ids.append(candidate_doc_ids[rerank_result.index])
            selected_contents.append(rerank_result.text)
            selected_scores.append(float(rerank_result.score))
        return selected_doc_ids, selected_contents, selected_scores

    def _build_prompt(self, query: str, selected_contents: list[str]) -> str:
        """Build the final DynamicRAG prompt."""
        context = "\n\n".join(f"[{index + 1}] {content}" for index, content in enumerate(selected_contents))
        return self.prompt_template.format(query=query, context=context or "(no evidence selected)")

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate with dynamic reranking and truncation."""
        query_text = self._service.get_query_text(query_id)
        fetch_k = max(top_k, self.candidate_top_k)
        candidates = await self._retrieval_pipeline._retrieve_by_id(query_id, fetch_k)
        candidate_doc_ids, candidate_contents = self._contents_from_results(candidates)
        rerank_results = await self.reranker.arerank(query_text, candidate_contents, top_k=top_k)
        selected_doc_ids, selected_contents, selected_scores = self._map_reranked_results(
            candidate_doc_ids,
            rerank_results,
        )

        prompt = self._build_prompt(query_text, selected_contents)
        response = await self._llm.ainvoke(prompt)
        tracker = TokenUsageTracker()
        token_usage = tracker.record(response)
        return GenerationResult(
            text=self._extract_text(response),
            token_usage=token_usage,
            metadata={
                "candidate_chunk_ids": candidate_doc_ids,
                "selected_chunk_ids": selected_doc_ids,
                "selected_scores": selected_scores,
                "effective_top_k": len(selected_doc_ids),
            },
        )


__all__ = [
    "DEFAULT_DYNAMIC_RAG_PROMPT",
    "DynamicRAGPipeline",
    "DynamicRAGPipelineConfig",
]
