"""Query Rewrite retrieval pipeline for AutoRAG-Research.

This pipeline implements an inference-time Rewrite-Retrieve flow inspired by
"Query Rewriting in Retrieval-Augmented Large Language Models"
(Ma et al., EMNLP 2023).

Scope note:
- This implementation covers the paper's practical query-rewrite-then-retrieve
  inference pattern.
- The paper's trainable/RL rewriter is intentionally out of scope for this MVP.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.injection import health_check_llm
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline

logger = logging.getLogger("AutoRAG-Research")

DEFAULT_QUERY_REWRITE_PROMPT_TEMPLATE = """Rewrite the following question into a concise search query.
Keep the original intent, add missing retrieval hints only when helpful, and return only the rewritten query.

Question: {query}
Rewritten query:"""


@dataclass(kw_only=True)
class QueryRewritePipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for Query Rewrite retrieval pipeline."""

    llm: str | BaseLanguageModel
    retrieval_pipeline_name: str
    prompt_template: str = field(default=DEFAULT_QUERY_REWRITE_PROMPT_TEMPLATE)
    _retrieval_pipeline: BaseRetrievalPipeline | None = field(default=None, repr=False)

    def __setattr__(self, name: str, value: Any) -> None:
        """Auto-convert string LLM config names to model instances."""
        if name == "llm" and isinstance(value, str):
            from autorag_research.injection import load_llm

            value = load_llm(value)
            health_check_llm(value)
        super().__setattr__(name, value)

    def inject_retrieval_pipeline(self, pipeline: BaseRetrievalPipeline) -> None:
        """Inject the wrapped retrieval pipeline instance."""
        self._retrieval_pipeline = pipeline

    def get_pipeline_class(self) -> type["QueryRewriteRetrievalPipeline"]:
        """Return the QueryRewriteRetrievalPipeline class."""
        return QueryRewriteRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for QueryRewriteRetrievalPipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)

        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "prompt_template": self.prompt_template,
        }


class QueryRewriteRetrievalPipeline(BaseRetrievalPipeline):
    """Rewrite the query with an LLM, then delegate retrieval to another pipeline."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        prompt_template: str = DEFAULT_QUERY_REWRITE_PROMPT_TEMPLATE,
        schema: Any | None = None,
    ):
        """Initialize query rewrite retrieval pipeline."""
        if "{query}" not in prompt_template:
            msg = "prompt_template must contain '{query}' placeholder"
            raise ValueError(msg)

        self.llm = llm
        self._retrieval_pipeline = retrieval_pipeline
        self.prompt_template = prompt_template

        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return query rewrite pipeline configuration for storage."""
        model_name = getattr(self.llm, "model_name", None)
        if model_name is None or not isinstance(model_name, str):
            model_name = type(self.llm).__name__

        return {
            "type": "query_rewrite",
            "prompt_template": self.prompt_template,
            "retrieval_pipeline_id": getattr(self._retrieval_pipeline, "pipeline_id", None),
            "wrapped_pipeline_type": type(self._retrieval_pipeline).__name__,
            "llm_model": model_name,
        }

    @staticmethod
    def _extract_response_content(response: Any) -> str:
        """Extract text content from an LLM response."""
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)

    async def _rewrite_query(self, query_text: str) -> str:
        """Rewrite the query text with the configured LLM prompt."""
        prompt = self.prompt_template.format(query=query_text)
        response = await self.llm.ainvoke(prompt)
        rewritten_query = self._extract_response_content(response).strip()

        if rewritten_query:
            return rewritten_query

        logger.warning("Query rewrite returned empty text; falling back to original query")
        return query_text

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        """Fetch query text by ID, rewrite it, and retrieve with the wrapped pipeline."""
        query_texts = self._service.fetch_query_texts([query_id])
        if not query_texts:
            return []

        return await self._retrieve_by_text(query_texts[0], top_k)

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """Rewrite raw query text, then retrieve with the wrapped pipeline."""
        rewritten_query = await self._rewrite_query(query_text)
        return await self._retrieval_pipeline.retrieve(rewritten_query, top_k)


__all__ = [
    "DEFAULT_QUERY_REWRITE_PROMPT_TEMPLATE",
    "QueryRewritePipelineConfig",
    "QueryRewriteRetrievalPipeline",
]
