"""DynamicRAG inference-time LLM reranker.

DynamicRAG inference uses a listwise LLM reranker agent: given the query and a
numbered candidate set, it generates the ordered subset of document identifiers
needed for answering. The generated list directly determines the dynamic k, and
`None` is a valid zero-document decision.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import Field

from autorag_research.rerankers.base import BaseReranker, RerankResult

logger = logging.getLogger("AutoRAG-Research")

DEFAULT_DYNAMIC_RAG_RERANK_PROMPT = """You are a DynamicRAG reranker.

Given a query and numbered documents [1]..[N], output ONLY the identifiers of documents needed to answer the query,
comma-separated, in relevance order. Stop when the selected documents collectively provide enough information. Output
None if no documents are required.

Query:
{query}

Documents:
{documents}

Selected document identifiers:"""


class DynamicRAGReranker(BaseReranker):
    """Prompted LLM listwise reranker for DynamicRAG dynamic-k selection."""

    model_name: str = Field(default="dynamic-rag", description="DynamicRAG reranker name.")
    llm: Any | None = Field(default=None, description="LangChain language model used as the reranker agent.")
    prompt_template: str = Field(
        default=DEFAULT_DYNAMIC_RAG_RERANK_PROMPT,
        description="Prompt template with {query} and {documents} placeholders for listwise DynamicRAG reranking.",
    )
    max_doc_chars: int | None = Field(default=2000, description="Optional per-document truncation for prompt safety.")

    def _require_llm(self) -> Any:
        """Return the configured LLM or raise the rerank-time configuration error."""
        if self.llm is None:
            msg = "DynamicRAGReranker requires llm before rerank"
            raise ValueError(msg)
        return self.llm

    def _format_documents(self, documents: list[str]) -> str:
        """Format numbered documents for the listwise prompt."""
        formatted_documents: list[str] = []
        for index, document in enumerate(documents, start=1):
            text = str(document)
            if self.max_doc_chars is not None and self.max_doc_chars >= 0:
                text = text[: self.max_doc_chars]
            formatted_documents.append(f"[{index}] {text}")
        return "\n\n".join(formatted_documents)

    def _format_prompt(self, query: str, documents: list[str]) -> str:
        """Build the DynamicRAG reranker prompt."""
        return self.prompt_template.format(query=query, documents=self._format_documents(documents))

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text from a LangChain response."""
        return str(response.content) if hasattr(response, "content") else str(response)

    @staticmethod
    def _parse_selected_indices(response_text: str, document_count: int) -> tuple[list[int], bool]:
        """Parse ordered unique zero-based indices and whether a None decision was stated.

        A response whose leading token is ``None`` (ignoring punctuation/brackets) is an explicit
        zero-document decision even when illustrative digits follow, e.g. ``None of the documents 1, 2``.
        """
        none_prefixed = re.match(r"^[\s\[\(\"'`*-]*none\b", response_text, flags=re.IGNORECASE) is not None
        none_stated = none_prefixed or re.search(r"\bnone\b", response_text, flags=re.IGNORECASE) is not None
        selected_indices: list[int] = []
        seen: set[int] = set()
        if not none_prefixed:
            for match in re.finditer(r"\d+", response_text):
                document_id = int(match.group())
                index = document_id - 1
                if 0 <= index < document_count and index not in seen:
                    selected_indices.append(index)
                    seen.add(index)
        return selected_indices, none_stated

    @staticmethod
    def _results_from_indices(indices: list[int], documents: list[str]) -> list[RerankResult]:
        """Create synthetic relevance scores from generated output rank."""
        selected_count = len(indices)
        return [
            RerankResult(index=index, text=documents[index], score=float(selected_count - position))
            for position, index in enumerate(indices)
        ]

    def _parse_results(self, response_text: str, documents: list[str], top_k: int | None) -> list[RerankResult] | None:
        """Parse LLM output into rerank results; return None when the output is unparseable."""
        selected_indices, none_stated = self._parse_selected_indices(response_text, len(documents))
        if none_stated and not selected_indices:
            return []
        if not selected_indices:
            return None
        if top_k is not None:
            selected_indices = selected_indices[:top_k]
        return self._results_from_indices(selected_indices, documents)

    @staticmethod
    def _raise_unparseable(response_text: str) -> ValueError:
        """Build the rerank failure error for output that is neither IDs nor None."""
        snippet = response_text.strip().replace("\n", " ")[:200]
        msg = f"DynamicRAG reranker output is not an ordered document-ID list or None: {snippet!r}"
        return ValueError(msg)

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents by prompting the DynamicRAG LLM reranker agent.

        Unparseable output gets one bounded retry; a second unparseable response raises,
        because the paper's reranker contract is an ordered ID subset or an explicit None.
        """
        if not documents:
            return []
        llm = self._require_llm()
        prompt = self._format_prompt(query, documents)
        response_text = self._extract_text(llm.invoke(prompt))
        results = self._parse_results(response_text, documents, top_k)
        if results is not None:
            return results
        logger.warning("DynamicRAG reranker output was unparseable; retrying once")
        response_text = self._extract_text(llm.invoke(prompt))
        results = self._parse_results(response_text, documents, top_k)
        if results is not None:
            return results
        raise self._raise_unparseable(response_text)

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Asynchronously rerank documents by prompting the DynamicRAG LLM reranker agent.

        Unparseable output gets one bounded retry; a second unparseable response raises,
        because the paper's reranker contract is an ordered ID subset or an explicit None.
        """
        if not documents:
            return []
        llm = self._require_llm()
        prompt = self._format_prompt(query, documents)
        response_text = self._extract_text(await llm.ainvoke(prompt))
        results = self._parse_results(response_text, documents, top_k)
        if results is not None:
            return results
        logger.warning("DynamicRAG reranker output was unparseable; retrying once")
        response_text = self._extract_text(await llm.ainvoke(prompt))
        results = self._parse_results(response_text, documents, top_k)
        if results is not None:
            return results
        raise self._raise_unparseable(response_text)


__all__ = ["DEFAULT_DYNAMIC_RAG_RERANK_PROMPT", "DynamicRAGReranker"]
