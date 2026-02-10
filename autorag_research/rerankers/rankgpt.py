"""RankGPT reranker implementation using LLMs for listwise ranking."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field

from autorag_research.rerankers.api_base import APIReranker
from autorag_research.rerankers.base import RerankResult

if TYPE_CHECKING:
    pass

RANKGPT_PROMPT_TEMPLATE = """You are RankGPT, an intelligent assistant that can rank passages based on their relevance to the query.

The following are {num_passages} passages, each indicated by a numerical identifier [].

{passages}

Query: {query}

Rank the {num_passages} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [] > [], e.g., [1] > [2] > [3]. Only respond with the ranking results, do not say any word or explain."""


class RankGPTReranker(APIReranker):
    """Reranker using LLM-based listwise ranking (RankGPT method).

    This reranker uses an LLM to rank documents by comparing them all at once
    (listwise ranking), which is more effective than pointwise scoring.

    Requires a LangChain-compatible LLM (e.g., ChatOpenAI).

    Reference: "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents"
    https://arxiv.org/abs/2304.09542
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = Field(default="gpt-4o-mini", description="Model name for display purposes.")
    llm: Any = Field(default=None, description="LangChain LLM instance for ranking.")
    max_passages_per_call: int = Field(
        default=20,
        description="Maximum number of passages to rank in a single LLM call. "
        "If more passages are provided, sliding window is used.",
    )
    window_size: int = Field(default=10, description="Sliding window size for large document sets.")

    def model_post_init(self, __context) -> None:
        """Validate LLM after model creation."""
        if self.llm is None:
            msg = "LLM instance is required. Pass a LangChain-compatible LLM."
            raise ValueError(msg)

    def _format_passages(self, documents: list[str]) -> str:
        """Format passages with numerical identifiers."""
        return "\n\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(documents))

    def _parse_ranking(self, response: str, num_docs: int) -> list[int]:
        """Parse LLM response to extract ranking order.

        Args:
            response: LLM response text containing ranking.
            num_docs: Number of documents that were ranked.

        Returns:
            List of 0-indexed positions in ranked order.
        """
        # Extract numbers from the response
        pattern = r"\[(\d+)\]"
        matches = re.findall(pattern, response)

        # Convert to 0-indexed and filter valid indices
        ranked_indices = []
        seen = set()
        for match in matches:
            idx = int(match) - 1  # Convert to 0-indexed
            if 0 <= idx < num_docs and idx not in seen:
                ranked_indices.append(idx)
                seen.add(idx)

        # Add any missing indices at the end
        for i in range(num_docs):
            if i not in seen:
                ranked_indices.append(i)

        return ranked_indices

    def _rank_single_window(self, query: str, documents: list[str]) -> list[int]:
        """Rank documents in a single window using LLM."""
        prompt = RANKGPT_PROMPT_TEMPLATE.format(
            num_passages=len(documents),
            passages=self._format_passages(documents),
            query=query,
        )

        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        return self._parse_ranking(response_text, len(documents))

    async def _arank_single_window(self, query: str, documents: list[str]) -> list[int]:
        """Rank documents in a single window using LLM asynchronously."""
        prompt = RANKGPT_PROMPT_TEMPLATE.format(
            num_passages=len(documents),
            passages=self._format_passages(documents),
            query=query,
        )

        response = await self.llm.ainvoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        return self._parse_ranking(response_text, len(documents))

    def _sliding_window_rank(self, query: str, documents: list[str], top_k: int) -> list[int]:
        """Use sliding window for large document sets.

        This implements the sliding window approach from the RankGPT paper
        where we iteratively rank windows of documents and bubble up the best ones.
        """
        if len(documents) <= self.max_passages_per_call:
            return self._rank_single_window(query, documents)

        # Start with initial ranking (indices)
        current_order = list(range(len(documents)))

        # Slide from end to beginning
        step = self.window_size
        for start in range(len(documents) - self.max_passages_per_call, -1, -step):
            end = min(start + self.max_passages_per_call, len(documents))
            window_indices = current_order[start:end]
            window_docs = [documents[i] for i in window_indices]

            # Rank this window
            window_ranking = self._rank_single_window(query, window_docs)

            # Map back to original indices
            reordered = [window_indices[i] for i in window_ranking]
            current_order = current_order[:start] + reordered + current_order[end:]

        return current_order[:top_k]

    async def _asliding_window_rank(self, query: str, documents: list[str], top_k: int) -> list[int]:
        """Use sliding window for large document sets asynchronously."""
        if len(documents) <= self.max_passages_per_call:
            return await self._arank_single_window(query, documents)

        current_order = list(range(len(documents)))
        step = self.window_size

        for start in range(len(documents) - self.max_passages_per_call, -1, -step):
            end = min(start + self.max_passages_per_call, len(documents))
            window_indices = current_order[start:end]
            window_docs = [documents[i] for i in window_indices]

            window_ranking = await self._arank_single_window(query, window_docs)
            reordered = [window_indices[i] for i in window_ranking]
            current_order = current_order[:start] + reordered + current_order[end:]

        return current_order[:top_k]

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents using RankGPT method.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. If None, returns all documents.

        Returns:
            List of RerankResult objects sorted by relevance (descending).
        """
        if not documents:
            return []

        top_k = top_k or len(documents)
        top_k = min(top_k, len(documents))

        ranked_indices = self._sliding_window_rank(query, documents, top_k)

        # Create results with descending scores based on rank position
        results = []
        for rank, idx in enumerate(ranked_indices[:top_k]):
            score = 1.0 - (rank / len(documents))  # Higher rank = higher score
            results.append(RerankResult(index=idx, text=documents[idx], score=score))

        return results

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents asynchronously using RankGPT method.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. If None, returns all documents.

        Returns:
            List of RerankResult objects sorted by relevance (descending).
        """
        if not documents:
            return []

        top_k = top_k or len(documents)
        top_k = min(top_k, len(documents))

        ranked_indices = await self._asliding_window_rank(query, documents, top_k)

        results = []
        for rank, idx in enumerate(ranked_indices[:top_k]):
            score = 1.0 - (rank / len(documents))
            results.append(RerankResult(index=idx, text=documents[idx], score=score))

        return results
