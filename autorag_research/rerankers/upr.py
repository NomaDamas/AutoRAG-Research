"""UPR (Unsupervised Passage Reranker) implementation using LLMs."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field

from autorag_research.rerankers.api_base import APIReranker
from autorag_research.rerankers.base import RerankResult

if TYPE_CHECKING:
    pass

UPR_PROMPT_TEMPLATE = """Passage: {passage}

Please write a question based on this passage."""


class UPRReranker(APIReranker):
    """Reranker using UPR (Unsupervised Passage Reranker) method.

    UPR generates a question from each passage using an LLM, then computes
    similarity between the generated questions and the original query.
    Passages whose generated questions are more similar to the original
    query are ranked higher.

    This approach doesn't require labeled training data and can work with
    any LLM that can generate questions from passages.

    Reference: "Improving Passage Retrieval with Zero-Shot Question Generation"
    https://arxiv.org/abs/2204.07496
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = Field(default="gpt-4o-mini", description="Model name for display purposes.")
    llm: Any = Field(default=None, description="LangChain LLM instance for question generation.")
    use_logprobs: bool = Field(
        default=False,
        description="If True, uses log probabilities for scoring. Requires LLM that supports logprobs (e.g., OpenAI).",
    )

    def model_post_init(self, __context) -> None:
        """Validate LLM after model creation."""
        if self.llm is None:
            msg = "LLM instance is required. Pass a LangChain-compatible LLM."
            raise ValueError(msg)

    def _generate_question(self, passage: str) -> str:
        """Generate a question from a passage using LLM."""
        prompt = UPR_PROMPT_TEMPLATE.format(passage=passage)
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    async def _agenerate_question(self, passage: str) -> str:
        """Generate a question from a passage using LLM asynchronously."""
        prompt = UPR_PROMPT_TEMPLATE.format(passage=passage)
        response = await self.llm.ainvoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def _compute_similarity(self, query: str, generated_question: str) -> float:
        """Compute similarity between query and generated question.

        Uses simple word overlap ratio as similarity metric.
        For production use, consider using embedding-based similarity.
        """
        query_words = set(query.lower().split())
        question_words = set(generated_question.lower().split())

        if not query_words or not question_words:
            return 0.0

        intersection = query_words & question_words
        union = query_words | question_words
        return len(intersection) / len(union)

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents using UPR method.

        For each document:
        1. Generate a question that the passage could answer
        2. Compute similarity between generated question and original query
        3. Rank by similarity score

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. If None, returns all documents.

        Returns:
            List of RerankResult objects sorted by relevance score (descending).
        """
        if not documents:
            return []

        top_k = top_k or len(documents)
        top_k = min(top_k, len(documents))

        # Generate questions and compute scores
        scores: list[tuple[int, float, str]] = []
        for i, doc in enumerate(documents):
            generated_question = self._generate_question(doc)
            score = self._compute_similarity(query, generated_question)
            scores.append((i, score, doc))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        return [RerankResult(index=idx, text=text, score=score) for idx, score, text in scores[:top_k]]

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents asynchronously using UPR method.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. If None, returns all documents.

        Returns:
            List of RerankResult objects sorted by relevance score (descending).
        """
        if not documents:
            return []

        top_k = top_k or len(documents)
        top_k = min(top_k, len(documents))

        # Generate questions for all documents
        questions = await asyncio.gather(*[self._agenerate_question(doc) for doc in documents])

        # Compute scores
        scores: list[tuple[int, float, str]] = []
        for i, (doc, generated_question) in enumerate(zip(documents, questions, strict=True)):
            score = self._compute_similarity(query, generated_question)
            scores.append((i, score, doc))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return [RerankResult(index=idx, text=text, score=score) for idx, score, text in scores[:top_k]]
