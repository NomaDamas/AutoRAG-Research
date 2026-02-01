"""ET2RAG (Efficient Test-Time RAG) Pipeline.

This module implements the ET2RAG algorithm from the paper:
"ET2-RAG: Efficient and Effective Test-Time Retrieval-Augmented Generation"
(arXiv:2511.01059)

The key insight of ET2RAG is to use majority voting on context subsets, not on generated
responses. The algorithm:
1. Creates multiple context subsets from retrieved documents
2. Generates PARTIAL responses for each subset (different prompts!)
3. Uses semantic similarity voting to select the BEST SUBSET
4. Generates a FULL response with the selected subset

This differs from naive ensemble approaches that generate N responses from the same prompt.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseGenerationPipelineConfig
from autorag_research.evaluation.metrics import calculate_cosine_similarity
from autorag_research.orm.service.generation_pipeline import GenerationResult
from autorag_research.orm.uow.generation_uow import GenerationUnitOfWork
from autorag_research.pipelines.generation.base import BaseGenerationPipeline
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline

logger = logging.getLogger("AutoRAG-Research")

DEFAULT_PROMPT_TEMPLATE = """Answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""


class OrganizationStrategy(Enum):
    """Strategies for organizing retrieved documents into subsets.

    Different strategies are optimal for different types of data:
    - QA: For factoid QA datasets (TriviaQA, PopQA) where top1 is usually relevant
    - RECIPE: For long-document datasets (Recipe1M) where each doc is self-contained
    - IMAGE: For image captioning (COCO) where multiple captions provide context
    """

    QA = "qa"
    RECIPE = "recipe"
    IMAGE = "image"


@dataclass(kw_only=True)
class ET2RAGPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for ET2RAG pipeline.

    Attributes:
        embedding_model: Embedding model for similarity computation (string config name or instance).
        organization_strategy: Strategy for creating context subsets (qa/recipe/image).
        num_subsets: Number of subsets to create (None = auto-determine based on strategy).
        partial_generation_max_tokens: Max tokens for partial responses during voting.
        full_generation_max_tokens: Max tokens for final generation (None = no limit).
        prompt_template: Custom prompt template with {context} and {query} placeholders.
    """

    embedding_model: str | Embeddings
    organization_strategy: str = "qa"
    num_subsets: int | None = None
    partial_generation_max_tokens: int = 100
    full_generation_max_tokens: int | None = None
    prompt_template: str | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        """Auto-convert string embedding_model to instance."""
        if name == "embedding_model" and isinstance(value, str):
            from autorag_research.injection import load_embedding_model

            value = load_embedding_model(value)
        super().__setattr__(name, value)

    def get_pipeline_class(self) -> type:
        """Return the ET2RAGPipeline class."""
        return ET2RAGPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for ET2RAGPipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)
        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "embedding_model": self.embedding_model,
            "organization_strategy": OrganizationStrategy(self.organization_strategy),
            "num_subsets": self.num_subsets,
            "partial_generation_max_tokens": self.partial_generation_max_tokens,
            "full_generation_max_tokens": self.full_generation_max_tokens,
            "prompt_template": self.prompt_template,
        }


class ET2RAGPipeline(BaseGenerationPipeline):
    """ET2RAG pipeline implementing subset-based majority voting.

    The algorithm creates context subsets from retrieved documents, generates
    partial responses for each subset, uses semantic similarity voting to select
    the best subset, then generates a full response with that subset.

    This approach is more robust than naive ensemble methods because:
    1. Each subset provides different context, not just sampling variation
    2. Voting identifies which context subset is most consistent
    3. Final generation uses full token budget with optimal context

    Example:
        ```python
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        from autorag_research.orm.connection import DBConnection
        from autorag_research.pipelines.generation.et2rag import ET2RAGPipeline
        from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline

        db = DBConnection.from_config()
        session_factory = db.get_session_factory()

        # Create retrieval pipeline
        retrieval_pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            name="bm25_baseline",
            index_path="/path/to/index",
        )

        # Create ET2RAG generation pipeline
        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="et2rag_v1",
            llm=ChatOpenAI(model="gpt-4"),
            retrieval_pipeline=retrieval_pipeline,
            embedding_model=OpenAIEmbeddings(),
            organization_strategy=OrganizationStrategy.QA,
            num_subsets=5,
        )

        # Run pipeline
        results = pipeline.run(top_k=10)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        embedding_model: Embeddings,
        organization_strategy: OrganizationStrategy = OrganizationStrategy.QA,
        num_subsets: int | None = None,
        partial_generation_max_tokens: int = 100,
        full_generation_max_tokens: int | None = None,
        prompt_template: str | None = None,
        schema: Any | None = None,
    ) -> None:
        """Initialize ET2RAG pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            llm: LangChain BaseLanguageModel instance for text generation.
            retrieval_pipeline: Retrieval pipeline for fetching relevant context.
            embedding_model: Embedding model for computing semantic similarity.
            organization_strategy: Strategy for subset creation.
            num_subsets: Number of subsets (None = auto based on strategy).
            partial_generation_max_tokens: Token limit for partial responses.
            full_generation_max_tokens: Token limit for final response.
            prompt_template: Custom prompt template with {context} and {query} placeholders.
                If None, uses DEFAULT_PROMPT_TEMPLATE.
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        # Store parameters BEFORE calling super().__init__()
        # Required for _get_pipeline_config() to work correctly
        self._embedding_model = embedding_model
        self._organization_strategy = organization_strategy
        self._num_subsets = num_subsets
        self._partial_generation_max_tokens = partial_generation_max_tokens
        self._full_generation_max_tokens = full_generation_max_tokens
        self._prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE

        super().__init__(
            session_factory=session_factory,
            name=name,
            llm=llm,
            retrieval_pipeline=retrieval_pipeline,
            schema=schema,
        )

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return pipeline configuration for storage."""
        return {
            "name": self.name,
            "type": "et2rag",
            "llm": str(self._llm),
            "retrieval_pipeline": self._retrieval_pipeline.name,
            "embedding_model": str(self._embedding_model),
            "organization_strategy": self._organization_strategy.value,
            "num_subsets": self._num_subsets,
            "partial_generation_max_tokens": self._partial_generation_max_tokens,
            "full_generation_max_tokens": self._full_generation_max_tokens,
            "prompt_template": self._prompt_template,
        }

    def _generate(self, query: str, top_k: int) -> GenerationResult:
        """Execute ET2RAG generation for a single query.

        Algorithm:
        1. Retrieve documents (ranked by relevance)
        2. Fetch chunk contents from database
        3. Create context subsets based on organization strategy
        4. Generate partial responses for each subset (async, different prompts!)
        5. Compute similarity matrix between partial responses
        6. Majority voting to select best subset index
        7. Generate full response with selected subset
        8. Return result with comprehensive metadata

        Args:
            query: The query to answer.
            top_k: Number of documents to retrieve.

        Returns:
            GenerationResult with generated text and metadata.
        """
        # Step 1: Retrieve documents
        retrieval_results = self._retrieval_pipeline.retrieve(query, top_k)
        chunk_ids = [r["doc_id"] for r in retrieval_results]

        if not chunk_ids:
            return GenerationResult(
                text="I don't have enough information to answer this question.",
                token_usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                metadata={
                    "retrieval_chunk_ids": [],
                    "organization_strategy": self._organization_strategy.value,
                    "num_subsets": 0,
                    "selected_subset_index": -1,
                },
            )

        # Step 2: Fetch chunk contents
        chunk_contents = self._get_chunk_contents(chunk_ids)

        # Build ordered list of (chunk_id, content) tuples
        documents: list[tuple[int, str]] = []
        for chunk_id in chunk_ids:
            content = chunk_contents.get(chunk_id, "")
            if content:
                documents.append((chunk_id, content))

        if not documents:
            return GenerationResult(
                text="Retrieved documents had no content.",
                token_usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                metadata={
                    "retrieval_chunk_ids": chunk_ids,
                    "organization_strategy": self._organization_strategy.value,
                    "num_subsets": 0,
                    "selected_subset_index": -1,
                },
            )

        # Step 3: Create context subsets
        subsets = self._create_subsets(documents)

        if len(subsets) == 0:
            return GenerationResult(
                text="Could not create context subsets from retrieved documents.",
                token_usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                metadata={
                    "retrieval_chunk_ids": chunk_ids,
                    "organization_strategy": self._organization_strategy.value,
                    "num_subsets": 0,
                    "selected_subset_index": -1,
                },
            )

        # Step 4: Generate partial responses for each subset
        partial_responses, partial_token_usages = asyncio.run(
            self._generate_partial_responses_async(query, subsets)
        )

        # Handle single subset case - skip voting
        if len(subsets) == 1:
            selected_index = 0
            similarity_matrix = [[1.0]]
            confidence_score = 1.0
        else:
            # Step 5: Compute similarity matrix
            similarity_matrix = self._compute_similarity_matrix(partial_responses)

            # Step 6: Majority voting to select best subset
            selected_index, confidence_score = self._majority_voting(similarity_matrix)

        # Step 7: Generate full response with selected subset
        selected_subset = subsets[selected_index]
        full_response, full_token_usage = self._generate_full_response(query, selected_subset)

        # Step 8: Aggregate token usage (all partial + full)
        total_token_usage = self._aggregate_token_usage(partial_token_usages + [full_token_usage])

        # Build metadata
        metadata = {
            "retrieval_chunk_ids": chunk_ids,
            "organization_strategy": self._organization_strategy.value,
            "num_subsets": len(subsets),
            "selected_subset_index": selected_index,
            "selected_subset_chunk_ids": [doc[0] for doc in selected_subset],
            "confidence_score": confidence_score,
            "partial_responses": partial_responses,
            "similarity_matrix": similarity_matrix,
            "partial_token_usages": partial_token_usages,
            "full_token_usage": full_token_usage,
        }

        return GenerationResult(
            text=full_response,
            token_usage=total_token_usage,
            metadata=metadata,
        )

    def _create_subsets(
        self, documents: list[tuple[int, str]]
    ) -> list[list[tuple[int, str]]]:
        """Create context subsets based on organization strategy.

        Args:
            documents: List of (chunk_id, content) tuples, ordered by relevance.

        Returns:
            List of subsets, where each subset is a list of (chunk_id, content) tuples.
        """
        strategy_handlers = {
            OrganizationStrategy.QA: self._create_qa_subsets,
            OrganizationStrategy.RECIPE: self._create_recipe_subsets,
            OrganizationStrategy.IMAGE: self._create_image_subsets,
        }

        handler = strategy_handlers[self._organization_strategy]
        return handler(documents)

    def _create_qa_subsets(
        self, documents: list[tuple[int, str]]
    ) -> list[list[tuple[int, str]]]:
        """Create QA-style subsets: always include top1 + one additional.

        For factoid QA, top1 is usually relevant. Subsets explore different
        combinations: {top1}, {top1,top2}, {top1,top3}, ...

        Args:
            documents: Ranked documents.

        Returns:
            QA-style subsets.
        """
        if not documents:
            return []

        num_subsets = self._num_subsets or min(len(documents), 5)
        subsets: list[list[tuple[int, str]]] = []

        # First subset: just top1
        subsets.append([documents[0]])

        # Additional subsets: top1 + one from remaining
        for i in range(1, min(num_subsets, len(documents))):
            subsets.append([documents[0], documents[i]])

        return subsets

    def _create_recipe_subsets(
        self, documents: list[tuple[int, str]]
    ) -> list[list[tuple[int, str]]]:
        """Create Recipe-style subsets: single document per subset.

        For long documents (recipes, articles), each document is self-contained.
        Subsets: {top1}, {top2}, {top3}, ...

        Args:
            documents: Ranked documents.

        Returns:
            Recipe-style subsets.
        """
        if not documents:
            return []

        num_subsets = self._num_subsets or min(len(documents), 5)
        return [[doc] for doc in documents[:num_subsets]]

    def _create_image_subsets(
        self, documents: list[tuple[int, str]]
    ) -> list[list[tuple[int, str]]]:
        """Create Image-style subsets: 4 captions per subset from top-20.

        For image captioning (COCO), multiple captions provide context.
        Subsets use different combinations of 4 captions.

        Args:
            documents: Ranked documents (captions).

        Returns:
            Image-style subsets.
        """
        if not documents:
            return []

        # Limit to top 20 captions
        docs = documents[:20]
        num_subsets = self._num_subsets or 5
        subsets: list[list[tuple[int, str]]] = []

        # Create subsets with 4 captions each using different patterns
        patterns = [
            [0, 1, 2, 3],  # First 4
            [0, 1, 4, 5],  # Mix early
            [0, 2, 4, 6],  # Every other
            [0, 3, 6, 9],  # Spread out
            [1, 3, 5, 7],  # Odd positions
        ]

        for pattern in patterns[:num_subsets]:
            subset: list[tuple[int, str]] = []
            for idx in pattern:
                if idx < len(docs):
                    subset.append(docs[idx])
            if subset:  # Only add non-empty subsets
                subsets.append(subset)

        # If we couldn't create enough subsets, fall back to simple chunking
        if not subsets:
            chunk_size = 4
            for i in range(0, len(docs), chunk_size):
                chunk = docs[i : i + chunk_size]
                if chunk:
                    subsets.append(chunk)
                if len(subsets) >= num_subsets:
                    break

        return subsets

    async def _generate_partial_responses_async(
        self, query: str, subsets: list[list[tuple[int, str]]]
    ) -> tuple[list[str], list[dict]]:
        """Generate partial responses for each subset asynchronously.

        Each subset gets a DIFFERENT prompt because the context is different.
        This is the key difference from naive ensemble methods.

        Args:
            query: The query to answer.
            subsets: List of context subsets.

        Returns:
            Tuple of (list of responses, list of token usages).
        """
        tasks = []
        for subset in subsets:
            prompt = self._build_prompt(query, subset)
            tasks.append(self._generate_single_async(prompt, max_tokens=self._partial_generation_max_tokens))

        results = await asyncio.gather(*tasks)

        responses = [r[0] for r in results]
        token_usages = [r[1] for r in results]

        return responses, token_usages

    async def _generate_single_async(
        self, prompt: str, max_tokens: int | None = None
    ) -> tuple[str, dict]:
        """Generate a single response asynchronously.

        Args:
            prompt: The prompt to generate from.
            max_tokens: Optional token limit.

        Returns:
            Tuple of (response text, token usage dict).
        """
        # Configure generation parameters
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = await self._llm.ainvoke(prompt, **kwargs)
        text = response.content if hasattr(response, "content") else str(response)
        token_usage = self._extract_token_usage(response)

        return text, token_usage

    def _generate_full_response(
        self, query: str, subset: list[tuple[int, str]]
    ) -> tuple[str, dict]:
        """Generate the final full response with the selected subset.

        This is the SECOND stage of generation, after voting has selected
        the best context subset. Uses full token budget.

        Args:
            query: The query to answer.
            subset: The selected context subset.

        Returns:
            Tuple of (response text, token usage dict).
        """
        prompt = self._build_prompt(query, subset)

        kwargs: dict[str, Any] = {}
        if self._full_generation_max_tokens is not None:
            kwargs["max_tokens"] = self._full_generation_max_tokens

        response = self._llm.invoke(prompt, **kwargs)
        text = response.content if hasattr(response, "content") else str(response)
        token_usage = self._extract_token_usage(response)

        return text, token_usage

    def _build_prompt(self, query: str, subset: list[tuple[int, str]]) -> str:
        """Build a prompt from query and context subset.

        Uses the configured prompt_template with {context} and {query} placeholders.

        Args:
            query: The query to answer.
            subset: List of (chunk_id, content) tuples.

        Returns:
            Formatted prompt string.
        """
        context_parts = []
        for i, (_, content) in enumerate(subset, 1):
            context_parts.append(f"[Document {i}]\n{content}")

        context = "\n\n".join(context_parts)

        return self._prompt_template.format(context=context, query=query)

    def _compute_similarity_matrix(self, responses: list[str]) -> list[list[float]]:
        """Compute pairwise cosine similarity matrix between responses.

        Args:
            responses: List of response texts.

        Returns:
            NxN similarity matrix as nested lists.
        """
        if not responses:
            return [[]]

        # Embed all responses
        embeddings = self._embedding_model.embed_documents(responses)
        embeddings_array = np.array(embeddings)

        n = len(responses)
        similarity_matrix: list[list[float]] = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    similarity_matrix[i][j] = float(
                        calculate_cosine_similarity(embeddings_array[i], embeddings_array[j])
                    )

        return similarity_matrix

    def _majority_voting(self, similarity_matrix: list[list[float]]) -> tuple[int, float]:
        """Select the best subset index using majority voting.

        The candidate with highest total similarity to others wins.
        This identifies the most "central" or "consensus" response.

        Args:
            similarity_matrix: NxN similarity matrix.

        Returns:
            Tuple of (selected index, confidence score).
        """
        n = len(similarity_matrix)

        if n == 0:
            return 0, 0.0

        if n == 1:
            return 0, 1.0

        # Sum similarities (excluding self-similarity on diagonal)
        scores = []
        for i in range(n):
            score = sum(similarity_matrix[i][j] for j in range(n) if i != j)
            scores.append(score)

        selected_index = int(np.argmax(scores))

        # Confidence: how much better is the winner vs average?
        max_score = scores[selected_index]
        avg_score = sum(scores) / len(scores)
        confidence = max_score / avg_score if avg_score > 0 else 1.0

        return selected_index, confidence

    def _get_chunk_contents(self, chunk_ids: list[int]) -> dict[int, str]:
        """Fetch chunk contents from database.

        Args:
            chunk_ids: List of chunk IDs to fetch.

        Returns:
            Dict mapping chunk_id to content.
        """
        with GenerationUnitOfWork(self.session_factory, self._schema) as uow:
            chunks = uow.chunks.get_by_ids(chunk_ids)
            return {chunk.id: chunk.contents for chunk in chunks}

    def _aggregate_token_usage(self, token_usages: list[dict]) -> dict:
        """Aggregate multiple token usages into one.

        Args:
            token_usages: List of token usage dicts.

        Returns:
            Aggregated token usage dict.
        """
        total_prompt = sum(tu.get("prompt_tokens", 0) for tu in token_usages)
        total_completion = sum(tu.get("completion_tokens", 0) for tu in token_usages)
        total = sum(tu.get("total_tokens", 0) for tu in token_usages)

        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total,
        }

    def _extract_token_usage(self, response: Any) -> dict:
        """Extract token usage from LangChain response.

        Args:
            response: LLM response object.

        Returns:
            Dictionary with prompt_tokens, completion_tokens, total_tokens.
        """
        # Try to get usage from response metadata (LangChain style)
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            return {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        elif hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("token_usage", {})
            if usage:
                return {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

        # Default to zeros if no usage info available
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
