"""Test cases for ET2RAGPipeline.

Tests the corrected ET2RAG (Efficient Test-Time Retrieval Augmented Generation) pipeline
which implements subset-based majority voting:
1. Creates context subsets from retrieved documents
2. Generates PARTIAL responses for each subset (different prompts!)
3. Uses semantic similarity voting to select the BEST SUBSET
4. Generates a FULL response with the selected subset

Tests are written in TDD style based on the corrected algorithm.
"""

import math
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.embeddings.fake import FakeEmbeddings
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.executor_result import ExecutorResultRepository
from autorag_research.pipelines.generation.et2rag import (
    ET2RAGPipeline,
    OrganizationStrategy,
)
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
    create_mock_llm,
)


class TestOrganizationStrategy:
    """Tests for OrganizationStrategy enum."""

    def test_strategy_values(self):
        """Test that all strategies have correct string values."""
        assert OrganizationStrategy.QA.value == "qa"
        assert OrganizationStrategy.RECIPE.value == "recipe"
        assert OrganizationStrategy.IMAGE.value == "image"

    def test_strategy_from_string(self):
        """Test creating strategy from string value."""
        assert OrganizationStrategy("qa") == OrganizationStrategy.QA
        assert OrganizationStrategy("recipe") == OrganizationStrategy.RECIPE
        assert OrganizationStrategy("image") == OrganizationStrategy.IMAGE


class TestSubsetCreation:
    """Tests for subset creation strategies."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM with async support."""
        mock = create_mock_llm(
            response_text="Generated answer.",
            token_usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        )
        mock.ainvoke = AsyncMock(return_value=mock.invoke.return_value)
        return mock

    @pytest.fixture
    def mock_embeddings(self):
        """Create fake embeddings for similarity computation."""
        return FakeEmbeddings(size=384)

    @pytest.fixture
    def mock_retrieval_pipeline(self):
        """Create a mock retrieval pipeline."""
        mock = MagicMock()
        mock.pipeline_id = 1
        mock.name = "mock_retrieval_pipeline"
        mock.retrieve = MagicMock(return_value=[])
        return mock

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory: sessionmaker[Session]):
        """Cleanup fixture that deletes pipeline results after test."""
        created_pipeline_ids: list[int] = []

        yield created_pipeline_ids

        session = session_factory()
        try:
            executor_repo = ExecutorResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                executor_repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for subset testing."""
        return [
            (1, "Document 1 content about machine learning"),
            (2, "Document 2 content about neural networks"),
            (3, "Document 3 content about deep learning"),
            (4, "Document 4 content about transformers"),
            (5, "Document 5 content about attention"),
            (6, "Document 6 content about embeddings"),
            (7, "Document 7 content about tokenization"),
            (8, "Document 8 content about fine-tuning"),
            (9, "Document 9 content about pre-training"),
            (10, "Document 10 content about inference"),
        ]

    def test_create_qa_subsets(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_retrieval_pipeline,
        mock_embeddings,
        cleanup_pipeline_results: list[int],
        sample_documents,
    ):
        """Test QA strategy: {top1}, {top1,top2}, {top1,top3}, ..."""
        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_qa_subsets",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            organization_strategy=OrganizationStrategy.QA,
            num_subsets=5,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        subsets = pipeline._create_qa_subsets(sample_documents)

        # Should create 5 subsets
        assert len(subsets) == 5

        # First subset: just top1
        assert len(subsets[0]) == 1
        assert subsets[0][0][0] == 1  # chunk_id 1

        # Remaining subsets: top1 + one additional
        for i in range(1, 5):
            assert len(subsets[i]) == 2
            assert subsets[i][0][0] == 1  # Always includes top1
            assert subsets[i][1][0] == i + 1  # Plus document at index i

    def test_create_qa_subsets_auto_num(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_retrieval_pipeline,
        mock_embeddings,
        cleanup_pipeline_results: list[int],
    ):
        """Test QA strategy with auto num_subsets (None)."""
        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_qa_auto_num",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            organization_strategy=OrganizationStrategy.QA,
            num_subsets=None,  # Auto
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        # With 3 documents, should create min(3, 5) = 3 subsets
        documents = [(1, "A"), (2, "B"), (3, "C")]
        subsets = pipeline._create_qa_subsets(documents)

        assert len(subsets) == 3

    def test_create_recipe_subsets(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_retrieval_pipeline,
        mock_embeddings,
        cleanup_pipeline_results: list[int],
        sample_documents,
    ):
        """Test Recipe strategy: {top1}, {top2}, {top3}, ..."""
        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_recipe_subsets",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            organization_strategy=OrganizationStrategy.RECIPE,
            num_subsets=5,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        subsets = pipeline._create_recipe_subsets(sample_documents)

        # Should create 5 subsets, each with 1 document
        assert len(subsets) == 5

        # Each subset should have exactly 1 document
        for i, subset in enumerate(subsets):
            assert len(subset) == 1
            assert subset[0][0] == i + 1  # chunk_id matches position

    def test_create_image_subsets(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_retrieval_pipeline,
        mock_embeddings,
        cleanup_pipeline_results: list[int],
        sample_documents,
    ):
        """Test Image strategy: 4 captions per subset."""
        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_image_subsets",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            organization_strategy=OrganizationStrategy.IMAGE,
            num_subsets=5,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        subsets = pipeline._create_image_subsets(sample_documents)

        # Should create multiple subsets
        assert len(subsets) >= 1

        # Each subset should have up to 4 documents
        for subset in subsets:
            assert len(subset) <= 4
            assert len(subset) >= 1

        # First subset should use pattern [0,1,2,3]
        first_subset_ids = [doc[0] for doc in subsets[0]]
        assert first_subset_ids == [1, 2, 3, 4]

    def test_create_subsets_empty_documents(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_retrieval_pipeline,
        mock_embeddings,
        cleanup_pipeline_results: list[int],
    ):
        """Test subset creation with empty document list."""
        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_empty_docs",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            organization_strategy=OrganizationStrategy.QA,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        subsets = pipeline._create_subsets([])
        assert subsets == []


class TestET2RAGPipelineUnit:
    """Unit tests for ET2RAGPipeline core algorithm."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM with async support."""
        mock = create_mock_llm(
            response_text="Generated answer.",
            token_usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        )
        mock.ainvoke = AsyncMock(return_value=mock.invoke.return_value)
        return mock

    @pytest.fixture
    def mock_embeddings(self):
        """Create fake embeddings for similarity computation."""
        return FakeEmbeddings(size=384)

    @pytest.fixture
    def mock_retrieval_pipeline(self):
        """Create a mock retrieval pipeline."""
        mock = MagicMock()
        mock.pipeline_id = 1
        mock.name = "mock_retrieval_pipeline"

        def mock_retrieve(query_text: str, top_k: int):
            return [
                {"doc_id": 1, "score": 0.9},
                {"doc_id": 2, "score": 0.8},
                {"doc_id": 3, "score": 0.7},
                {"doc_id": 4, "score": 0.6},
                {"doc_id": 5, "score": 0.5},
            ][:top_k]

        mock.retrieve = mock_retrieve
        return mock

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory: sessionmaker[Session]):
        """Cleanup fixture that deletes pipeline results after test."""
        created_pipeline_ids: list[int] = []

        yield created_pipeline_ids

        session = session_factory()
        try:
            executor_repo = ExecutorResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                executor_repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    @pytest.fixture
    def pipeline(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_retrieval_pipeline,
        mock_embeddings,
        cleanup_pipeline_results: list[int],
    ):
        """Create an ET2RAGPipeline instance for testing."""
        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_et2rag_pipeline",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            organization_strategy=OrganizationStrategy.QA,
            num_subsets=3,
            partial_generation_max_tokens=100,
            full_generation_max_tokens=500,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)
        return pipeline

    def test_init_with_default_params(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_retrieval_pipeline,
        mock_embeddings,
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline is created with correct default parameter values."""
        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_et2rag_default_params",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline.pipeline_id > 0
        assert pipeline._organization_strategy == OrganizationStrategy.QA
        assert pipeline._num_subsets is None  # auto-determine
        assert pipeline._partial_generation_max_tokens == 100
        assert pipeline._full_generation_max_tokens is None
        assert pipeline._embedding_model == mock_embeddings

    def test_init_with_custom_params(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_retrieval_pipeline,
        mock_embeddings,
        cleanup_pipeline_results: list[int],
    ):
        """Test that custom parameters are stored correctly."""
        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_et2rag_custom_params",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            organization_strategy=OrganizationStrategy.RECIPE,
            num_subsets=7,
            partial_generation_max_tokens=200,
            full_generation_max_tokens=1000,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline._organization_strategy == OrganizationStrategy.RECIPE
        assert pipeline._num_subsets == 7
        assert pipeline._partial_generation_max_tokens == 200
        assert pipeline._full_generation_max_tokens == 1000

    def test_get_pipeline_config(self, pipeline, mock_retrieval_pipeline, mock_embeddings):
        """Test that pipeline config dict contains all expected fields."""
        config = pipeline._get_pipeline_config()

        assert config["name"] == "test_et2rag_pipeline"
        assert config["type"] == "et2rag"
        assert "llm" in config
        assert config["retrieval_pipeline"] == mock_retrieval_pipeline.name
        assert "embedding_model" in config
        assert config["organization_strategy"] == "qa"
        assert config["num_subsets"] == 3
        assert config["partial_generation_max_tokens"] == 100
        assert config["full_generation_max_tokens"] == 500
        assert "prompt_template" in config

    def test_build_prompt_single_document(self, pipeline):
        """Test prompt building with single document."""
        subset = [(1, "Document content here")]
        prompt = pipeline._build_prompt("What is AI?", subset)

        assert "What is AI?" in prompt
        assert "Document content here" in prompt
        assert "[Document 1]" in prompt

    def test_build_prompt_multiple_documents(self, pipeline):
        """Test prompt building with multiple documents."""
        subset = [
            (1, "First document"),
            (2, "Second document"),
            (3, "Third document"),
        ]
        prompt = pipeline._build_prompt("Test query", subset)

        assert "Test query" in prompt
        assert "[Document 1]" in prompt
        assert "[Document 2]" in prompt
        assert "[Document 3]" in prompt
        assert "First document" in prompt
        assert "Second document" in prompt
        assert "Third document" in prompt

    def test_build_prompt_with_custom_template(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_retrieval_pipeline,
        mock_embeddings,
        cleanup_pipeline_results: list[int],
    ):
        """Test prompt building with custom template."""
        custom_template = """You are a helpful assistant.

Based on the following documents:
{context}

Please answer: {query}"""

        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_custom_prompt",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            prompt_template=custom_template,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        subset = [(1, "Test content")]
        prompt = pipeline._build_prompt("What is AI?", subset)

        assert "You are a helpful assistant" in prompt
        assert "Based on the following documents:" in prompt
        assert "Please answer: What is AI?" in prompt
        assert "Test content" in prompt

    def test_default_prompt_template_used_when_none(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_retrieval_pipeline,
        mock_embeddings,
        cleanup_pipeline_results: list[int],
    ):
        """Test that default template is used when prompt_template is None."""
        from autorag_research.pipelines.generation.et2rag import DEFAULT_PROMPT_TEMPLATE

        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_default_template",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            prompt_template=None,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline._prompt_template == DEFAULT_PROMPT_TEMPLATE

    def test_compute_similarity_matrix(self, pipeline):
        """Test similarity matrix computation between responses."""
        responses = [
            "Machine learning is a field of AI.",
            "Deep learning uses neural networks.",
            "AI is artificial intelligence.",
        ]

        similarity_matrix = pipeline._compute_similarity_matrix(responses)

        # Verify matrix dimensions (NxN)
        n = len(responses)
        assert len(similarity_matrix) == n
        assert all(len(row) == n for row in similarity_matrix)

        # Verify diagonal elements are 1.0 (self-similarity)
        assert all(similarity_matrix[i][i] == 1.0 for i in range(n))

        # Verify symmetry (similarity[i][j] == similarity[j][i])
        for i in range(n):
            for j in range(n):
                assert similarity_matrix[i][j] == similarity_matrix[j][i]

        # Verify values are finite
        for row in similarity_matrix:
            assert all(math.isfinite(v) for v in row)

    def test_majority_voting(self, pipeline):
        """Test majority voting selects subset with highest consensus."""
        # Create a similarity matrix where subset 1 has highest consensus
        similarity_matrix = [
            [1.0, 0.8, 0.2],  # Subset 0: sum = 1.0 (exclude self)
            [0.8, 1.0, 0.9],  # Subset 1: sum = 1.7 (highest)
            [0.2, 0.9, 1.0],  # Subset 2: sum = 1.1
        ]

        selected_idx, confidence = pipeline._majority_voting(similarity_matrix)

        # Subset 1 should be selected (highest sum of similarities)
        assert selected_idx == 1
        assert confidence > 1.0  # Winner is above average

    def test_majority_voting_single_subset(self, pipeline):
        """Test majority voting with single subset (edge case)."""
        similarity_matrix = [[1.0]]

        selected_idx, confidence = pipeline._majority_voting(similarity_matrix)

        assert selected_idx == 0
        assert confidence == 1.0

    def test_aggregate_token_usage(self, pipeline):
        """Test token usage aggregation across partial + full generations."""
        token_usages = [
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},  # partial 1
            {"prompt_tokens": 100, "completion_tokens": 60, "total_tokens": 160},  # partial 2
            {"prompt_tokens": 100, "completion_tokens": 55, "total_tokens": 155},  # partial 3
            {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},  # full
        ]

        aggregated = pipeline._aggregate_token_usage(token_usages)

        assert aggregated["prompt_tokens"] == 500
        assert aggregated["completion_tokens"] == 265
        assert aggregated["total_tokens"] == 765

    def test_aggregate_token_usage_empty(self, pipeline):
        """Test token usage aggregation with empty list."""
        aggregated = pipeline._aggregate_token_usage([])

        assert aggregated["prompt_tokens"] == 0
        assert aggregated["completion_tokens"] == 0
        assert aggregated["total_tokens"] == 0


class TestDifferentPromptsForSubsets:
    """Tests verifying that each subset gets a DIFFERENT prompt."""

    @pytest.fixture
    def mock_embeddings(self):
        """Create fake embeddings."""
        return FakeEmbeddings(size=384)

    @pytest.fixture
    def mock_retrieval_pipeline(self):
        """Create a mock retrieval pipeline."""
        mock = MagicMock()
        mock.pipeline_id = 1
        mock.name = "mock_retrieval"
        mock.retrieve = MagicMock(
            return_value=[
                {"doc_id": 1, "score": 0.9},
                {"doc_id": 2, "score": 0.8},
                {"doc_id": 3, "score": 0.7},
            ]
        )
        return mock

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory: sessionmaker[Session]):
        """Cleanup fixture."""
        created_pipeline_ids: list[int] = []
        yield created_pipeline_ids
        session = session_factory()
        try:
            executor_repo = ExecutorResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                executor_repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_different_prompts_for_each_subset(
        self,
        session_factory: sessionmaker[Session],
        mock_embeddings,
        mock_retrieval_pipeline,
        cleanup_pipeline_results: list[int],
    ):
        """Test that each subset gets a unique prompt with different context."""
        # Track all prompts passed to ainvoke
        captured_prompts: list[str] = []

        mock_llm = create_mock_llm(
            response_text="Answer",
            token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        async def capture_ainvoke(prompt, **kwargs):
            captured_prompts.append(prompt)
            return mock_llm.invoke.return_value

        mock_llm.ainvoke = capture_ainvoke

        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_different_prompts",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            organization_strategy=OrganizationStrategy.QA,
            num_subsets=3,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        # Create subsets manually
        documents = [
            (1, "Content A"),
            (2, "Content B"),
            (3, "Content C"),
        ]
        subsets = pipeline._create_qa_subsets(documents)

        # Generate partial responses
        await pipeline._generate_partial_responses_async("Test query", subsets)

        # Verify we got 3 different prompts (one per subset)
        assert len(captured_prompts) == 3

        # Verify prompts are DIFFERENT (different contexts!)
        assert len(set(captured_prompts)) == 3, "Each subset should produce a unique prompt"

        # Verify first prompt has only Document 1
        assert "[Document 1]" in captured_prompts[0]
        assert "[Document 2]" not in captured_prompts[0]

        # Verify second prompt has Documents 1 and 2
        assert "[Document 1]" in captured_prompts[1]
        assert "[Document 2]" in captured_prompts[1]


class TestVotingSelectsSubset:
    """Tests verifying that voting selects SUBSET, not response."""

    @pytest.fixture
    def mock_embeddings(self):
        """Create fake embeddings."""
        return FakeEmbeddings(size=384)

    @pytest.fixture
    def mock_retrieval_pipeline(self):
        """Create a mock retrieval pipeline."""
        mock = MagicMock()
        mock.pipeline_id = 1
        mock.name = "mock_retrieval"
        mock.retrieve = MagicMock(
            return_value=[
                {"doc_id": 1, "score": 0.9},
                {"doc_id": 2, "score": 0.8},
                {"doc_id": 3, "score": 0.7},
            ]
        )
        return mock

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory: sessionmaker[Session]):
        """Cleanup fixture."""
        created_pipeline_ids: list[int] = []
        yield created_pipeline_ids
        session = session_factory()
        try:
            executor_repo = ExecutorResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                executor_repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    def test_voting_returns_subset_index(
        self,
        session_factory: sessionmaker[Session],
        mock_embeddings,
        mock_retrieval_pipeline,
        cleanup_pipeline_results: list[int],
    ):
        """Test that voting returns an index into subsets, not responses."""
        mock_llm = create_mock_llm(
            response_text="Answer",
            token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm.invoke.return_value)

        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_voting_index",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            organization_strategy=OrganizationStrategy.QA,
            num_subsets=3,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        # Simulate voting
        similarity_matrix = [
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.8],  # Subset 1 has highest similarity sum
            [0.3, 0.8, 1.0],
        ]

        selected_idx, _ = pipeline._majority_voting(similarity_matrix)

        # Selected index should be usable to index into subsets
        assert 0 <= selected_idx < 3

    def test_metadata_contains_selected_subset_info(
        self,
        session_factory: sessionmaker[Session],
        mock_embeddings,
        mock_retrieval_pipeline,
        cleanup_pipeline_results: list[int],
    ):
        """Test that result metadata contains info about selected subset."""
        mock_llm = create_mock_llm(
            response_text="Answer",
            token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm.invoke.return_value)

        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_subset_metadata",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            organization_strategy=OrganizationStrategy.QA,
            num_subsets=3,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Test query", top_k=3)

        # Verify metadata has subset-related fields
        assert "selected_subset_index" in result.metadata
        assert "selected_subset_chunk_ids" in result.metadata
        assert "num_subsets" in result.metadata
        assert "organization_strategy" in result.metadata

        # Verify types
        assert isinstance(result.metadata["selected_subset_index"], int)
        assert isinstance(result.metadata["selected_subset_chunk_ids"], list)


class TestFullGenerationWithSelectedSubset:
    """Tests verifying that full generation happens AFTER voting with selected subset."""

    @pytest.fixture
    def mock_embeddings(self):
        """Create fake embeddings."""
        return FakeEmbeddings(size=384)

    @pytest.fixture
    def mock_retrieval_pipeline(self):
        """Create a mock retrieval pipeline."""
        mock = MagicMock()
        mock.pipeline_id = 1
        mock.name = "mock_retrieval"
        mock.retrieve = MagicMock(
            return_value=[
                {"doc_id": 1, "score": 0.9},
                {"doc_id": 2, "score": 0.8},
                {"doc_id": 3, "score": 0.7},
            ]
        )
        return mock

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory: sessionmaker[Session]):
        """Cleanup fixture."""
        created_pipeline_ids: list[int] = []
        yield created_pipeline_ids
        session = session_factory()
        try:
            executor_repo = ExecutorResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                executor_repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    def test_full_generation_called_after_voting(
        self,
        session_factory: sessionmaker[Session],
        mock_embeddings,
        mock_retrieval_pipeline,
        cleanup_pipeline_results: list[int],
    ):
        """Test that sync invoke (full generation) is called after async ainvoke (partial)."""
        invoke_calls: list[str] = []

        mock_llm = create_mock_llm(
            response_text="Full answer",
            token_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )

        original_invoke = mock_llm.invoke

        def tracking_invoke(prompt, **kwargs):
            invoke_calls.append(("invoke", prompt[:50]))
            return original_invoke(prompt, **kwargs)

        async def tracking_ainvoke(prompt, **kwargs):
            invoke_calls.append(("ainvoke", prompt[:50]))
            return original_invoke(prompt, **kwargs)

        mock_llm.invoke = tracking_invoke
        mock_llm.ainvoke = tracking_ainvoke

        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_full_gen",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            organization_strategy=OrganizationStrategy.QA,
            num_subsets=3,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        pipeline._generate("Test query", top_k=3)

        # Should have 3 ainvoke calls (partial) + 1 invoke call (full)
        ainvoke_calls = [c for c in invoke_calls if c[0] == "ainvoke"]
        sync_invoke_calls = [c for c in invoke_calls if c[0] == "invoke"]

        assert len(ainvoke_calls) == 3, "Should have 3 partial generation calls"
        assert len(sync_invoke_calls) == 1, "Should have 1 full generation call"

        # Full generation should be LAST
        assert invoke_calls[-1][0] == "invoke", "Full generation should be last call"

    def test_token_usage_includes_partial_and_full(
        self,
        session_factory: sessionmaker[Session],
        mock_embeddings,
        mock_retrieval_pipeline,
        cleanup_pipeline_results: list[int],
    ):
        """Test that token usage aggregates both partial and full generations."""
        mock_llm = create_mock_llm(
            response_text="Answer",
            token_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm.invoke.return_value)

        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_token_aggregation",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            organization_strategy=OrganizationStrategy.QA,
            num_subsets=3,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Test query", top_k=3)

        # 3 partial + 1 full = 4 calls * 150 tokens each = 600 total
        assert result.token_usage["total_tokens"] == 600
        assert result.token_usage["prompt_tokens"] == 400
        assert result.token_usage["completion_tokens"] == 200

    def test_metadata_contains_partial_and_full_token_usage(
        self,
        session_factory: sessionmaker[Session],
        mock_embeddings,
        mock_retrieval_pipeline,
        cleanup_pipeline_results: list[int],
    ):
        """Test that metadata separates partial and full token usage."""
        mock_llm = create_mock_llm(
            response_text="Answer",
            token_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm.invoke.return_value)

        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_separated_token_usage",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            organization_strategy=OrganizationStrategy.QA,
            num_subsets=3,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Test query", top_k=3)

        # Metadata should have separate fields
        assert "partial_token_usages" in result.metadata
        assert "full_token_usage" in result.metadata

        # Should have 3 partial usages
        assert len(result.metadata["partial_token_usages"]) == 3

        # Full usage should be a single dict
        assert isinstance(result.metadata["full_token_usage"], dict)


class TestET2RAGPipelineEdgeCases:
    """Edge case tests for ET2RAGPipeline."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM with async support."""
        mock = create_mock_llm(
            response_text="Edge case answer.",
            token_usage={
                "prompt_tokens": 50,
                "completion_tokens": 25,
                "total_tokens": 75,
            },
        )
        mock.ainvoke = AsyncMock(return_value=mock.invoke.return_value)
        return mock

    @pytest.fixture
    def mock_embeddings(self):
        """Create fake embeddings."""
        return FakeEmbeddings(size=384)

    @pytest.fixture
    def mock_retrieval_pipeline(self):
        """Create a mock retrieval pipeline."""
        mock = MagicMock()
        mock.pipeline_id = 1
        mock.name = "mock_retrieval_edge"

        def mock_retrieve(query_text: str, top_k: int):
            return [{"doc_id": 1, "score": 0.9}]

        mock.retrieve = mock_retrieve
        return mock

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory: sessionmaker[Session]):
        """Cleanup fixture."""
        created_pipeline_ids: list[int] = []

        yield created_pipeline_ids

        session = session_factory()
        try:
            executor_repo = ExecutorResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                executor_repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    def test_single_subset(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_retrieval_pipeline,
        mock_embeddings,
        cleanup_pipeline_results: list[int],
    ):
        """Test pipeline with single subset (no voting needed)."""
        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_et2rag_single_subset",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            organization_strategy=OrganizationStrategy.QA,
            num_subsets=1,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Test query", top_k=1)

        # Single subset should be selected directly
        assert result.text is not None
        assert result.metadata["num_subsets"] == 1
        assert result.metadata["selected_subset_index"] == 0
        assert result.metadata["confidence_score"] == 1.0

    def test_empty_retrieval_results(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_embeddings,
        cleanup_pipeline_results: list[int],
    ):
        """Test handling of empty retrieval results."""
        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 999
        mock_retrieval.name = "empty_retrieval"
        mock_retrieval.retrieve.return_value = []

        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_et2rag_empty_retrieval",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
            embedding_model=mock_embeddings,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Query with no results", top_k=5)

        # Should produce a fallback result
        assert result.text is not None
        assert result.metadata["retrieval_chunk_ids"] == []
        assert result.metadata["num_subsets"] == 0
        assert result.metadata["selected_subset_index"] == -1

    def test_identical_partial_responses(
        self,
        session_factory: sessionmaker[Session],
        mock_retrieval_pipeline,
        mock_embeddings,
        cleanup_pipeline_results: list[int],
    ):
        """Test when all partial responses are identical (high consensus)."""
        mock_llm = create_mock_llm(
            response_text="Identical answer from all subsets.",
            token_usage={
                "prompt_tokens": 50,
                "completion_tokens": 25,
                "total_tokens": 75,
            },
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm.invoke.return_value)

        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_et2rag_identical",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            organization_strategy=OrganizationStrategy.RECIPE,
            num_subsets=3,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        # Extend mock to return more documents
        mock_retrieval_pipeline.retrieve = MagicMock(
            return_value=[
                {"doc_id": 1, "score": 0.9},
                {"doc_id": 2, "score": 0.8},
                {"doc_id": 3, "score": 0.7},
            ]
        )

        result = pipeline._generate("Test query", top_k=3)

        # All partial responses should be identical
        partial_responses = result.metadata["partial_responses"]
        assert all(r == partial_responses[0] for r in partial_responses)

        # Result should still be valid
        assert result.text is not None
        assert result.metadata["confidence_score"] >= 0.0


class TestET2RAGPipelineIntegration:
    """Integration tests for ET2RAGPipeline."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM with async support for integration tests."""
        mock = create_mock_llm(
            response_text="Generated answer for integration test.",
            token_usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        )
        mock.ainvoke = AsyncMock(return_value=mock.invoke.return_value)
        return mock

    @pytest.fixture
    def mock_embeddings(self):
        """Create fake embeddings for similarity computation."""
        return FakeEmbeddings(size=384)

    @pytest.fixture
    def mock_retrieval_pipeline(self):
        """Create a mock retrieval pipeline."""
        mock = MagicMock()
        mock.pipeline_id = 1
        mock.name = "mock_retrieval_integration"

        def mock_retrieve(query_text: str, top_k: int):
            return [
                {"doc_id": 1, "score": 0.9},
                {"doc_id": 2, "score": 0.8},
                {"doc_id": 3, "score": 0.7},
            ][:top_k]

        mock.retrieve = mock_retrieve
        return mock

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory: sessionmaker[Session]):
        """Cleanup fixture that deletes pipeline results after test."""
        created_pipeline_ids: list[int] = []

        yield created_pipeline_ids

        session = session_factory()
        try:
            executor_repo = ExecutorResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                executor_repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    @pytest.fixture
    def pipeline(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_retrieval_pipeline,
        mock_embeddings,
        cleanup_pipeline_results: list[int],
    ):
        """Create an ET2RAGPipeline instance for integration testing."""
        pipeline = ET2RAGPipeline(
            session_factory=session_factory,
            name="test_et2rag_integration",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            embedding_model=mock_embeddings,
            organization_strategy=OrganizationStrategy.QA,
            num_subsets=3,
            partial_generation_max_tokens=50,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)
        return pipeline

    def test_run_pipeline(self, pipeline, session_factory: sessionmaker[Session]):
        """Test running the full pipeline with PipelineTestVerifier."""
        result = pipeline.run(top_k=2, batch_size=10)

        config = PipelineTestConfig(
            pipeline_type="generation",
            expected_total_queries=5,  # Seed data has 5 queries
            check_token_usage=True,
            check_execution_time=True,
            check_persistence=True,
        )
        verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
        verifier.verify_all()

    def test_run_pipeline_token_aggregation(self, pipeline):
        """Test that token usage is correctly aggregated across all queries."""
        result = pipeline.run(top_k=2, batch_size=10)

        # With 5 queries, top_k=2, QA strategy creates 2 subsets (top1, top1+top2):
        # (2 partial + 1 full) * 5 queries * 150 = 2250 total tokens
        expected_total = 5 * 3 * 150
        assert result["token_usage"]["total_tokens"] == expected_total

    def test_metadata_contains_new_algorithm_fields(self, pipeline, session_factory: sessionmaker[Session]):
        """Test that executor results contain new ET2RAG algorithm metadata."""
        pipeline.run(top_k=2, batch_size=10)

        session = session_factory()
        try:
            repo = ExecutorResultRepository(session)
            results = repo.get_by_pipeline_id(pipeline.pipeline_id)

            assert len(results) == 5

            for result in results:
                metadata = result.result_metadata
                assert metadata is not None

                # Check for NEW ET2RAG metadata fields
                assert "organization_strategy" in metadata
                assert "num_subsets" in metadata
                assert "selected_subset_index" in metadata
                assert "selected_subset_chunk_ids" in metadata
                assert "partial_responses" in metadata
                assert "partial_token_usages" in metadata
                assert "full_token_usage" in metadata
                assert "confidence_score" in metadata
        finally:
            session.close()
