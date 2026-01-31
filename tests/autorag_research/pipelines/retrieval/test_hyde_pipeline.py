"""Test cases for HyDE (Hypothetical Document Embeddings) Retrieval Pipeline.

Tests the HyDE retrieval pipeline logic using mocked LLM and embedding models.
HyDE generates hypothetical documents from queries using an LLM, then embeds
those documents for vector search.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake import FakeListLLM
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.schema import Chunk
from autorag_research.pipelines.retrieval.hyde import (
    DEFAULT_HYDE_PROMPT_TEMPLATE,
    HyDEPipelineConfig,
    HyDERetrievalPipeline,
)
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
)


class TestHyDEPipelineConfig:
    """Tests for HyDEPipelineConfig dataclass."""

    def test_config_get_pipeline_class(self):
        """Test that config returns correct pipeline class."""
        llm = FakeListLLM(responses=["hypothetical document"])
        embedding = FakeEmbeddings(size=768)

        config = HyDEPipelineConfig(
            name="test_hyde",
            llm=llm,
            embedding=embedding,
        )

        assert config.get_pipeline_class() == HyDERetrievalPipeline

    def test_config_get_pipeline_kwargs(self):
        """Test that config returns correct pipeline kwargs."""
        llm = FakeListLLM(responses=["hypothetical document"])
        embedding = FakeEmbeddings(size=768)
        custom_template = "Custom template: {question}"

        config = HyDEPipelineConfig(
            name="test_hyde",
            llm=llm,
            embedding=embedding,
            prompt_template=custom_template,
        )

        kwargs = config.get_pipeline_kwargs()

        assert kwargs["llm"] is llm
        assert kwargs["embedding"] is embedding
        assert kwargs["prompt_template"] == custom_template

    def test_config_default_prompt_template(self):
        """Test that config uses default prompt template."""
        llm = FakeListLLM(responses=["hypothetical document"])
        embedding = FakeEmbeddings(size=768)

        config = HyDEPipelineConfig(
            name="test_hyde",
            llm=llm,
            embedding=embedding,
        )

        assert config.prompt_template == DEFAULT_HYDE_PROMPT_TEMPLATE

    @pytest.mark.api
    def test_config_string_llm_conversion(self):
        """Test that string LLM config is converted to instance via __setattr__."""
        embedding = FakeEmbeddings(size=768)

        # This test requires the mock config to exist
        with patch("autorag_research.injection.load_llm") as mock_load:
            mock_llm = MagicMock()
            mock_load.return_value = mock_llm

            config = HyDEPipelineConfig(
                name="test_hyde",
                llm="mock",  # String that triggers conversion
                embedding=embedding,
            )

            mock_load.assert_called_once_with("mock")
            assert config.llm is mock_llm

    @pytest.mark.api
    def test_config_string_embedding_conversion(self):
        """Test that string embedding config is converted to instance via __setattr__."""
        llm = FakeListLLM(responses=["hypothetical document"])

        with patch("autorag_research.injection.load_embedding_model") as mock_load:
            mock_embedding = MagicMock()
            mock_load.return_value = mock_embedding

            config = HyDEPipelineConfig(
                name="test_hyde",
                llm=llm,
                embedding="mock",  # String that triggers conversion
            )

            mock_load.assert_called_once_with("mock")
            assert config.embedding is mock_embedding


class TestHyDERetrievalPipeline:
    """Tests for HyDERetrievalPipeline."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns predictable hypothetical documents."""
        return FakeListLLM(responses=["This is a hypothetical document about machine learning."])

    @pytest.fixture
    def mock_embedding(self):
        """Create a mock embedding model that returns consistent embeddings."""
        return FakeEmbeddings(size=768)

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory: sessionmaker[Session]):
        """Cleanup fixture that deletes pipeline results after test."""
        created_pipeline_ids: list[int] = []

        yield created_pipeline_ids

        session = session_factory()
        try:
            result_repo = ChunkRetrievedResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                result_repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()


    def test_pipeline_creation_custom_template(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_embedding,
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline accepts custom prompt template."""
        custom_template = "Write a Wikipedia passage about: {question}"

        pipeline = HyDERetrievalPipeline(
            session_factory=session_factory,
            name="test_hyde_custom_template",
            llm=mock_llm,
            embedding=mock_embedding,
            prompt_template=custom_template,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline.prompt_template == custom_template

    def test_pipeline_config(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_embedding,
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline config is correct."""
        custom_template = "Custom: {question}"

        pipeline = HyDERetrievalPipeline(
            session_factory=session_factory,
            name="test_hyde_config",
            llm=mock_llm,
            embedding=mock_embedding,
            prompt_template=custom_template,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["type"] == "hyde"
        assert config["prompt_template"] == custom_template


    def test_generate_hypothetical_document_uses_template(
        self,
        session_factory: sessionmaker[Session],
        mock_embedding,
        cleanup_pipeline_results: list[int],
    ):
        """Test that hypothetical document generation uses prompt template."""
        llm = MagicMock()
        llm.invoke.return_value = "Hypothetical response"
        custom_template = "Custom prompt for: {question}"

        pipeline = HyDERetrievalPipeline(
            session_factory=session_factory,
            name="test_hyde_template_usage",
            llm=llm,
            embedding=mock_embedding,
            prompt_template=custom_template,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        pipeline._generate_hypothetical_document("test query")

        # Verify the template was used
        llm.invoke.assert_called_once_with("Custom prompt for: test query")

    def test_run_full_pipeline(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_embedding,
        cleanup_pipeline_results: list[int],
    ):
        """Test running the full pipeline with mocked components."""
        from autorag_research.orm.repository.query import QueryRepository

        # Count actual queries in database
        session = session_factory()
        try:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()
        finally:
            session.close()

        # Mock service results - return results for each query
        mock_result = [
            {"doc_id": 1, "score": 0.9, "content": "Content 1"},
            {"doc_id": 2, "score": 0.8, "content": "Content 2"},
        ]

        with patch(
            "autorag_research.orm.service.retrieval_pipeline.RetrievalPipelineService.vector_search_by_embedding"
        ) as mock_search:
            mock_search.return_value = mock_result

            pipeline = HyDERetrievalPipeline(
                session_factory=session_factory,
                name="test_hyde_full_run",
                llm=mock_llm,
                embedding=mock_embedding,
            )
            cleanup_pipeline_results.append(pipeline.pipeline_id)

            result = pipeline.run(top_k=3)

            # Verify using test utilities
            config = PipelineTestConfig(
                pipeline_type="retrieval",
                expected_total_queries=query_count,
                expected_min_results=0,
                check_persistence=True,
            )
            verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
            verifier.verify_all()

    def test_retrieve_single_query(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_embedding,
        cleanup_pipeline_results: list[int],
    ):
        """Test single query retrieval via retrieve() method."""
        mock_result = [
            {"doc_id": 1, "score": 0.95, "content": "Relevant content"},
        ]

        with patch(
            "autorag_research.orm.service.retrieval_pipeline.RetrievalPipelineService.vector_search_by_embedding"
        ) as mock_search:
            mock_search.return_value = mock_result

            pipeline = HyDERetrievalPipeline(
                session_factory=session_factory,
                name="test_hyde_single_retrieve",
                llm=mock_llm,
                embedding=mock_embedding,
            )
            cleanup_pipeline_results.append(pipeline.pipeline_id)

            results = pipeline.retrieve("What is deep learning?", top_k=5)

            assert len(results) == 1
            assert results[0]["doc_id"] == 1
            assert results[0]["score"] == 0.95
