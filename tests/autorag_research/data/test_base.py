"""Tests for base DataIngestor classes.

Unit tests for TextEmbeddingDataIngestor and MultiModalEmbeddingDataIngestor base classes.
"""

from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core import MockEmbedding
from llama_index.core.embeddings import MultiModalEmbedding

from autorag_research.data.base import MultiModalEmbeddingDataIngestor, TextEmbeddingDataIngestor
from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding
from autorag_research.exceptions import EmbeddingError, ServiceNotSetError

EMBEDDING_DIM = 768


# ==================== Concrete Test Implementations ====================


class ConcreteTextIngestor(TextEmbeddingDataIngestor):
    """Concrete implementation for testing TextEmbeddingDataIngestor."""

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
    ) -> None:
        pass

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        return "string"


class ConcreteMultiModalIngestor(MultiModalEmbeddingDataIngestor):
    """Concrete implementation for testing MultiModalEmbeddingDataIngestor."""

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
    ) -> None:
        pass

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        return "bigint"


# ==================== Fixtures ====================


@pytest.fixture
def mock_embedding_model():
    return MockEmbedding(EMBEDDING_DIM)


@pytest.fixture
def mock_multi_modal_embedding():
    """Create a mock MultiModalEmbedding model."""
    mock = MagicMock(spec=MultiModalEmbedding)
    mock.aget_query_embedding = AsyncMock(return_value=[0.1] * EMBEDDING_DIM)
    mock.aget_image_embedding = AsyncMock(return_value=[0.2] * EMBEDDING_DIM)
    return mock


@pytest.fixture
def mock_late_interaction_embedding():
    """Create a mock MultiVectorMultiModalEmbedding model."""
    mock = MagicMock(spec=MultiVectorMultiModalEmbedding)
    mock.aget_query_embedding = AsyncMock(return_value=[[0.1] * EMBEDDING_DIM])
    mock.aget_image_embedding = AsyncMock(return_value=[[0.2] * EMBEDDING_DIM])
    return mock


@pytest.fixture
def mock_text_service():
    """Create a mock TextDataIngestionService."""
    service = MagicMock()
    service.embed_all_queries = MagicMock()
    service.embed_all_chunks = MagicMock()
    return service


@pytest.fixture
def mock_multi_modal_service():
    """Create a mock MultiModalIngestionService."""
    service = MagicMock()
    service.embed_all_queries = MagicMock()
    service.embed_all_image_chunks = MagicMock()
    service.embed_all_queries_multi_vector = MagicMock()
    service.embed_all_image_chunks_multi_vector = MagicMock()
    return service


# ==================== TextEmbeddingDataIngestor Tests ====================


class TestTextEmbeddingDataIngestorEmbedAll:
    """Tests for TextEmbeddingDataIngestor.embed_all()."""

    def test_embed_all_raises_service_not_set_error(self, mock_embedding_model):
        """embed_all should raise ServiceNotSetError when service is not set."""
        ingestor = ConcreteTextIngestor(mock_embedding_model)

        with pytest.raises(ServiceNotSetError):
            ingestor.embed_all()

    def test_embed_all_calls_service_methods(self, mock_embedding_model, mock_text_service):
        """embed_all should call embed_all_queries and embed_all_chunks on service."""
        ingestor = ConcreteTextIngestor(mock_embedding_model)
        ingestor.set_service(mock_text_service)

        ingestor.embed_all(max_concurrency=8, batch_size=64)

        mock_text_service.embed_all_queries.assert_called_once_with(
            mock_embedding_model.aget_query_embedding,
            batch_size=64,
            max_concurrency=8,
        )
        mock_text_service.embed_all_chunks.assert_called_once_with(
            mock_embedding_model.aget_text_embedding,
            batch_size=64,
            max_concurrency=8,
        )

    def test_embed_all_uses_default_parameters(self, mock_embedding_model, mock_text_service):
        """embed_all should use default parameters when not specified."""
        ingestor = ConcreteTextIngestor(mock_embedding_model)
        ingestor.set_service(mock_text_service)

        ingestor.embed_all()

        mock_text_service.embed_all_queries.assert_called_once_with(
            mock_embedding_model.aget_query_embedding,
            batch_size=128,
            max_concurrency=16,
        )
        mock_text_service.embed_all_chunks.assert_called_once_with(
            mock_embedding_model.aget_text_embedding,
            batch_size=128,
            max_concurrency=16,
        )


# ==================== MultiModalEmbeddingDataIngestor Tests ====================


class TestMultiModalEmbeddingDataIngestorEmbedAll:
    """Tests for MultiModalEmbeddingDataIngestor.embed_all()."""

    def test_embed_all_raises_embedding_not_set_error(self):
        """embed_all should raise EmbeddingNotSetError when embedding_model is not set."""
        ingestor = ConcreteMultiModalIngestor(embedding_model=None)

        with pytest.raises(EmbeddingError):
            ingestor.embed_all()

    def test_embed_all_raises_service_not_set_error(self, mock_multi_modal_embedding):
        """embed_all should raise ServiceNotSetError when service is not set."""
        ingestor = ConcreteMultiModalIngestor(embedding_model=mock_multi_modal_embedding)

        with pytest.raises(ServiceNotSetError):
            ingestor.embed_all()

    def test_embed_all_calls_service_methods(self, mock_multi_modal_embedding, mock_multi_modal_service):
        """embed_all should call embed_all_queries and embed_all_image_chunks on service."""
        ingestor = ConcreteMultiModalIngestor(embedding_model=mock_multi_modal_embedding)
        ingestor.set_service(mock_multi_modal_service)

        ingestor.embed_all(max_concurrency=8, batch_size=64)

        mock_multi_modal_service.embed_all_queries.assert_called_once_with(
            mock_multi_modal_embedding.aget_query_embedding,
            batch_size=64,
            max_concurrency=8,
        )
        mock_multi_modal_service.embed_all_image_chunks.assert_called_once_with(
            mock_multi_modal_embedding.aget_image_embedding,
            batch_size=64,
            max_concurrency=8,
        )

    def test_embed_all_uses_default_parameters(self, mock_multi_modal_embedding, mock_multi_modal_service):
        """embed_all should use default parameters when not specified."""
        ingestor = ConcreteMultiModalIngestor(embedding_model=mock_multi_modal_embedding)
        ingestor.set_service(mock_multi_modal_service)

        ingestor.embed_all()

        mock_multi_modal_service.embed_all_queries.assert_called_once_with(
            mock_multi_modal_embedding.aget_query_embedding,
            batch_size=128,
            max_concurrency=16,
        )
        mock_multi_modal_service.embed_all_image_chunks.assert_called_once_with(
            mock_multi_modal_embedding.aget_image_embedding,
            batch_size=128,
            max_concurrency=16,
        )


class TestMultiModalEmbeddingDataIngestorEmbedAllLateInteraction:
    """Tests for MultiModalEmbeddingDataIngestor.embed_all_late_interaction()."""

    def test_embed_all_late_interaction_raises_embedding_not_set_error(self):
        """embed_all_late_interaction should raise EmbeddingNotSetError when model not set."""
        ingestor = ConcreteMultiModalIngestor(late_interaction_embedding_model=None)

        with pytest.raises(EmbeddingError):
            ingestor.embed_all_late_interaction()

    def test_embed_all_late_interaction_raises_service_not_set_error(self, mock_late_interaction_embedding):
        """embed_all_late_interaction should raise ServiceNotSetError when service not set."""
        ingestor = ConcreteMultiModalIngestor(late_interaction_embedding_model=mock_late_interaction_embedding)

        with pytest.raises(ServiceNotSetError):
            ingestor.embed_all_late_interaction()

    def test_embed_all_late_interaction_calls_service_methods(
        self, mock_late_interaction_embedding, mock_multi_modal_service
    ):
        """embed_all_late_interaction should call multi_vector methods on service."""
        ingestor = ConcreteMultiModalIngestor(late_interaction_embedding_model=mock_late_interaction_embedding)
        ingestor.set_service(mock_multi_modal_service)

        ingestor.embed_all_late_interaction(max_concurrency=8, batch_size=64)

        mock_multi_modal_service.embed_all_queries_multi_vector.assert_called_once_with(
            mock_late_interaction_embedding.aget_query_embedding,
            batch_size=64,
            max_concurrency=8,
        )
        mock_multi_modal_service.embed_all_image_chunks_multi_vector.assert_called_once_with(
            mock_late_interaction_embedding.aget_image_embedding,
            batch_size=64,
            max_concurrency=8,
        )

    def test_embed_all_late_interaction_uses_default_parameters(
        self, mock_late_interaction_embedding, mock_multi_modal_service
    ):
        """embed_all_late_interaction should use default parameters when not specified."""
        ingestor = ConcreteMultiModalIngestor(late_interaction_embedding_model=mock_late_interaction_embedding)
        ingestor.set_service(mock_multi_modal_service)

        ingestor.embed_all_late_interaction()

        mock_multi_modal_service.embed_all_queries_multi_vector.assert_called_once_with(
            mock_late_interaction_embedding.aget_query_embedding,
            batch_size=128,
            max_concurrency=16,
        )
        mock_multi_modal_service.embed_all_image_chunks_multi_vector.assert_called_once_with(
            mock_late_interaction_embedding.aget_image_embedding,
            batch_size=128,
            max_concurrency=16,
        )
