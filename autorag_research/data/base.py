from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings import MultiModalEmbedding

from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding
from autorag_research.exceptions import EmbeddingError, ServiceNotSetError
from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService
from autorag_research.orm.service.text_ingestion import TextDataIngestionService


@dataclass
class QueryMetadata:
    """Lightweight query metadata for subset sampling."""

    query_id: str
    query_text: str
    gold_ids: list[str]
    gold_answer: str | None


class DataIngestor(ABC):
    def __init__(self):
        self.dataset = None

    @abstractmethod
    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
    ) -> None:
        """Ingest data from the specified source. This process does not include an embedding process.

        Args:
            subset: Dataset split to ingest (train, dev, or test).
            query_limit: Maximum number of queries to ingest. None means no limit.
            min_corpus_cnt: Maximum number of corpus items to ingest.
                          When set, gold IDs from selected queries are always included,
                          plus random samples to reach the limit. None means no limit.
        """
        pass

    @abstractmethod
    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        """Detect the primary key type used in the dataset."""
        pass


class TextEmbeddingDataIngestor(DataIngestor, ABC):
    def __init__(self, embedding_model: BaseEmbedding, **kwargs):
        super().__init__()
        self.service: TextDataIngestionService | None = None
        self.embedding_model = embedding_model

    def embed_all(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        """Embed all queries and text chunks."""
        if self.service is None:
            raise ServiceNotSetError
        if self.embedding_model is None:
            raise EmbeddingError
        self.service.embed_all_queries(
            self.embedding_model.aget_query_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
        self.service.embed_all_chunks(
            self.embedding_model.aget_text_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )

    def set_service(self, service: TextDataIngestionService) -> None:
        self.service = service


class MultiModalEmbeddingDataIngestor(DataIngestor, ABC):
    def __init__(
        self,
        embedding_model: MultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
        **kwargs,
    ):
        super().__init__()
        self.service: MultiModalIngestionService | None = None
        self.embedding_model = embedding_model
        self.late_interaction_embedding_model = late_interaction_embedding_model

    def set_service(self, service: MultiModalIngestionService) -> None:
        self.service = service

    def embed_all(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        """Embed all queries and image chunks using single-vector embedding model."""
        if self.embedding_model is None:
            raise EmbeddingError
        if self.service is None:
            raise ServiceNotSetError
        self.service.embed_all_queries(
            self.embedding_model.aget_query_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
        self.service.embed_all_image_chunks(
            self.embedding_model.aget_image_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )

    def embed_all_late_interaction(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        """Embed all queries and image chunks using multi-vector embedding model."""
        if self.late_interaction_embedding_model is None:
            raise EmbeddingError
        if self.service is None:
            raise ServiceNotSetError
        self.service.embed_all_queries_multi_vector(
            self.late_interaction_embedding_model.aget_query_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
        self.service.embed_all_image_chunks_multi_vector(
            self.late_interaction_embedding_model.aget_image_embedding,  # ty: ignore[invalid-argument-type]
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
