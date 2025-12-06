from abc import ABC, abstractmethod
from typing import Literal

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings import MultiModalEmbedding

from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding
from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService
from autorag_research.orm.service.text_ingestion import TextDataIngestionService


class DataIngestor(ABC):
    def __init__(self):
        self.dataset = None

    @abstractmethod
    def ingest(self, subset: Literal["train", "dev", "test"] = "test"):
        """Ingest data from the specified source. This process does not include an embedding process."""
        pass

    @abstractmethod
    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        """Detect the primary key type used in the dataset."""
        pass


class TextEmbeddingDataIngestor(DataIngestor, ABC):
    def __init__(self, embedding_model: BaseEmbedding):
        super().__init__()
        self.service = None
        self.embedding_model = embedding_model

    @abstractmethod
    def embed_all(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        pass

    def set_service(self, service: TextDataIngestionService) -> None:
        self.service = service


class MultiModalEmbeddingDataIngestor(DataIngestor, ABC):
    def __init__(
        self,
        embedding_model: MultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
    ):
        super().__init__()
        self.service = None
        self.embedding_model = embedding_model
        self.late_interaction_embedding_model = late_interaction_embedding_model

    def set_service(self, service: MultiModalIngestionService) -> None:
        self.service = service

    @abstractmethod
    def embed_all(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        pass

    @abstractmethod
    def embed_all_late_interaction(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        pass
