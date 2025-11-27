from abc import ABC, abstractmethod
from typing import Literal

from llama_index.core.base.embeddings.base import BaseEmbedding


class DataIngestor(ABC):
    def __init__(self):
        self.dataset = None

    @abstractmethod
    def ingest(self, ingest_cnt: int, random_state: int = 42, subset: Literal["train", "val", "test"] = "test"):
        """Ingest data from the specified source."""
        pass


class TextEmbeddingDataIngestor(DataIngestor, ABC):
    def __init__(self, embedding_model: BaseEmbedding):
        super().__init__()
        self.embedding_model = embedding_model

    @abstractmethod
    async def embed_queries(self):
        pass

    @abstractmethod
    async def embed_chunks(self):
        pass

    @abstractmethod
    def embed_all(self, concurrent_limit: int = 16) -> None:
        pass
