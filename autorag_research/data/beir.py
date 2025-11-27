from typing import Literal

from datasets import load_dataset
from llama_index.core.base.embeddings.base import BaseEmbedding

from autorag_research.data.base import TextEmbeddingDataIngestor
from autorag_research.exceptions import UnsupportedDataSubsetError


class BEIRIngestor(TextEmbeddingDataIngestor):
    def __init__(self, embedding_model: BaseEmbedding, dataset_name: str):
        super().__init__(embedding_model)
        self.dataset_name = dataset_name
        self.dataset_repo_path = f"BeIR/{dataset_name}"
        self.qrels_repo_path = f"BeIR/{dataset_name}-qrels"
        self.dataset = load_dataset(self.dataset_repo_path)
        self.qrels = load_dataset(self.qrels_repo_path)

    def ingest(self, ingest_cnt: int, random_state: int = 42, subset: Literal["train", "val", "test"] = "train"):
        if subset != "train":
            raise UnsupportedDataSubsetError(["val", "test"])

    async def embed_queries(self):
        pass

    async def embed_chunks(self):
        pass
