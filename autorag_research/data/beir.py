import logging
import os
from typing import Literal

from beir.datasets.data_loader import GenericDataLoader
from beir.util import download_and_unzip
from llama_index.core.base.embeddings.base import BaseEmbedding

from autorag_research.data import USER_DATA_DIR
from autorag_research.data.base import TextEmbeddingDataIngestor
from autorag_research.exceptions import ServiceNotSetError

logger = logging.getLogger("AutoRAG-Research")


class BEIRIngestor(TextEmbeddingDataIngestor):
    def __init__(self, embedding_model: BaseEmbedding, dataset_name: str):
        super().__init__(embedding_model)
        self.dataset_name = dataset_name
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
        out_dir = os.path.join(USER_DATA_DIR, "beir", self.dataset_name)
        self.data_path = download_and_unzip(url, out_dir)

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        corpus, queries, _ = GenericDataLoader(
            data_folder=self.data_path,
        ).load(split="test")
        all_ids = list(corpus.keys()) + list(queries.keys())
        if any(isinstance(id_value, str) for id_value in all_ids):
            return "string"
        return "bigint"

    def ingest(self, subset: Literal["train", "dev", "test"] = "test"):
        corpus, queries, qrels = GenericDataLoader(
            data_folder=self.data_path,
        ).load(split=subset)

        if self.service is None:
            raise ServiceNotSetError

        # Ingest Queries
        logger.info(f"Ingesting {len(queries)} queries from BEIR dataset '{self.dataset_name}'...")
        qids = list(queries.keys())
        query_contents = list(queries.values())
        self.service.add_queries([
            {
                "id": qid,
                "contents": query,
            }
            for qid, query in zip(qids, query_contents, strict=True)
        ])

        # Ingest Corpus (concat title and text which is conventional in BeIR and Pyserini)
        logger.info(f"Ingesting {len(corpus)} corpus documents from BEIR dataset '{self.dataset_name}'...")
        corpus_ids = list(corpus.keys())
        corpus_contents = [(doc.get("title", "") + " " + doc["text"]).strip() for doc in corpus.values()]
        self.service.add_chunks([
            {
                "id": cid,
                "contents": content,
            }
            for cid, content in zip(corpus_ids, corpus_contents, strict=True)
        ])

        # Ingest qrels
        from autorag_research.orm.models import and_all, or_all

        for qid, doc_dict in qrels.items():
            gt_ids = self.filter_valid_retrieval_gt_ids(doc_dict)
            if not gt_ids:
                continue
            if self.dataset_name == "hotpotqa":
                # Multi-hop: each document is a separate hop in the chain
                self.service.add_retrieval_gt(qid, and_all(gt_ids), chunk_type="text")
            else:
                # Single-hop: all documents are OR alternatives (any is correct)
                self.service.add_retrieval_gt(qid, or_all(gt_ids), chunk_type="text")

        self.service.clean()

    @staticmethod
    def filter_valid_retrieval_gt_ids(dictionary: dict[str, int]) -> list[str | int]:
        # From a given dict, return only keys that the value is more than zero
        return [k for k, v in dictionary.items() if v > 0]

    def embed_all(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        if self.service is None:
            raise ServiceNotSetError
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
