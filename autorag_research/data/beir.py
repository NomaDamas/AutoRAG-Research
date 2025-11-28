import logging
import os
from typing import Literal

from beir.datasets.data_loader import GenericDataLoader
from beir.util import download_and_unzip
from llama_index.core.base.embeddings.base import BaseEmbedding

from autorag_research.data import USER_DATA_DIR
from autorag_research.data.base import TextEmbeddingDataIngestor
from autorag_research.orm.service.text_ingestion import TextDataIngestionService

logger = logging.getLogger("AutoRAG-Research")


class BEIRIngestor(TextEmbeddingDataIngestor):
    def __init__(
        self, text_data_ingestion_service: TextDataIngestionService, embedding_model: BaseEmbedding, dataset_name: str
    ):
        super().__init__(text_data_ingestion_service, embedding_model)
        self.dataset_name = dataset_name
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
        out_dir = os.path.join(USER_DATA_DIR, "beir", self.dataset_name)
        self.data_path = download_and_unzip(url, out_dir)

    def ingest(self, subset: Literal["train", "dev", "test"] = "test"):
        corpus, queries, qrels = GenericDataLoader(
            data_folder=self.data_path,
        ).load(split=subset)

        # Ingest Queries
        logger.info(f"Ingesting {len(queries)} queries from BEIR dataset '{self.dataset_name}'...")
        qids = [int(x) for x in list(queries.keys())]
        query_contents = list(queries.values())
        self.service.add_queries_simple(query_contents, qids)

        # Ingest Corpus
        logger.info(f"Ingesting {len(corpus)} corpus documents from BEIR dataset '{self.dataset_name}'...")
        # Concat title and text which is conventional in BeIR and Pyserini
        corpus_ids = [int(x) for x in list(corpus.keys())]
        corpus_contents = [(doc.get("title", "") + " " + doc["text"]).strip() for doc in corpus.values()]
        self.service.add_chunks_simple(corpus_contents, corpus_ids)
        for qid, doc_dict in qrels.items():
            gt_ids = self.filter_valid_retrieval_gt_ids(doc_dict)
            gt_ids = [int(x) for x in gt_ids]
            if self.dataset_name == "hotpotqa":
                self.service.add_retrieval_gt_multihop(int(qid), [[gt_id] for gt_id in gt_ids])
            else:
                self.service.add_retrieval_gt_simple(int(qid), gt_ids)

    @staticmethod
    def filter_valid_retrieval_gt_ids(dictionary: dict[str, int]) -> list[str]:
        # From a given dict, return only keys that the value is more than zero
        return [k for k, v in dictionary.items() if v > 0]

    async def embed_queries(self):
        pass

    async def embed_chunks(self):
        pass

    def embed_all(self, concurrent_limit: int = 16) -> None:
        pass
