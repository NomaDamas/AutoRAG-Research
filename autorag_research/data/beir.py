import logging
import os
import random
from typing import Literal, get_args

from beir.datasets.data_loader import GenericDataLoader
from beir.util import download_and_unzip
from llama_index.core.base.embeddings.base import BaseEmbedding

from autorag_research.data import USER_DATA_DIR
from autorag_research.data.base import TextEmbeddingDataIngestor
from autorag_research.data.registry import register_ingestor
from autorag_research.exceptions import ServiceNotSetError, UnsupportedDataSubsetError

logger = logging.getLogger("AutoRAG-Research")

RANDOM_SEED = 42

# BEIR available datasets
BEIR_DATASETS = Literal[
    # "msmarco",
    "trec-covid",
    "nfcorpus",
    "nq",
    "hotpotqa",
    "fiqa",
    "arguana",
    "webis-touche2020",
    "cqadupstack",
    "quora",
    "dbpedia-entity",
    "scidocs",
    "fever",
    "climate-fever",
    "scifact",
]


@register_ingestor(
    name="beir",
    description="BEIR benchmark datasets for information retrieval",
)
class BEIRIngestor(TextEmbeddingDataIngestor):
    def __init__(self, embedding_model: BaseEmbedding, dataset_name: BEIR_DATASETS):
        super().__init__(embedding_model)
        if dataset_name not in get_args(BEIR_DATASETS):
            raise UnsupportedDataSubsetError
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

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
    ) -> None:
        corpus, queries, qrels = GenericDataLoader(
            data_folder=self.data_path,
        ).load(split=subset)

        if self.service is None:
            raise ServiceNotSetError

        rng = random.Random(RANDOM_SEED)

        # Step 1: Sample queries and collect gold IDs (only when min_corpus_cnt is set)
        qids, filtered_qrels, gold_corpus_ids = self._sample_queries(
            queries, qrels, query_limit, rng, collect_gold_ids=min_corpus_cnt is not None
        )

        # Step 2: Filter corpus
        corpus_ids = self._filter_corpus(corpus, gold_corpus_ids, min_corpus_cnt, rng)
        corpus_ids_set = set(corpus_ids)

        # Step 3: Ingest data
        self._ingest_queries(queries, qids)
        self._ingest_corpus(corpus, corpus_ids)
        self._ingest_qrels(filtered_qrels, corpus_ids_set)

        self.service.clean()

    def _sample_queries(
        self,
        queries: dict,
        qrels: dict,
        query_limit: int | None,
        rng: random.Random,
        collect_gold_ids: bool = False,
    ) -> tuple[list, dict[str, dict[str, int]], set[str]]:
        """Sample queries and collect gold corpus IDs.

        Args:
            collect_gold_ids: If True, collect gold corpus IDs from qrels.
                Only needed when min_corpus_cnt is set.
        """
        qids = list(queries.keys())
        if query_limit is not None and query_limit < len(qids):
            qids = rng.sample(qids, query_limit)
            logger.info(f"Sampled {len(qids)} queries from {len(queries)} total")

        gold_corpus_ids: set[str] = set()
        filtered_qrels: dict[str, dict[str, int]] = {}
        for qid in qids:
            if qid not in qrels:
                continue

            filtered_qrels[qid] = qrels[qid]

            if collect_gold_ids:
                gold_corpus_ids.update(doc_id for doc_id, score in qrels[qid].items() if score > 0)

        return qids, filtered_qrels, gold_corpus_ids

    def _filter_corpus(
        self,
        corpus: dict,
        gold_corpus_ids: set[str],
        min_corpus_cnt: int | None,
        rng: random.Random,
    ) -> list:
        """Filter corpus to include gold IDs + random samples up to limit."""
        corpus_ids = list(corpus.keys())
        if min_corpus_cnt is None:
            return corpus_ids

        # Always include gold IDs
        selected_corpus_ids = list(gold_corpus_ids & set(corpus_ids))
        remaining_corpus_ids = [cid for cid in corpus_ids if cid not in gold_corpus_ids]

        # Add random samples if we need more
        additional_needed = min_corpus_cnt - len(selected_corpus_ids)
        if additional_needed > 0 and remaining_corpus_ids:
            additional_ids = rng.sample(
                remaining_corpus_ids,
                min(additional_needed, len(remaining_corpus_ids)),
            )
            selected_corpus_ids.extend(additional_ids)

        logger.info(
            f"Corpus subset: {len(gold_corpus_ids)} gold IDs + "
            f"{len(selected_corpus_ids) - len(gold_corpus_ids)} random = {len(selected_corpus_ids)} total"
        )
        return selected_corpus_ids

    def _ingest_queries(self, queries: dict, qids: list) -> None:
        """Ingest selected queries."""
        if self.service is None:
            raise ServiceNotSetError
        logger.info(f"Ingesting {len(qids)} queries from BEIR dataset '{self.dataset_name}'...")
        self.service.add_queries([{"id": qid, "contents": queries[qid]} for qid in qids])

    def _ingest_corpus(self, corpus: dict, corpus_ids: list) -> None:
        """Ingest selected corpus documents."""
        if self.service is None:
            raise ServiceNotSetError
        logger.info(f"Ingesting {len(corpus_ids)} corpus documents from BEIR dataset '{self.dataset_name}'...")
        self.service.add_chunks([
            {
                "id": cid,
                "contents": (corpus[cid].get("title", "") + " " + corpus[cid]["text"]).strip(),
            }
            for cid in corpus_ids
        ])

    def _ingest_qrels(self, filtered_qrels: dict[str, dict[str, int]], corpus_ids_set: set) -> None:
        """Ingest qrels for selected queries and existing corpus."""
        if self.service is None:
            raise ServiceNotSetError

        from autorag_research.orm.models import and_all, or_all

        for qid, doc_dict in filtered_qrels.items():
            gt_ids = [gid for gid in self.filter_valid_retrieval_gt_ids(doc_dict) if gid in corpus_ids_set]
            if not gt_ids:
                continue
            if self.dataset_name == "hotpotqa":
                self.service.add_retrieval_gt(qid, and_all(gt_ids), chunk_type="text")
            else:
                self.service.add_retrieval_gt(qid, or_all(gt_ids), chunk_type="text")

    @staticmethod
    def filter_valid_retrieval_gt_ids(dictionary: dict[str, int]) -> list[str | int]:
        # From a given dict, return only keys that the value is more than zero
        return [k for k, v in dictionary.items() if v > 0]
