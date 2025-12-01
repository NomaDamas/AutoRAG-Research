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


def _try_convert_to_int(value: str) -> int | None:
    """Try to convert a string to integer, return None if not possible."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _reassign_string_ids_to_integers(  # noqa: C901
    corpus: dict[str, dict],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
) -> tuple[dict[str, dict], dict[str, str], dict[str, dict[str, int]], dict[str, int], dict[str, int]]:
    """Reassign string IDs to unique integers when they cannot be converted directly.

    This function handles BEIR datasets where IDs are non-numeric strings (e.g., "doc_a", "q_1").
    It creates a mapping from original string IDs to new unique integer IDs.

    The integer ID assignment starts from 1 and increments for each unique string ID.
    Query IDs and corpus IDs are assigned from separate counters to avoid potential
    conflicts and maintain clarity.

    Args:
        corpus: Dictionary mapping doc_id (str) to document dict with "title" and "text".
        queries: Dictionary mapping query_id (str) to query string.
        qrels: Dictionary mapping query_id (str) to dict of doc_id (str) -> relevance (int).

    Returns:
        Tuple of:
        - Updated corpus with integer-convertible string keys
        - Updated queries with integer-convertible string keys
        - Updated qrels with integer-convertible string keys
        - Query ID mapping (original_str_id -> new_int_id)
        - Corpus ID mapping (original_str_id -> new_int_id)
    """
    # Check if we need to reassign IDs
    needs_query_reassignment = any(_try_convert_to_int(qid) is None for qid in queries)
    needs_corpus_reassignment = any(_try_convert_to_int(cid) is None for cid in corpus)

    # Also check for ID uniqueness when converted to int (e.g., "1" and "01" would both become 1)
    if not needs_query_reassignment:
        converted_qids = [_try_convert_to_int(qid) for qid in queries]
        if len(converted_qids) != len(set(converted_qids)):
            needs_query_reassignment = True

    if not needs_corpus_reassignment:
        converted_cids = [_try_convert_to_int(cid) for cid in corpus]
        if len(converted_cids) != len(set(converted_cids)):
            needs_corpus_reassignment = True

    query_id_map: dict[str, int] = {}
    corpus_id_map: dict[str, int] = {}

    # Reassign query IDs if needed
    if needs_query_reassignment:
        logger.info("Reassigning query string IDs to unique integers...")
        for idx, qid in enumerate(queries.keys(), start=1):
            query_id_map[qid] = idx
        # Create new queries dict with string keys that can be converted to int
        new_queries = {str(query_id_map[qid]): query for qid, query in queries.items()}
    else:
        # Keep original IDs, create identity mapping for consistency
        for qid in queries:
            query_id_map[qid] = int(qid)
        new_queries = queries

    # Reassign corpus IDs if needed
    if needs_corpus_reassignment:
        logger.info("Reassigning corpus string IDs to unique integers...")
        for idx, cid in enumerate(corpus.keys(), start=1):
            corpus_id_map[cid] = idx
        # Create new corpus dict with string keys that can be converted to int
        new_corpus = {str(corpus_id_map[cid]): doc for cid, doc in corpus.items()}
    else:
        # Keep original IDs, create identity mapping for consistency
        for cid in corpus:
            corpus_id_map[cid] = int(cid)
        new_corpus = corpus

    # Update qrels with new IDs
    if needs_query_reassignment or needs_corpus_reassignment:
        new_qrels: dict[str, dict[str, int]] = {}
        for old_qid, doc_dict in qrels.items():
            new_qid = str(query_id_map[old_qid])
            new_qrels[new_qid] = {}
            for old_cid, relevance in doc_dict.items():
                # Only include corpus IDs that exist in the corpus
                if old_cid in corpus_id_map:
                    new_cid = str(corpus_id_map[old_cid])
                    new_qrels[new_qid][new_cid] = relevance
                else:
                    logger.warning(f"Corpus ID '{old_cid}' in qrels not found in corpus, skipping.")
    else:
        new_qrels = qrels

    return new_corpus, new_queries, new_qrels, query_id_map, corpus_id_map


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

        # Handle string IDs that cannot be converted to integers
        # This reassigns non-numeric string IDs to unique integers
        corpus, queries, qrels, _, _ = _reassign_string_ids_to_integers(corpus, queries, qrels)

        # Ingest Queries
        logger.info(f"Ingesting {len(queries)} queries from BEIR dataset '{self.dataset_name}'...")
        qids = [int(x) for x in list(queries.keys())]
        query_contents = list(queries.values())
        self.service.add_queries_simple(query_contents, qids)

        # Ingest Corpus (concat title and text which is conventional in BeIR and Pyserini)
        logger.info(f"Ingesting {len(corpus)} corpus documents from BEIR dataset '{self.dataset_name}'...")
        corpus_ids = [int(x) for x in list(corpus.keys())]
        corpus_contents = [(doc.get("title", "") + " " + doc["text"]).strip() for doc in corpus.values()]
        self.service.add_chunks_simple(corpus_contents, corpus_ids)

        # Ingest qrels
        from autorag_research.orm.models import and_all, or_all

        for qid, doc_dict in qrels.items():
            int_qid = int(qid)
            gt_ids = self.filter_valid_retrieval_gt_ids(doc_dict)
            gt_ids = [int(x) for x in gt_ids]
            if not gt_ids:
                continue
            if self.dataset_name == "hotpotqa":
                # Multi-hop: each document is a separate hop in the chain
                self.service.add_retrieval_gt(int_qid, and_all(gt_ids), chunk_type="text")
            else:
                # Single-hop: all documents are OR alternatives (any is correct)
                self.service.add_retrieval_gt(int_qid, or_all(gt_ids), chunk_type="text")

        self.service.clean()

    @staticmethod
    def filter_valid_retrieval_gt_ids(dictionary: dict[str, int]) -> list[str]:
        # From a given dict, return only keys that the value is more than zero
        return [k for k, v in dictionary.items() if v > 0]

    def embed_all(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
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
