import hashlib
import logging
from typing import Any, Literal, get_args

from datasets import load_dataset
from langchain_core.embeddings import Embeddings

from autorag_research.data.base import TextEmbeddingDataIngestor
from autorag_research.exceptions import ServiceNotSetError, UnsupportedDataSubsetError
from autorag_research.orm.models import or_all
from autorag_research.util import normalize_string

logger = logging.getLogger("AutoRAG-Research")

# RAGBench available configs
RAGBENCH_CONFIGS_LITERAL = Literal[
    "covidqa",
    "cuad",
    "delucionqa",
    "emanual",
    "expertqa",
    "finqa",
    "hagrid",
    "hotpotqa",
    "msmarco",
    "pubmedqa",
    "tatqa",
    "techqa",
]

SPLIT_MAPPING = {
    "train": "train",
    "dev": "validation",
    "test": "test",
}

DEFAULT_BATCH_SIZE = 1000


def extract_relevant_doc_indices(sentence_keys: list[str]) -> set[int]:
    doc_indices: set[int] = set()
    for key in sentence_keys:
        doc_idx = ""
        for char in key:
            if char.isdigit():
                doc_idx += char
            else:
                break
        if doc_idx:
            doc_indices.add(int(doc_idx))
    return doc_indices


def compute_chunk_id(content: str, config: str) -> str:
    normalized = normalize_string(content)
    content_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    return f"{config}_{content_hash}"


def _make_query_id(config: str, split: str, example_id: str) -> str:
    return f"{config}_{split}_{example_id}"


# Import here to avoid circular import at module level
from autorag_research.data.registry import register_ingestor  # noqa: E402


@register_ingestor(
    name="ragbench",
    description="RAGBench benchmark for RAG evaluation",
    hf_repo="ragbench-dumps",
)
class RAGBenchIngestor(TextEmbeddingDataIngestor):
    def __init__(
        self,
        embedding_model: Embeddings,
        config: RAGBENCH_CONFIGS_LITERAL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        super().__init__(embedding_model)
        if config not in get_args(RAGBENCH_CONFIGS_LITERAL):
            raise UnsupportedDataSubsetError
        self.config = config
        self.batch_size = batch_size

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        return "string"

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
    ) -> None:
        if self.service is None:
            raise ServiceNotSetError

        if min_corpus_cnt is not None:
            logger.warning(
                "min_corpus_cnt is ineffective for RAGBench. "
                "Each query has its own document set (1:N relation) without a shared corpus. "
                "Only query_limit is effective for this dataset."
            )

        hf_split = SPLIT_MAPPING[subset]

        self._ingest_config(self.config, hf_split, query_limit)

        self.service.clean()
        logger.info("RAGBench ingestion complete.")

    def _ingest_config(
        self,
        config: str,
        split: str,
        query_limit: int | None,
    ) -> None:
        if self.service is None:
            raise ServiceNotSetError

        ds = load_dataset(
            "galileo-ai/ragbench",
            config,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )

        seen_chunk_ids: set[str] = set()
        relation_chunk_ids_by_query: dict[str, set[str]] = {}
        query_metadata_by_query: dict[str, tuple[str, str | None]] = {}
        batch: list[dict[str, Any]] = []
        total_processed = 0

        for example in ds:
            example_dict: dict[str, Any] = example
            batch.append(example_dict)

            if len(batch) >= self.batch_size:
                self._process_batch(
                    config,
                    split,
                    batch,
                    seen_chunk_ids,
                    relation_chunk_ids_by_query,
                    query_metadata_by_query,
                )
                total_processed += len(batch)
                logger.info(f"[{config}] Processed {total_processed} examples...")
                batch = []

            if query_limit is not None and total_processed + len(batch) >= query_limit:
                remaining = query_limit - total_processed
                batch = batch[:remaining]
                break

        if batch:
            self._process_batch(
                config,
                split,
                batch,
                seen_chunk_ids,
                relation_chunk_ids_by_query,
                query_metadata_by_query,
            )
            total_processed += len(batch)

        logger.info(f"[{config}] Total examples processed: {total_processed}")
        logger.info(f"[{config}] Unique chunks: {len(seen_chunk_ids)}")

    def _process_batch(
        self,
        config: str,
        split: str,
        examples: list[dict[str, Any]],
        seen_chunk_ids: set[str],
        relation_chunk_ids_by_query: dict[str, set[str]],
        query_metadata_by_query: dict[str, tuple[str, str | None]] | None = None,
    ) -> None:
        if query_metadata_by_query is not None:
            self._record_query_metadata_and_warn_on_mismatch(config, split, examples, query_metadata_by_query)
        row_doc_to_chunk_mapping = self._ingest_chunks(config, examples, seen_chunk_ids)
        self._ingest_queries(config, split, examples)
        self._ingest_relations(config, split, examples, row_doc_to_chunk_mapping, relation_chunk_ids_by_query)

    def _record_query_metadata_and_warn_on_mismatch(
        self,
        config: str,
        split: str,
        examples: list[dict[str, Any]],
        query_metadata_by_query: dict[str, tuple[str, str | None]],
    ) -> None:
        for example in examples:
            example_id = str(example["id"])
            query_id = _make_query_id(config, split, example_id)
            question = str(example["question"])
            response = example.get("response")
            response_text = str(response) if response is not None else None
            current_metadata = (question, response_text)
            first_metadata = query_metadata_by_query.setdefault(query_id, current_metadata)

            if first_metadata != current_metadata:
                logger.warning(
                    "Duplicate RAGBench query id %s has different question/response metadata; "
                    "keeping first query metadata while unioning retrieval ground truth. "
                    "first_question=%r duplicate_question=%r first_response=%r duplicate_response=%r",
                    query_id,
                    first_metadata[0],
                    question,
                    first_metadata[1],
                    response_text,
                )

    def _ingest_chunks(
        self,
        config: str,
        examples: list[dict[str, Any]],
        seen_chunk_ids: set[str],
    ) -> dict[tuple[int, int], str]:
        if self.service is None:
            raise ServiceNotSetError

        chunks_to_add: list[dict[str, str | int | None]] = []
        row_doc_to_chunk_mapping: dict[tuple[int, int], str] = {}

        for row_idx, example in enumerate(examples):
            example_id = str(example["id"])
            documents = example.get("documents", [])

            for doc_idx, doc_text in enumerate(documents):
                if not doc_text or not doc_text.strip():
                    logger.warning(f"Empty document at index {doc_idx} for example {example_id}")
                    continue

                chunk_id = compute_chunk_id(doc_text, config)
                row_doc_to_chunk_mapping[(row_idx, doc_idx)] = chunk_id

                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    chunks_to_add.append({
                        "id": chunk_id,
                        "contents": doc_text,
                    })

        if chunks_to_add:
            self.service.add_chunks(chunks_to_add)

        return row_doc_to_chunk_mapping

    def _ingest_queries(
        self,
        config: str,
        split: str,
        examples: list[dict[str, Any]],
    ) -> None:
        if self.service is None:
            raise ServiceNotSetError

        queries: list[dict[str, str | list[str] | None]] = []
        seen_query_ids: set[str] = set()

        for example in examples:
            example_id = str(example["id"])
            query_id = _make_query_id(config, split, example_id)
            if query_id in seen_query_ids:
                continue
            seen_query_ids.add(query_id)

            response = example.get("response")

            # RAGBench duplicate IDs are expected to repeat the same query/response metadata. Query insertion remains
            # first-wins via the ingestion service's default duplicate-skip behavior, while retrieval GT is unioned.
            queries.append({
                "id": query_id,
                "contents": example["question"],
                "generation_gt": [response] if response else None,
            })

        if queries:
            self.service.add_queries(queries)

    def _ingest_relations(
        self,
        config: str,
        split: str,
        examples: list[dict[str, Any]],
        row_doc_to_chunk_mapping: dict[tuple[int, int], str],
        relation_chunk_ids_by_query: dict[str, set[str]] | None = None,
    ) -> None:
        if self.service is None:
            raise ServiceNotSetError

        for row_idx, example in enumerate(examples):
            example_id = str(example["id"])
            query_id = _make_query_id(config, split, example_id)

            sentence_keys = example.get("all_relevant_sentence_keys", [])
            if not sentence_keys:
                logger.warning(f"No relevant sentences for query {example_id}, skipping relations")
                continue

            relevant_doc_indices = extract_relevant_doc_indices(sentence_keys)
            if not relevant_doc_indices:
                continue

            documents = example.get("documents", [])
            valid_indices = [idx for idx in relevant_doc_indices if idx < len(documents)]
            if len(valid_indices) < len(relevant_doc_indices):
                logger.warning(f"Some sentence keys reference non-existent documents for example {example_id}")

            chunk_ids = [
                row_doc_to_chunk_mapping[(row_idx, idx)]
                for idx in sorted(valid_indices)
                if (row_idx, idx) in row_doc_to_chunk_mapping
            ]

            if chunk_ids:
                stored_chunk_ids = (
                    relation_chunk_ids_by_query.setdefault(query_id, set())
                    if relation_chunk_ids_by_query is not None
                    else set()
                )
                stored_chunk_ids.update(chunk_ids)
                gt_chunk_ids = sorted(stored_chunk_ids) if relation_chunk_ids_by_query is not None else chunk_ids
                self.service.add_retrieval_gt(query_id, or_all(gt_chunk_ids), chunk_type="text", upsert=True)
