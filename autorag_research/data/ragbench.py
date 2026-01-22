import hashlib
import logging
from typing import Any, Literal

from datasets import load_dataset
from llama_index.core.base.embeddings.base import BaseEmbedding

from autorag_research.data.base import TextEmbeddingDataIngestor
from autorag_research.exceptions import ServiceNotSetError
from autorag_research.orm.models import or_all
from autorag_research.util import normalize_string

logger = logging.getLogger("AutoRAG-Research")

RAGBENCH_CONFIGS = [
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


class RAGBenchIngestor(TextEmbeddingDataIngestor):
    def __init__(
        self,
        embedding_model: BaseEmbedding,
        configs: list[str] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        super().__init__(embedding_model)
        self.configs = configs if configs is not None else RAGBENCH_CONFIGS
        self.batch_size = batch_size
        self._validate_configs()

    def _validate_configs(self) -> None:
        for config in self.configs:
            if config not in RAGBENCH_CONFIGS:
                raise ValueError(f"Invalid config '{config}'. Valid configs: {RAGBENCH_CONFIGS}")  # noqa: TRY003

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        return "string"

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        corpus_limit: int | None = None,
    ) -> None:
        if self.service is None:
            raise ServiceNotSetError

        if corpus_limit is not None:
            logger.warning(
                "corpus_limit is ineffective for RAGBench. "
                "Each query has its own document set (1:N relation) without a shared corpus. "
                "Only query_limit is effective for this dataset."
            )

        hf_split = SPLIT_MAPPING[subset]

        for config in self.configs:
            logger.info(f"Ingesting RAGBench config '{config}' split '{hf_split}'...")
            self._ingest_config(config, hf_split, query_limit)
            logger.info(f"Completed ingestion for config '{config}'")

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
        batch: list[dict[str, Any]] = []
        total_processed = 0

        for example in ds:
            example_dict: dict[str, Any] = example  # type: ignore[assignment]
            batch.append(example_dict)

            if len(batch) >= self.batch_size:
                self._process_batch(config, split, batch, seen_chunk_ids)
                total_processed += len(batch)
                logger.info(f"[{config}] Processed {total_processed} examples...")
                batch = []

            if query_limit is not None and total_processed + len(batch) >= query_limit:
                remaining = query_limit - total_processed
                batch = batch[:remaining]
                break

        if batch:
            self._process_batch(config, split, batch, seen_chunk_ids)
            total_processed += len(batch)

        logger.info(f"[{config}] Total examples processed: {total_processed}")
        logger.info(f"[{config}] Unique chunks: {len(seen_chunk_ids)}")

    def _process_batch(
        self,
        config: str,
        split: str,
        examples: list[dict[str, Any]],
        seen_chunk_ids: set[str],
    ) -> None:
        doc_to_chunk_mapping = self._ingest_chunks(config, examples, seen_chunk_ids)
        self._ingest_queries(config, split, examples)
        self._ingest_relations(config, split, examples, doc_to_chunk_mapping)

    def _ingest_chunks(
        self,
        config: str,
        examples: list[dict[str, Any]],
        seen_chunk_ids: set[str],
    ) -> dict[tuple[str, int], str]:
        if self.service is None:
            raise ServiceNotSetError

        chunks_to_add: list[dict[str, str | int | None]] = []
        doc_to_chunk_mapping: dict[tuple[str, int], str] = {}

        for example in examples:
            example_id = str(example["id"])
            documents = example.get("documents", [])

            for doc_idx, doc_text in enumerate(documents):
                if not doc_text or not doc_text.strip():
                    logger.warning(f"Empty document at index {doc_idx} for example {example_id}")
                    continue

                chunk_id = compute_chunk_id(doc_text, config)
                doc_to_chunk_mapping[(example_id, doc_idx)] = chunk_id

                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    chunks_to_add.append({
                        "id": chunk_id,
                        "contents": doc_text,
                    })

        if chunks_to_add:
            self.service.add_chunks(chunks_to_add)

        return doc_to_chunk_mapping

    def _ingest_queries(
        self,
        config: str,
        split: str,
        examples: list[dict[str, Any]],
    ) -> None:
        if self.service is None:
            raise ServiceNotSetError

        queries: list[dict[str, str | list[str] | None]] = []

        for example in examples:
            example_id = str(example["id"])
            query_id = _make_query_id(config, split, example_id)
            response = example.get("response")

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
        doc_to_chunk_mapping: dict[tuple[str, int], str],
    ) -> None:
        if self.service is None:
            raise ServiceNotSetError

        for example in examples:
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
                doc_to_chunk_mapping[(example_id, idx)]
                for idx in sorted(valid_indices)
                if (example_id, idx) in doc_to_chunk_mapping
            ]

            if chunk_ids:
                self.service.add_retrieval_gt(query_id, or_all(chunk_ids), chunk_type="text")

    def embed_all(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        if self.service is None:
            raise ServiceNotSetError

        logger.info("Embedding all queries...")
        self.service.embed_all_queries(
            self.embedding_model.aget_query_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )

        logger.info("Embedding all chunks...")
        self.service.embed_all_chunks(
            self.embedding_model.aget_text_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )

        logger.info("RAGBench embedding complete.")
