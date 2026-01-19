import logging
import random
from typing import Any, Literal

from datasets import load_dataset
from llama_index.core.base.embeddings.base import BaseEmbedding

from autorag_research.data.base import TextEmbeddingDataIngestor, QueryMetadata
from autorag_research.exceptions import ServiceNotSetError
from autorag_research.orm.models import or_all

logger = logging.getLogger("AutoRAG-Research")

RANDOM_SEED = 42

BRIGHT_DOMAINS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "pony",
    "leetcode",
    "aops",
    "theoremqa_theorems",
    "theoremqa_questions",
]

DOMAINS_WITH_LONG_DOCS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "pony",
]

DOMAINS_WITHOUT_LONG_DOCS = ["leetcode", "aops", "theoremqa_theorems", "theoremqa_questions"]

DocumentMode = Literal["short", "long"]

BATCH_SIZE = 1000


class BRIGHTIngestor(TextEmbeddingDataIngestor):
    def __init__(
        self,
        embedding_model: BaseEmbedding,
        domains: list[str] | None = None,
        document_mode: DocumentMode = "short",
    ):
        super().__init__(embedding_model)
        self.domains = domains if domains is not None else BRIGHT_DOMAINS
        self.document_mode = document_mode
        self._validate_domains()

    def _validate_domains(self) -> None:
        for domain in self.domains:
            if domain not in BRIGHT_DOMAINS:
                raise ValueError(f"Invalid domain '{domain}'. Valid domains: {BRIGHT_DOMAINS}")  # noqa: TRY003
            if self.document_mode == "long" and domain in DOMAINS_WITHOUT_LONG_DOCS:
                raise ValueError(  # noqa: TRY003
                    f"Domain '{domain}' does not support long documents. "
                    f"Available domains for long_documents: {DOMAINS_WITH_LONG_DOCS}"
                )

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

        for domain in self.domains:
            logger.info(f"Ingesting domain '{domain}' with document_mode='{self.document_mode}'...")
            self._ingest_domain(domain, query_limit, corpus_limit)
            logger.info(f"Completed ingestion for domain '{domain}'")

        self.service.clean()
        logger.info("BRIGHT ingestion complete.")

    def _ingest_domain(
        self,
        domain: str,
        query_limit: int | None,
        corpus_limit: int | None,
    ) -> None:
        """Ingest a single domain with optional query/corpus limits.

        Flow (per-domain limits):
        1. First pass: collect query metadata (streaming) - lightweight
        2. Sample queries if query_limit is set
        3. Collect gold_ids from selected queries
        4. Ingest corpus with filtering (streaming) - gold IDs + reservoir sampling
        5. Ingest queries
        6. Ingest relations
        """
        if self.service is None:
            raise ServiceNotSetError

        rng = random.Random(RANDOM_SEED)  # noqa: S311

        # Step 1: Collect query metadata (streaming - just metadata, not embeddings)
        query_metadata_list = self._collect_query_metadata(domain)
        logger.info(f"[{domain}] Collected {len(query_metadata_list)} query metadata entries")

        # Step 2: Sample queries if limit is set
        if query_limit is not None and query_limit < len(query_metadata_list):
            query_metadata_list = rng.sample(query_metadata_list, query_limit)
            logger.info(f"[{domain}] Sampled {len(query_metadata_list)} queries")

        # Step 3: Collect all gold IDs from selected queries
        gold_ids_set: set[str] = set()
        for qm in query_metadata_list:
            gold_ids_set.update(_make_id(domain, gid) for gid in qm.gold_ids)
        logger.info(f"[{domain}] Total gold IDs: {len(gold_ids_set)}")

        # Step 4: Ingest corpus with filtering
        self._ingest_corpus_filtered(domain, gold_ids_set, corpus_limit, rng)

        # Step 5: Ingest queries
        self.service.add_queries([
            {
                "id": qm.query_id,
                "contents": qm.query_text,
                "generation_gt": [qm.gold_answer] if qm.gold_answer else None,
            }
            for qm in query_metadata_list
        ])
        logger.info(f"[{domain}] Ingested {len(query_metadata_list)} queries")

        # Step 6: Ingest relations
        for qm in query_metadata_list:
            if qm.gold_ids:
                chunk_ids = [_make_id(domain, gid) for gid in qm.gold_ids]
                self.service.add_retrieval_gt(qm.query_id, or_all(chunk_ids), chunk_type="text")

    def _collect_query_metadata(self, domain: str) -> list[QueryMetadata]:
        """Stream through examples dataset and collect lightweight query metadata."""
        ds = load_dataset("xlangai/BRIGHT", "examples", split=domain, streaming=True, trust_remote_code=True)
        metadata_list: list[QueryMetadata] = []

        for example in ds:
            example_dict: dict[str, Any] = example  # type: ignore[assignment]
            gold_ids = _get_gold_ids(example_dict, self.document_mode, domain)

            if not gold_ids:
                continue

            query_id = _make_id(domain, example_dict["id"])
            gold_answer = example_dict["gold_answer"]
            processed_answer = None if gold_answer == "N/A" else gold_answer

            metadata_list.append(
                QueryMetadata(
                    query_id=query_id,
                    query_text=example_dict["query"],
                    gold_ids=gold_ids,
                    gold_answer=processed_answer,
                )
            )

        return metadata_list

    def _ingest_corpus_filtered(
        self,
        domain: str,
        gold_ids_set: set[str],
        corpus_limit: int | None,
        rng: random.Random,
    ) -> int:
        """Ingest corpus with filtering: gold IDs always included, plus reservoir sampling for additional items."""
        if self.service is None:
            raise ServiceNotSetError

        config = "long_documents" if self.document_mode == "long" else "documents"
        ds = load_dataset("xlangai/BRIGHT", config, split=domain, streaming=True, trust_remote_code=True)

        # If no corpus_limit, ingest all (original behavior)
        if corpus_limit is None:
            return self._ingest_corpus_all(ds, domain)

        # With corpus_limit: gold IDs + reservoir sampling for additional
        gold_chunks: list[dict[str, str | int | None]] = []
        reservoir: list[dict[str, str | int | None]] = []
        reservoir_capacity = max(0, corpus_limit - len(gold_ids_set))
        stream_count = 0

        for doc in ds:
            doc_dict: dict[str, Any] = doc  # type: ignore[assignment]
            chunk_id = _make_id(domain, doc_dict["id"])
            chunk_data: dict[str, str | int | None] = {
                "id": chunk_id,
                "contents": doc_dict["content"],
            }

            if chunk_id in gold_ids_set:
                gold_chunks.append(chunk_data)
            elif reservoir_capacity > 0:
                # Reservoir sampling for non-gold items
                if len(reservoir) < reservoir_capacity:
                    reservoir.append(chunk_data)
                else:
                    # Replace with decreasing probability
                    j = rng.randint(0, stream_count)
                    if j < reservoir_capacity:
                        reservoir[j] = chunk_data

            stream_count += 1

        # Combine gold chunks + reservoir samples
        all_chunks = gold_chunks + reservoir
        total_count = len(all_chunks)

        # Batch insert
        for i in range(0, total_count, BATCH_SIZE):
            batch = all_chunks[i : i + BATCH_SIZE]
            self.service.add_chunks(batch)

        logger.info(
            f"[{domain}] Corpus subset: {len(gold_chunks)} gold + "
            f"{len(reservoir)} random = {total_count} total (from {stream_count} streamed)"
        )
        return total_count

    def _ingest_corpus_all(self, ds: Any, domain: str) -> int:
        """Ingest all corpus items (no filtering)."""
        if self.service is None:
            raise ServiceNotSetError

        chunks: list[dict[str, str | int | None]] = []
        total_count = 0

        for doc in ds:
            doc_dict: dict[str, Any] = doc  # type: ignore[assignment]
            chunk_id = _make_id(domain, doc_dict["id"])
            chunks.append({
                "id": chunk_id,
                "contents": doc_dict["content"],
            })
            total_count += 1

            if len(chunks) >= BATCH_SIZE:
                self.service.add_chunks(chunks)
                logger.info(f"[{domain}] Ingested {total_count} chunks so far...")
                chunks = []

        if chunks:
            self.service.add_chunks(chunks)

        logger.info(f"[{domain}] Total chunks ingested: {total_count}")
        return total_count

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

        logger.info("BRIGHT embedding complete.")


def _make_id(domain: str, source_id: str) -> str:
    return f"{domain}_{source_id}"


def _get_gold_ids(example: dict[str, Any], document_mode: DocumentMode, domain: str) -> list[str]:
    if document_mode == "long":
        if domain in DOMAINS_WITHOUT_LONG_DOCS:
            raise ValueError(  # noqa: TRY003
                f"Domain '{domain}' does not have long documents. Use document_mode='short' instead."
            )
        gold_ids = example["gold_ids_long"]
        return [gid for gid in gold_ids if gid != "N/A"]
    return example["gold_ids"]
