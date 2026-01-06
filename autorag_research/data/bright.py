import logging
from typing import Any, Literal

from datasets import load_dataset
from llama_index.core.base.embeddings.base import BaseEmbedding

from autorag_research.data.base import TextEmbeddingDataIngestor
from autorag_research.exceptions import ServiceNotSetError
from autorag_research.orm.models import or_all

logger = logging.getLogger("AutoRAG-Research")

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

    def ingest(self, subset: Literal["train", "dev", "test"] = "test") -> None:
        if self.service is None:
            raise ServiceNotSetError

        for domain in self.domains:
            logger.info(f"Ingesting domain '{domain}' with document_mode='{self.document_mode}'...")
            self._ingest_corpus(domain)
            query_ids, gold_ids_list = self._ingest_queries(domain)
            self._ingest_relations(query_ids, gold_ids_list)
            logger.info(f"Completed ingestion for domain '{domain}'")

        self.service.clean()
        logger.info("BRIGHT ingestion complete.")

    def _ingest_corpus(self, domain: str) -> int:
        if self.service is None:
            raise ServiceNotSetError

        config = "long_documents" if self.document_mode == "long" else "documents"
        ds = load_dataset("xlangai/BRIGHT", config, split=domain, streaming=True, trust_remote_code=True)

        chunks: list[dict[str, str | int | None]] = []
        total_count = 0

        for doc in ds:
            doc_dict: dict[str, Any] = doc  # type: ignore[assignment]
            chunk_id = _make_chunk_id(domain, doc_dict["id"])
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

    def _ingest_queries(self, domain: str) -> tuple[list[str], list[list[str]]]:
        if self.service is None:
            raise ServiceNotSetError

        ds = load_dataset("xlangai/BRIGHT", "examples", split=domain, streaming=True, trust_remote_code=True)

        queries: list[dict[str, str | list[str] | None]] = []
        query_ids: list[str] = []
        gold_ids_list: list[list[str]] = []

        for example in ds:
            example_dict: dict[str, Any] = example  # type: ignore[assignment]
            query_id = _make_query_id(domain, example_dict["id"])
            gold_ids = _get_gold_ids(example_dict, self.document_mode, domain)

            if not gold_ids:
                logger.warning(f"Query {query_id} has no valid gold IDs, skipping.")
                continue

            queries.append({
                "id": query_id,
                "contents": example_dict["query"],
                "generation_gt": _process_gold_answer(example_dict["gold_answer"]),
            })
            query_ids.append(query_id)
            gold_ids_list.append([_make_chunk_id(domain, gid) for gid in gold_ids])

        self.service.add_queries(queries)
        logger.info(f"[{domain}] Total queries ingested: {len(queries)}")
        return query_ids, gold_ids_list

    def _ingest_relations(self, query_ids: list[str], gold_ids_list: list[list[str]]) -> None:
        if self.service is None:
            raise ServiceNotSetError

        for query_id, gold_ids in zip(query_ids, gold_ids_list, strict=True):
            if not gold_ids:
                continue
            self.service.add_retrieval_gt(query_id, or_all(gold_ids), chunk_type="text")

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


def _make_query_id(domain: str, source_id: str) -> str:
    return f"{domain}_{source_id}"


def _make_chunk_id(domain: str, source_id: str) -> str:
    return f"{domain}_{source_id}"


def _process_gold_answer(gold_answer: str) -> list[str] | None:
    if gold_answer == "N/A":
        return None
    return [gold_answer]


def _get_gold_ids(example: dict[str, Any], document_mode: DocumentMode, domain: str) -> list[str]:
    if document_mode == "long":
        if domain in DOMAINS_WITHOUT_LONG_DOCS:
            raise ValueError(  # noqa: TRY003
                f"Domain '{domain}' does not have long documents. Use document_mode='short' instead."
            )
        gold_ids = example["gold_ids_long"]
        return [gid for gid in gold_ids if gid != "N/A"]
    return example["gold_ids"]
