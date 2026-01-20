import base64
import json
import logging
import random
import re
from typing import Any, Literal

from huggingface_hub import hf_hub_download
from llama_index.core.embeddings import MultiModalEmbedding

from autorag_research.data.base import MultiModalEmbeddingDataIngestor
from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding
from autorag_research.exceptions import EmbeddingNotSetError, ServiceNotSetError

logger = logging.getLogger("AutoRAG-Research")

RANDOM_SEED = 42
REPO_ID = "vectara/open_ragbench"
DEFAULT_SUBSET = "pdf/arxiv"


def make_chunk_id(doc_id: str, section_id: int) -> str:
    return f"{doc_id}_section_{section_id}"


def make_image_chunk_id(doc_id: str, section_id: int, img_key: str) -> str:
    return f"{doc_id}_section_{section_id}_img_{img_key}"


def extract_image_from_data_uri(data_uri: str) -> tuple[bytes, str]:
    match = re.match(r"data:([^;]+);base64,(.+)", data_uri)
    if not match:
        msg = f"Invalid data URI format: {data_uri[:50]}..."
        raise ValueError(msg)
    mimetype = match.group(1)
    base64_data = match.group(2)
    image_bytes = base64.b64decode(base64_data)
    return image_bytes, mimetype


class OpenRAGBenchIngestor(MultiModalEmbeddingDataIngestor):
    def __init__(
        self,
        embedding_model: MultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
        subset: str = DEFAULT_SUBSET,
    ):
        super().__init__(embedding_model, late_interaction_embedding_model)
        self.subset = subset

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        return "string"

    def _download_json(self, filename: str) -> dict[str, Any]:
        path = hf_hub_download(repo_id=REPO_ID, filename=f"{self.subset}/{filename}", repo_type="dataset")
        with open(path) as f:
            return json.load(f)

    def _load_corpus_document(self, doc_id: str) -> dict[str, Any]:
        return self._download_json(f"corpus/{doc_id}.json")

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        corpus_limit: int | None = None,
    ) -> None:
        if self.service is None:
            raise ServiceNotSetError

        rng = random.Random(RANDOM_SEED)  # noqa: S311

        queries_data = self._download_json("queries.json")
        answers_data = self._download_json("answers.json")
        qrels_data = self._download_json("qrels.json")

        query_ids = list(queries_data.keys())
        if query_limit is not None and query_limit < len(query_ids):
            query_ids = rng.sample(query_ids, query_limit)

        required_doc_ids: set[str] = set()
        for qid in query_ids:
            if qid in qrels_data:
                required_doc_ids.add(qrels_data[qid]["doc_id"])

        chunk_id_map = self._ingest_documents(required_doc_ids)
        self._ingest_queries_and_relations(query_ids, queries_data, answers_data, qrels_data, chunk_id_map)

        logger.info(f"OpenRAGBench ingestion complete: {len(query_ids)} queries, {len(required_doc_ids)} documents")

    def _ingest_documents(self, doc_ids: set[str]) -> dict[str, list[str]]:
        if self.service is None:
            raise ServiceNotSetError

        chunk_id_map: dict[str, list[str]] = {}

        for doc_id in doc_ids:
            try:
                doc = self._load_corpus_document(doc_id)
            except Exception:
                logger.exception(f"Failed to load document {doc_id}")
                continue

            doc_metadata = {
                "abstract": doc.get("abstract"),
                "categories": doc.get("categories", []),
                "published": doc.get("published"),
                "updated": doc.get("updated"),
            }

            authors = doc.get("authors", [])
            author_str = ", ".join(authors) if authors else None

            self.service.add_documents([
                {
                    "id": doc_id,
                    "title": doc.get("title"),
                    "author": author_str,
                    "doc_metadata": doc_metadata,
                }
            ])

            for section in doc.get("sections", []):
                section_id = section["section_id"]
                section_text = section.get("text", "")
                images = section.get("images", {})

                if not section_text and not images:
                    continue

                chunk_id = make_chunk_id(doc_id, section_id)

                if section_text:
                    self.service.add_chunks([
                        {
                            "id": chunk_id,
                            "contents": section_text,
                        }
                    ])

                image_chunk_ids: list[str] = []
                for img_key, img_data_uri in images.items():
                    try:
                        img_bytes, mimetype = extract_image_from_data_uri(img_data_uri)
                        img_chunk_id = make_image_chunk_id(doc_id, section_id, img_key)
                        self.service.add_image_chunks([
                            {
                                "id": img_chunk_id,
                                "contents": img_bytes,
                                "mimetype": mimetype,
                            }
                        ])
                        image_chunk_ids.append(img_chunk_id)
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_key} from section {section_id}: {e}")
                        continue

                chunk_id_map[chunk_id] = image_chunk_ids

        return chunk_id_map

    def _ingest_queries_and_relations(
        self,
        query_ids: list[str],
        queries_data: dict[str, Any],
        answers_data: dict[str, str],
        qrels_data: dict[str, dict[str, Any]],
        chunk_id_map: dict[str, list[str]],
    ) -> None:
        if self.service is None:
            raise ServiceNotSetError

        self.service.add_queries([
            {
                "id": qid,
                "contents": queries_data[qid]["query"],
                "generation_gt": [answers_data[qid]] if qid in answers_data else None,
            }
            for qid in query_ids
        ])

        retrieval_gt_items: list[tuple[str, str]] = []
        for qid in query_ids:
            if qid not in qrels_data:
                continue

            qrel = qrels_data[qid]
            doc_id = qrel["doc_id"]
            section_id = qrel["section_id"]
            chunk_id = make_chunk_id(doc_id, section_id)

            if chunk_id not in chunk_id_map:
                continue

            retrieval_gt_items.append((qid, chunk_id))

        if retrieval_gt_items:
            self.service.add_retrieval_gt_batch(retrieval_gt_items, chunk_type="text")  # ty: ignore[invalid-argument-type]

    def embed_all(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        if self.embedding_model is None:
            raise EmbeddingNotSetError
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
        self.service.embed_all_image_chunks(
            self.embedding_model.aget_image_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )

    def embed_all_late_interaction(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        if self.late_interaction_embedding_model is None:
            raise EmbeddingNotSetError
        if self.service is None:
            raise ServiceNotSetError

        self.service.embed_all_queries_multi_vector(
            self.late_interaction_embedding_model.aget_query_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
        self.service.embed_all_chunks_multi_vector(
            self.late_interaction_embedding_model.aget_text_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
        self.service.embed_all_image_chunks_multi_vector(
            self.late_interaction_embedding_model.aget_image_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
