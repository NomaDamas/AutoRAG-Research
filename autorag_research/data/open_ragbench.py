import json
import logging
import random
from typing import Any, Literal

from huggingface_hub import hf_hub_download
from llama_index.core.embeddings import MultiModalEmbedding

from autorag_research.data.base import MultiModalEmbeddingDataIngestor
from autorag_research.data.registry import register_ingestor
from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding
from autorag_research.exceptions import ServiceNotSetError
from autorag_research.orm.models import ImageId, TextId
from autorag_research.util import extract_image_from_data_uri

logger = logging.getLogger("AutoRAG-Research")

RANDOM_SEED = 42
REPO_ID = "vectara/open_ragbench"
DATA_PATH = "pdf/arxiv"


def make_id(*parts: str | int) -> str:
    """Generate ID by joining parts with underscore."""
    return "_".join(str(p) for p in parts)


@register_ingestor(
    name="open-ragbench",
    description="The Open RAG Benchmark is a unique, high-quality Retrieval-Augmented Generation (RAG) dataset constructed directly from arXiv PDF documents, specifically designed for evaluating RAG systems with a focus on multimodal PDF understanding, made by Vectara.",
    hf_repo="open-ragbench-dumps",
)
class OpenRAGBenchIngestor(MultiModalEmbeddingDataIngestor):
    def __init__(
        self,
        embedding_model: MultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
        data_path: str = DATA_PATH,
    ):
        super().__init__(embedding_model, late_interaction_embedding_model)
        self.data_path = data_path

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        return "string"

    def _download_json(self, filename: str) -> dict[str, Any]:
        path = hf_hub_download(repo_id=REPO_ID, filename=f"{self.data_path}/{filename}", repo_type="dataset")
        with open(path) as f:
            return json.load(f)

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
                "min_corpus_cnt is ineffective for OpenRAGBench. "
                "Each query maps to a specific document section (1:1 relation). "
                "Only query_limit is effective for this dataset."
            )

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

    def _ingest_documents(self, doc_ids: set[str]) -> dict[str, dict[str, list[str]]]:  # noqa: C901
        """Ingest documents with proper hierarchy: File -> Document -> Page -> Chunk."""
        if self.service is None:
            raise ServiceNotSetError

        chunk_id_map: dict[str, dict[str, list[str]]] = {}
        pdf_urls_data = self._download_json("pdf_urls.json")

        for doc_id in doc_ids:
            try:
                doc = self._download_json(f"corpus/{doc_id}.json")
            except Exception:
                logger.exception(f"Failed to load document {doc_id}")
                continue

            pdf_url = pdf_urls_data.get(doc_id)

            # 1. Create File record
            file_id = None
            if pdf_url:
                file_id = make_id(doc_id, "file")
                self.service.add_files([{"id": file_id, "type": "raw", "path": pdf_url}])

            # 2. Create Document
            authors = doc.get("authors", [])
            self.service.add_documents([
                {
                    "id": doc_id,
                    "path": file_id,
                    "title": doc.get("title"),
                    "author": ", ".join(authors) if authors else None,
                    "doc_metadata": {
                        "abstract": doc.get("abstract"),
                        "categories": doc.get("categories", []),
                        "published": doc.get("published"),
                        "updated": doc.get("updated"),
                        "pdf_url": pdf_url,
                    },
                }
            ])

            for section in doc.get("sections", []):
                section_id = section["section_id"]
                section_text = section.get("text", "")
                images = section.get("images", {})
                tables = section.get("tables", {})

                if not section_text and not images and not tables:
                    continue

                page_id = make_id(doc_id, "page", section_id)
                chunk_id = make_id(doc_id, "section", section_id)

                # 3. Create Page
                self.service.add_pages([{"id": page_id, "document_id": doc_id, "page_num": section_id}])

                # 4. Create ImageChunks
                image_chunk_ids: list[str] = []
                for img_key, img_data_uri in images.items():
                    try:
                        img_bytes, mimetype = extract_image_from_data_uri(img_data_uri)
                        img_chunk_id = make_id(doc_id, "section", section_id, "img", img_key)
                        self.service.add_image_chunks([
                            {"id": img_chunk_id, "contents": img_bytes, "mimetype": mimetype, "parent_page": page_id}
                        ])
                        image_chunk_ids.append(img_chunk_id)
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_key} from section {section_id}: {e}")

                # 5. Create Text Chunk
                chunk_ids_to_link: list[str] = []
                if section_text:
                    self.service.add_chunks([{"id": chunk_id, "contents": section_text, "is_table": False}])
                    chunk_ids_to_link.append(chunk_id)

                # 6. Create Table Chunks
                table_chunk_ids: list[str] = []
                for table_key, table_data in tables.items():
                    table_chunk_id = make_id(doc_id, "section", section_id, "table", table_key)
                    self.service.add_chunks([
                        {
                            "id": table_chunk_id,
                            "contents": str(table_data),
                            "is_table": True,
                            "table_type": "markdown",
                        }
                    ])
                    chunk_ids_to_link.append(table_chunk_id)
                    table_chunk_ids.append(table_chunk_id)

                # 7. Link all chunks to page at once (1:N relationship)
                if chunk_ids_to_link:
                    self.service.link_page_to_chunks(page_id, chunk_ids_to_link)

                chunk_id_map[chunk_id] = {"images": image_chunk_ids, "tables": table_chunk_ids}

        return chunk_id_map

    def _ingest_queries_and_relations(
        self,
        query_ids: list[str],
        queries_data: dict[str, Any],
        answers_data: dict[str, str],
        qrels_data: dict[str, dict[str, Any]],
        chunk_id_map: dict[str, dict[str, list[str]]],
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

        for qid in query_ids:
            if qid not in qrels_data:
                continue

            qrel = qrels_data[qid]
            doc_id = qrel["doc_id"]
            section_id = qrel["section_id"]
            chunk_id = make_id(doc_id, "section", section_id)

            if chunk_id not in chunk_id_map:
                continue

            related_assets = chunk_id_map.get(chunk_id, {"images": [], "tables": []})

            # Build OR expression for all GT chunks (text + table + image)
            gt_expr: Any = TextId(chunk_id)

            for table_chunk_id in related_assets["tables"]:
                gt_expr = gt_expr | TextId(table_chunk_id)

            for image_chunk_id in related_assets["images"]:
                gt_expr = gt_expr | ImageId(image_chunk_id)

            # Register all GT for this query at once (mixed mode)
            self.service.add_retrieval_gt(qid, gt_expr)

    def embed_all(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        super().embed_all(max_concurrency=max_concurrency, batch_size=batch_size)
        if self.embedding_model is None or self.service is None:
            return
        self.service.embed_all_chunks(
            self.embedding_model.aget_text_embedding, batch_size=batch_size, max_concurrency=max_concurrency
        )

    def embed_all_late_interaction(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        super().embed_all_late_interaction(max_concurrency=max_concurrency, batch_size=batch_size)
        if self.late_interaction_embedding_model is None or self.service is None:
            return
        self.service.embed_all_chunks_multi_vector(
            self.late_interaction_embedding_model.aget_text_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
