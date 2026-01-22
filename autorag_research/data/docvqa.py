import io
import logging
import random
from typing import Any, Literal

from datasets import load_dataset
from llama_index.core.embeddings import MultiModalEmbedding
from PIL import Image

from autorag_research.data.base import MultiModalEmbeddingDataIngestor
from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding
from autorag_research.exceptions import EmbeddingNotSetError, ServiceNotSetError

logger = logging.getLogger("AutoRAG-Research")

RANDOM_SEED = 42
DATASET_NAME = "openbmb/VisRAG-Ret-Test-MP-DocVQA"


class DocVQAIngestor(MultiModalEmbeddingDataIngestor):
    def __init__(
        self,
        embedding_model: MultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
    ):
        super().__init__(embedding_model, late_interaction_embedding_model)

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

        rng = random.Random(RANDOM_SEED)  # noqa: S311

        logger.info("Loading DocVQA dataset configurations...")
        corpus_dataset = load_dataset(DATASET_NAME, "corpus", split="train")
        queries_dataset = load_dataset(DATASET_NAME, "queries", split="train")
        qrels_dataset = load_dataset(DATASET_NAME, "qrels", split="train")

        qrels_map = self._build_qrels_map(qrels_dataset)

        selected_queries, required_corpus_ids = self._select_queries_and_corpus(
            queries_dataset,
            qrels_map,
            corpus_dataset,
            query_limit,
            corpus_limit,
            rng,
        )

        self._ingest_image_chunks(corpus_dataset, required_corpus_ids)
        self._ingest_queries(selected_queries)
        self._ingest_qrels(selected_queries, qrels_map, required_corpus_ids)

    def _build_qrels_map(self, qrels_dataset: Any) -> dict[str, str]:
        return {row["query-id"]: row["corpus-id"] for row in qrels_dataset}

    def _select_queries_and_corpus(
        self,
        queries_dataset: Any,
        qrels_map: dict[str, str],
        corpus_dataset: Any,
        query_limit: int | None,
        corpus_limit: int | None,
        rng: random.Random,
    ) -> tuple[list[dict[str, Any]], set[str]]:
        all_queries = list(queries_dataset)

        if query_limit is not None and query_limit < len(all_queries):
            selected_queries = rng.sample(all_queries, query_limit)
            logger.info(f"Sampled {len(selected_queries)} queries from {len(all_queries)} total")
        else:
            selected_queries = all_queries

        gold_corpus_ids = {qrels_map[q["query-id"]] for q in selected_queries if q["query-id"] in qrels_map}

        if corpus_limit is None:
            required_corpus_ids = {row["corpus-id"] for row in corpus_dataset}
        else:
            all_corpus_ids = [row["corpus-id"] for row in corpus_dataset]
            non_gold_ids = [cid for cid in all_corpus_ids if cid not in gold_corpus_ids]

            additional_needed = corpus_limit - len(gold_corpus_ids)
            if additional_needed > 0:
                sampled_ids = rng.sample(non_gold_ids, min(additional_needed, len(non_gold_ids)))
                required_corpus_ids = gold_corpus_ids | set(sampled_ids)
            else:
                required_corpus_ids = gold_corpus_ids

            logger.info(
                f"Corpus subset: {len(gold_corpus_ids)} gold IDs + "
                f"{len(required_corpus_ids) - len(gold_corpus_ids)} random = {len(required_corpus_ids)} total"
            )

        return selected_queries, required_corpus_ids

    def _ingest_image_chunks(self, corpus_dataset: Any, required_corpus_ids: set[str]) -> list[int | str]:
        if self.service is None:
            raise ServiceNotSetError

        image_chunks = []
        for row in corpus_dataset:
            corpus_id = row["corpus-id"]
            if corpus_id not in required_corpus_ids:
                continue
            image = row["image"]
            image_bytes, mimetype = self._pil_image_to_bytes(image)
            image_chunks.append({
                "id": corpus_id,
                "contents": image_bytes,
                "mimetype": mimetype,
            })

        logger.info(f"Ingesting {len(image_chunks)} image chunks...")
        return self.service.add_image_chunks(image_chunks)

    def _ingest_queries(self, selected_queries: list[dict[str, Any]]) -> list[int | str]:
        if self.service is None:
            raise ServiceNotSetError

        queries = [
            {
                "id": row["query-id"],
                "contents": row["query"],
                "generation_gt": row["answer"],
            }
            for row in selected_queries
        ]

        logger.info(f"Ingesting {len(queries)} queries...")
        return self.service.add_queries(queries)

    def _ingest_qrels(
        self,
        selected_queries: list[dict[str, Any]],
        qrels_map: dict[str, str],
        required_corpus_ids: set[str],
    ) -> None:
        if self.service is None:
            raise ServiceNotSetError

        items = []
        for query in selected_queries:
            query_id = query["query-id"]
            if query_id not in qrels_map:
                continue
            corpus_id = qrels_map[query_id]
            if corpus_id not in required_corpus_ids:
                continue
            items.append((query_id, corpus_id))

        logger.info(f"Ingesting {len(items)} retrieval relations...")
        self.service.add_retrieval_gt_batch(items, chunk_type="image")

    @staticmethod
    def _pil_image_to_bytes(image: Image.Image) -> tuple[bytes, str]:
        buffer = io.BytesIO()
        img_format = "PNG" if image.mode in ("RGBA", "LA", "P") else "JPEG"
        image.save(buffer, format=img_format)
        mimetype = f"image/{img_format.lower()}"
        return buffer.getvalue(), mimetype

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
        self.service.embed_all_image_chunks_multi_vector(
            self.late_interaction_embedding_model.aget_image_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
