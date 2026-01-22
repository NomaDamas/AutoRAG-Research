import io
import random
from typing import Literal

from datasets import load_dataset
from llama_index.core.embeddings import MultiModalEmbedding
from PIL import Image

from autorag_research.data.base import MultiModalEmbeddingDataIngestor
from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding
from autorag_research.exceptions import EmbeddingNotSetError, ServiceNotSetError

RANDOM_SEED = 42
DATASET_NAME = "openbmb/VisRAG-Ret-Test-InfoVQA"


class InfoVQAIngestor(MultiModalEmbeddingDataIngestor):
    def __init__(
        self,
        embedding_model: MultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
    ):
        super().__init__(embedding_model, late_interaction_embedding_model)
        self._corpus_ds = None
        self._queries_ds = None
        self._qrels_ds = None

    def _load_datasets(self) -> None:
        if self._corpus_ds is None:
            self._corpus_ds = load_dataset(DATASET_NAME, "corpus", split="train")
        if self._queries_ds is None:
            self._queries_ds = load_dataset(DATASET_NAME, "queries", split="train")
        if self._qrels_ds is None:
            self._qrels_ds = load_dataset(DATASET_NAME, "qrels", split="train")

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

        self._load_datasets()

        qrels_list = list(self._qrels_ds)  # ty: ignore[invalid-argument-type]
        queries_list = list(self._queries_ds)  # ty: ignore[invalid-argument-type]
        corpus_list = list(self._corpus_ds)  # ty: ignore[invalid-argument-type]

        corpus_id_to_row = {row["corpus-id"]: row for row in corpus_list}
        query_id_to_row = {row["query-id"]: row for row in queries_list}

        total_query_count = len(queries_list)
        effective_limit = total_query_count
        if query_limit is not None:
            effective_limit = min(effective_limit, query_limit)
        if corpus_limit is not None:
            effective_limit = min(effective_limit, corpus_limit)

        if effective_limit < total_query_count:
            rng = random.Random(RANDOM_SEED)  # noqa: S311
            selected_query_indices = sorted(rng.sample(range(total_query_count), effective_limit))
        else:
            selected_query_indices = list(range(total_query_count))

        selected_query_ids = {queries_list[i]["query-id"] for i in selected_query_indices}

        selected_qrels = [qrel for qrel in qrels_list if qrel["query-id"] in selected_query_ids]
        selected_corpus_ids = {qrel["corpus-id"] for qrel in selected_qrels}

        image_chunks_data = []
        for corpus_id in selected_corpus_ids:
            corpus_row = corpus_id_to_row[corpus_id]
            image_bytes, mimetype = self._pil_image_to_bytes(corpus_row["image"])
            image_chunks_data.append({
                "id": corpus_id,
                "contents": image_bytes,
                "mimetype": mimetype,
            })

        queries_data = []
        for query_id in selected_query_ids:
            query_row = query_id_to_row[query_id]
            queries_data.append({
                "id": query_id,
                "contents": query_row["query"],
                "generation_gt": list(query_row["answer"]),
            })

        self.service.add_image_chunks(image_chunks_data)
        self.service.add_queries(queries_data)

        qrel_pairs = [(qrel["query-id"], qrel["corpus-id"]) for qrel in selected_qrels]
        self.service.add_retrieval_gt_batch(qrel_pairs, chunk_type="image")

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

    @staticmethod
    def _pil_image_to_bytes(image: Image.Image) -> tuple[bytes, str]:
        buffer = io.BytesIO()
        img_format = "PNG" if image.mode in ("RGBA", "LA", "P") else "JPEG"
        image.save(buffer, format=img_format)
        mimetype = f"image/{img_format.lower()}"
        return buffer.getvalue(), mimetype
