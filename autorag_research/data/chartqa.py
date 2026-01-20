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


class ChartQAIngestor(MultiModalEmbeddingDataIngestor):
    def __init__(
        self,
        embedding_model: MultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
        split: str = "test",
    ):
        super().__init__(embedding_model, late_interaction_embedding_model)
        self.ds = load_dataset("HuggingFaceM4/ChartQA", split=split)

    def detect_primary_key_type(self) -> Literal["bigint"] | Literal["string"]:
        return "bigint"

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        corpus_limit: int | None = None,
    ) -> None:
        super().ingest(subset, query_limit, corpus_limit)
        if self.service is None:
            raise ServiceNotSetError

        image_list: list[Image.Image] = list(self.ds["image"])  # ty: ignore[non-subscriptable]
        queries: list[str] = list(self.ds["query"])  # ty: ignore[non-subscriptable]
        labels: list[list[str]] = list(self.ds["label"])  # ty: ignore[non-subscriptable]

        if not (len(image_list) == len(queries) == len(labels)):
            raise ValueError("Length mismatch among image_list, queries, and labels.")  # noqa: TRY003

        total_count = len(queries)
        effective_limit = total_count
        if query_limit is not None:
            effective_limit = min(effective_limit, query_limit)
        if corpus_limit is not None:
            effective_limit = min(effective_limit, corpus_limit)

        if effective_limit < total_count:
            rng = random.Random(RANDOM_SEED)  # noqa: S311
            selected_indices = sorted(rng.sample(range(total_count), effective_limit))
        else:
            selected_indices = list(range(total_count))

        image_list = [image_list[i] for i in selected_indices]
        queries = [queries[i] for i in selected_indices]
        labels = [labels[i] for i in selected_indices]

        image_bytes_list = self._pil_images_to_bytes(image_list)

        query_pk_list = self.service.add_queries([
            {"contents": query, "generation_gt": label} for query, label in zip(queries, labels, strict=True)
        ])

        image_chunk_pk_list = self.service.add_image_chunks([
            {"contents": content, "mimetype": mimetype} for content, mimetype in image_bytes_list
        ])

        self._ingest_qrels(query_pk_list, image_chunk_pk_list)

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

    def _ingest_qrels(self, query_pk_list: list[int | str], image_chunk_pk_list: list[int | str]) -> None:
        if self.service is None:
            raise ServiceNotSetError
        self.service.add_retrieval_gt_batch(
            [
                (query_pk, image_chunk_pk)
                for query_pk, image_chunk_pk in zip(query_pk_list, image_chunk_pk_list, strict=True)
            ],
            chunk_type="image",
        )

    @staticmethod
    def _pil_images_to_bytes(image_list: list[Image.Image]) -> list[tuple[bytes, str]]:
        results = []
        for img in image_list:
            buffer = io.BytesIO()
            img_format = "PNG" if img.mode in ("RGBA", "LA", "P") else "JPEG"
            img.save(buffer, format=img_format)
            mimetype = f"image/{img_format.lower()}"
            results.append((buffer.getvalue(), mimetype))
        return results
