import ast
import random
from abc import ABC
from typing import Literal

from datasets import load_dataset
from llama_index.core.embeddings import MultiModalEmbedding

from autorag_research.data.base import MultiModalEmbeddingDataIngestor
from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding
from autorag_research.exceptions import (
    EmbeddingNotSetError,
    InvalidDatasetNameError,
    ServiceNotSetError,
)
from autorag_research.util import pil_image_to_bytes

RANDOM_SEED = 42

ViDoReDatasets = [
    "arxivqa_test_subsampled",  # options
    "docvqa_test_subsampled",  # no answer
    "infovqa_test_subsampled",  # no answer
    "tabfquad_test_subsampled",
    "tatdqa_test",
    "shiftproject_test",
    "syntheticDocQA_artificial_intelligence_test",
    "syntheticDocQA_energy_test",
    "syntheticDocQA_government_reports_test",
    "syntheticDocQA_healthcare_industry_test",
]


class ViDoReIngestor(MultiModalEmbeddingDataIngestor, ABC):
    def __init__(
        self,
        dataset_name: str,
        embedding_model: MultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
    ):
        super().__init__(embedding_model, late_interaction_embedding_model)
        self.ds = load_dataset(f"vidore/{dataset_name}")["test"]  # ty: ignore[non-subscriptable]
        if dataset_name not in ViDoReDatasets:
            raise InvalidDatasetNameError(dataset_name)

    def embed_all(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        """Embed all queries and image chunks using single-vector embedding model.

        Args:
            max_concurrency: Maximum number of concurrent embedding operations.
            batch_size: Number of items to process per batch.

        Raises:
            EmbeddingNotSetError: If embedding_model is not set.
        """
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
        """Embed all queries and image chunks using multi-vector (late interaction) embedding model.

        Args:
            max_concurrency: Maximum number of concurrent embedding operations.
            batch_size: Number of items to process per batch.

        Raises:
            EmbeddingNotSetError: If late_interaction_embedding_model is not set.
        """
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

    def ingest_qrels(self, query_pk_list: list[int | str], image_chunk_pk_list: list[int | str]) -> None:
        """Add retrieval ground truth for image chunks (1:1 query to image mapping)."""
        if self.service is None:
            raise ServiceNotSetError
        self.service.add_retrieval_gt_batch(
            [
                (query_pk, image_chunk_pk)
                for query_pk, image_chunk_pk in zip(query_pk_list, image_chunk_pk_list, strict=True)
            ],
            chunk_type="image",
        )


class ViDoReArxivQAIngestor(ViDoReIngestor):
    def __init__(
        self,
        embedding_model: MultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
    ):
        super().__init__(
            "arxivqa_test_subsampled",
            embedding_model,
            late_interaction_embedding_model,
        )

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

        image_list = list(self.ds["image"])  # ty: ignore[invalid-argument-type]
        queries = list(self.ds["query"])  # ty: ignore[invalid-argument-type]
        options = ["\n".join(ast.literal_eval(opt)) for opt in list(self.ds["options"])]  # ty: ignore[invalid-argument-type]
        queries = [
            f"Given the following query and options, select the correct option.\n\nQuery: {q}\n\nOptions: {opts}"
            for q, opts in zip(queries, options, strict=True)
        ]
        answers = list(self.ds["answer"])  # ty: ignore[invalid-argument-type]

        if not (len(image_list) == len(queries) == len(answers)):
            raise ValueError("Length mismatch among image_list, queries, and answers.")  # noqa: TRY003

        # For ViDoRe, each query has exactly one corresponding image (1:1 mapping)
        # So corpus_limit and query_limit effectively mean the same thing
        # Use the minimum of both limits if both are set
        total_count = len(queries)
        effective_limit = total_count
        if query_limit is not None:
            effective_limit = min(effective_limit, query_limit)
        if corpus_limit is not None:
            effective_limit = min(effective_limit, corpus_limit)

        # Sample indices if limit is less than total
        if effective_limit < total_count:
            rng = random.Random(RANDOM_SEED)  # noqa: S311
            selected_indices = sorted(rng.sample(range(total_count), effective_limit))
        else:
            selected_indices = list(range(total_count))

        # Filter data by selected indices
        image_list = [image_list[i] for i in selected_indices]
        queries = [queries[i] for i in selected_indices]
        answers = [answers[i] for i in selected_indices]

        # Convert PIL images to bytes
        image_bytes_list = [pil_image_to_bytes(img) for img in image_list]

        query_pk_list = self.service.add_queries([
            {
                "contents": query,
                "generation_gt": [ans],
            }
            for query, ans in zip(queries, answers, strict=True)
        ])

        image_chunk_pk_list = self.service.add_image_chunks([
            {
                "contents": content,
                "mimetype": mimetype,
            }
            for content, mimetype in image_bytes_list
        ])

        self.ingest_qrels(query_pk_list, image_chunk_pk_list)
