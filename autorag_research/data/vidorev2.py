"""ViDoReV2 Dataset Ingestors for AutoRAG-Research.

ViDoReV2 (Visual Document Retrieval Benchmark 2) is a multi-modal benchmark
for evaluating document retrieval systems. Unlike V1, V2 uses a BEIR-like
structure with separate corpus, queries, and qrels subsets.

Key Features:
- BEIR-like structure: corpus, queries, qrels, docs
- Many-to-many query-to-corpus relations (qrels)
- Multi-lingual support (EN/FR/DE/ES)
- Streaming support for large datasets

Available Datasets:
- esg_reports_v2: ESG sustainability reports (228 queries, 1538 pages)
- biomedical_lectures_v2: Biomedical lecture slides (640 queries, 1016 pages)
- economics_reports_v2: Economics and finance reports (232 queries, 452 pages)
- esg_reports_human_labeled_v2: Human-labeled ESG reports (52 queries, 1538 pages)
"""

import random
from enum import Enum
from typing import Any, Literal

from datasets import load_dataset
from llama_index.core.embeddings import MultiModalEmbedding
from PIL import Image

from autorag_research.data.base import MultiModalEmbeddingDataIngestor
from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding
from autorag_research.exceptions import EmbeddingNotSetError, ServiceNotSetError
from autorag_research.util import pil_image_to_bytes

RANDOM_SEED = 42


class ViDoReV2DatasetName(str, Enum):
    """Available ViDoReV2 dataset names."""

    ESG_REPORTS_V2 = "esg_reports_v2"
    BIOMEDICAL_LECTURES_V2 = "biomedical_lectures_v2"
    ECONOMICS_REPORTS_V2 = "economics_reports_v2"


class ViDoReV2Ingestor(MultiModalEmbeddingDataIngestor):
    """Ingestor for ViDoReV2 datasets using streaming.

    ViDoReV2 datasets have a BEIR-like structure with:
    - corpus: Document page images with corpus-id
    - queries: Text queries with query-id
    - qrels: Relevance judgments mapping query-id to corpus-id
    - docs: Optional document metadata

    This ingestor uses streaming mode to avoid downloading entire datasets.

    Example:
        ```python
        from autorag_research.data.vidorev2 import ViDoReV2Ingestor, ViDoReV2DatasetName
        from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService

        ingestor = ViDoReV2Ingestor(ViDoReV2DatasetName.ESG_REPORTS_V2)
        ingestor.set_service(service)
        ingestor.ingest(query_limit=10, corpus_limit=50)
        ```
    """

    def __init__(
        self,
        dataset_name: ViDoReV2DatasetName,
        embedding_model: MultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
    ):
        """Initialize ViDoReV2 ingestor.

        Args:
            dataset_name: Name of the ViDoReV2 dataset to ingest.
            embedding_model: Optional multi-modal embedding model for single-vector embeddings.
            late_interaction_embedding_model: Optional multi-vector embedding model.
        """
        super().__init__(embedding_model, late_interaction_embedding_model)
        self.dataset_name = dataset_name

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        """ViDoReV2 uses string IDs (e.g., 'page_123')."""
        return "string"

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        corpus_limit: int | None = None,
    ) -> None:
        """Ingest ViDoReV2 dataset using streaming.

        Args:
            subset: Dataset split (ViDoReV2 only has 'test' split).
            query_limit: Maximum number of queries to ingest.
            corpus_limit: Maximum number of corpus items to ingest.
                         Gold IDs from selected queries are always included.
        """
        import pandas as pd

        super().ingest(subset, query_limit, corpus_limit)
        if self.service is None:
            raise ServiceNotSetError

        dataset_path = f"vidore/{self.dataset_name.value}"

        # Step 1: Load qrels into pandas and process with groupby
        from datasets import Dataset

        qrels_ds: Dataset = load_dataset(dataset_path, "qrels", streaming=False, split="test")  # ty: ignore[invalid-assignment]
        qrels_df: pd.DataFrame = qrels_ds.to_pandas()  # ty: ignore[invalid-assignment]
        qrels_df["query-id"] = qrels_df["query-id"].astype(str)
        qrels_df["corpus-id"] = qrels_df["corpus-id"].astype(str)

        # Group by query-id: get corpus IDs and unique answers per query
        grouped = qrels_df.groupby("query-id").agg(
            corpus_ids=pd.NamedAgg(column="corpus-id", aggfunc=list),
            answers=pd.NamedAgg(column="answer", aggfunc=lambda x: list(x.dropna().unique())),
        )

        # Step 2: Sample queries if limit is set
        if query_limit is not None and query_limit < len(grouped):
            selected_queries_df = grouped.sample(n=query_limit, random_state=RANDOM_SEED)
        else:
            selected_queries_df = grouped

        selected_query_ids = list(selected_queries_df.index)
        query_to_corpus_ids: dict[str, list[str]] = selected_queries_df["corpus_ids"].to_dict()
        query_to_answers: dict[str, list[str]] = selected_queries_df["answers"].to_dict()

        # Step 3: Get gold corpus IDs from selected queries
        gold_corpus_ids: set[str] = set()
        for corpus_list in query_to_corpus_ids.values():
            gold_corpus_ids.update(corpus_list)

        # Step 4: Determine final corpus IDs (gold + additional if needed for corpus_limit)
        all_corpus_ids = set(qrels_df["corpus-id"].unique())
        if corpus_limit is None or len(gold_corpus_ids) >= corpus_limit:
            selected_corpus_ids = gold_corpus_ids
        else:
            # Need more corpus items - sample from non-gold items
            remaining_slots = corpus_limit - len(gold_corpus_ids)
            available = list(all_corpus_ids - gold_corpus_ids)
            rng = random.Random(RANDOM_SEED)  # noqa: S311
            additional = rng.sample(available, min(remaining_slots, len(available)))
            selected_corpus_ids = gold_corpus_ids | set(additional)

        # Step 5: Load and ingest corpus (images) - streaming for large image data
        corpus_ds = load_dataset(dataset_path, "corpus", streaming=True, split="test")
        corpus_id_to_pk = self._ingest_corpus(corpus_ds, selected_corpus_ids)

        # Step 6: Load and ingest queries - streaming
        queries_ds = load_dataset(dataset_path, "queries", streaming=True, split="test")
        query_id_to_pk = self._ingest_queries(queries_ds, selected_query_ids, query_to_answers)

        # Step 7: Create retrieval relations
        self._ingest_qrels(
            query_to_corpus_ids,
            selected_query_ids,
            query_id_to_pk,
            corpus_id_to_pk,
        )

    def _ingest_corpus(self, corpus_ds: Any, selected_ids: set[str]) -> dict[str, str]:
        """Ingest corpus images.

        Args:
            corpus_ds: Streaming corpus dataset.
            selected_ids: Set of corpus IDs to ingest.

        Returns:
            Mapping from corpus-id to database primary key.
        """
        if self.service is None:
            raise ServiceNotSetError

        corpus_id_to_pk: dict[str, str] = {}
        image_chunks_batch: list[dict] = []

        for row in corpus_ds:
            corpus_id = str(row["corpus-id"])
            if corpus_id not in selected_ids:
                continue

            image: Image.Image = row["image"]
            img_bytes, mimetype = pil_image_to_bytes(image)

            image_chunks_batch.append({
                "id": corpus_id,
                "contents": img_bytes,
                "mimetype": mimetype,
            })
            corpus_id_to_pk[corpus_id] = corpus_id

        if image_chunks_batch:
            self.service.add_image_chunks(image_chunks_batch)

        return corpus_id_to_pk

    def _ingest_queries(
        self,
        queries_ds: Any,
        selected_ids: list[str],
        query_to_answers: dict[str, list[str]],
    ) -> dict[str, str]:
        """Ingest queries.

        Args:
            queries_ds: Streaming queries dataset.
            selected_ids: List of query IDs to ingest.
            query_to_answers: Mapping from query-id to list of ground truth answers (from qrels).

        Returns:
            Mapping from query-id to database primary key.
        """
        if self.service is None:
            raise ServiceNotSetError

        selected_ids_set = set(selected_ids)
        query_id_to_pk: dict[str, str] = {}
        queries_batch: list[dict] = []

        for row in queries_ds:
            query_id = str(row["query-id"])
            if query_id not in selected_ids_set:
                continue

            query_text = str(row["query"])

            query_data: dict = {
                "id": query_id,
                "contents": query_text,
            }

            # Get answers from qrels (not from queries dataset)
            answers = query_to_answers.get(query_id)
            if answers:
                query_data["generation_gt"] = answers

            queries_batch.append(query_data)
            query_id_to_pk[query_id] = query_id

        if queries_batch:
            self.service.add_queries(queries_batch)

        return query_id_to_pk

    def _ingest_qrels(
        self,
        query_to_corpus: dict[str, list[str]],
        selected_query_ids: list[str],
        query_id_to_pk: dict[str, str],
        corpus_id_to_pk: dict[str, str],
    ) -> None:
        """Create retrieval relations from qrels.

        Args:
            query_to_corpus: Mapping of query-id to list of relevant corpus-ids.
            selected_query_ids: Query IDs that were ingested.
            query_id_to_pk: Mapping from query-id to database PK.
            corpus_id_to_pk: Mapping from corpus-id to database PK.
        """
        if self.service is None:
            raise ServiceNotSetError

        from autorag_research.orm.models import image, or_all

        for query_id in selected_query_ids:
            query_pk = query_id_to_pk.get(query_id)
            if query_pk is None:
                continue

            corpus_ids = query_to_corpus.get(query_id, [])
            # Filter to only include corpus IDs that were ingested
            valid_corpus_pks = [corpus_id_to_pk[cid] for cid in corpus_ids if cid in corpus_id_to_pk]

            if valid_corpus_pks:
                # Use or_all with image wrapper for multiple relevant image chunks
                self.service.add_retrieval_gt(
                    query_pk,  # ty: ignore[invalid-argument-type]
                    or_all(valid_corpus_pks, image),
                    chunk_type="image",
                )

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
        """Embed all queries and image chunks using multi-vector embedding model.

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
