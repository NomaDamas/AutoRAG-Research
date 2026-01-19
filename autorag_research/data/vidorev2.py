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

import io
import random
from collections import defaultdict
from enum import Enum
from typing import Any, Literal

from datasets import load_dataset
from llama_index.core.embeddings import MultiModalEmbedding
from PIL import Image

from autorag_research.data.base import MultiModalEmbeddingDataIngestor
from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding
from autorag_research.exceptions import EmbeddingNotSetError, ServiceNotSetError

RANDOM_SEED = 42


class ViDoReV2DatasetName(str, Enum):
    """Available ViDoReV2 dataset names."""

    ESG_REPORTS_V2 = "esg_reports_v2"
    BIOMEDICAL_LECTURES_V2 = "biomedical_lectures_v2"
    ECONOMICS_REPORTS_V2 = "economics_reports_v2"
    ESG_REPORTS_HUMAN_LABELED_V2 = "esg_reports_human_labeled_v2"


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
        super().ingest(subset, query_limit, corpus_limit)
        if self.service is None:
            raise ServiceNotSetError

        dataset_path = f"vidore/{self.dataset_name.value}"

        # Step 1: Load qrels to build query->corpus mappings
        qrels_ds = load_dataset(dataset_path, "qrels", streaming=True, split="test")
        query_to_corpus_ids, all_query_ids, all_corpus_ids = self._load_qrels(qrels_ds)

        # Step 2: Select queries (random sample if limit is set)
        selected_query_ids = self._select_items(list(all_query_ids), query_limit, "queries")

        # Step 3: Determine required corpus IDs (gold IDs from selected queries)
        required_corpus_ids: set[str] = set()
        for qid in selected_query_ids:
            required_corpus_ids.update(query_to_corpus_ids.get(qid, []))

        # Step 4: If corpus_limit is set, add random corpus items to reach limit
        selected_corpus_ids = self._select_corpus_with_gold_ids(required_corpus_ids, all_corpus_ids, corpus_limit)

        # Step 5: Load and ingest corpus (images)
        corpus_ds = load_dataset(dataset_path, "corpus", streaming=True, split="test")
        corpus_id_to_pk = self._ingest_corpus(corpus_ds, selected_corpus_ids)

        # Step 6: Load and ingest queries
        queries_ds = load_dataset(dataset_path, "queries", streaming=True, split="test")
        query_id_to_pk = self._ingest_queries(queries_ds, selected_query_ids)

        # Step 7: Create retrieval relations
        self._ingest_qrels(
            query_to_corpus_ids,
            selected_query_ids,
            query_id_to_pk,
            corpus_id_to_pk,
        )

    def _load_qrels(self, qrels_ds: Any) -> tuple[dict[str, list[str]], set[str], set[str]]:
        """Load qrels and build query-to-corpus mappings.

        Args:
            qrels_ds: Streaming qrels dataset.

        Returns:
            Tuple of (query_to_corpus mapping, all query IDs, all corpus IDs).
        """
        query_to_corpus: dict[str, list[str]] = defaultdict(list)
        all_query_ids: set[str] = set()
        all_corpus_ids: set[str] = set()

        for row in qrels_ds:
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])
            query_to_corpus[query_id].append(corpus_id)
            all_query_ids.add(query_id)
            all_corpus_ids.add(corpus_id)

        return dict(query_to_corpus), all_query_ids, all_corpus_ids

    def _select_items(self, items: list[str], limit: int | None, item_type: str) -> list[str]:
        """Select items with optional random sampling.

        Args:
            items: List of item IDs.
            limit: Maximum number of items to select (None = all).
            item_type: Type name for logging.

        Returns:
            Selected item IDs.
        """
        if limit is None or limit >= len(items):
            return items

        rng = random.Random(RANDOM_SEED)  # noqa: S311
        return rng.sample(items, limit)

    def _select_corpus_with_gold_ids(
        self,
        required_ids: set[str],
        all_corpus_ids: set[str],
        corpus_limit: int | None,
    ) -> set[str]:
        """Select corpus IDs ensuring gold IDs are included.

        Args:
            required_ids: Gold corpus IDs that must be included.
            all_corpus_ids: All available corpus IDs.
            corpus_limit: Maximum corpus items (None = all).

        Returns:
            Set of selected corpus IDs.
        """
        if corpus_limit is None:
            return all_corpus_ids

        # Always include gold IDs
        selected = set(required_ids)

        # If we still have room, add random items
        remaining_slots = corpus_limit - len(selected)
        if remaining_slots > 0:
            available = list(all_corpus_ids - selected)
            rng = random.Random(RANDOM_SEED)  # noqa: S311
            additional = rng.sample(available, min(remaining_slots, len(available)))
            selected.update(additional)

        return selected

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
            img_bytes, mimetype = self._pil_image_to_bytes(image)

            image_chunks_batch.append({
                "id": corpus_id,
                "contents": img_bytes,
                "mimetype": mimetype,
            })
            corpus_id_to_pk[corpus_id] = corpus_id

        if image_chunks_batch:
            self.service.add_image_chunks(image_chunks_batch)

        return corpus_id_to_pk

    def _ingest_queries(self, queries_ds: Any, selected_ids: list[str]) -> dict[str, str]:
        """Ingest queries.

        Args:
            queries_ds: Streaming queries dataset.
            selected_ids: List of query IDs to ingest.

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
            answer = row.get("answer")

            query_data: dict = {
                "id": query_id,
                "contents": query_text,
            }
            if answer is not None:
                query_data["generation_gt"] = [str(answer)]

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

    @staticmethod
    def _pil_image_to_bytes(image: Image.Image) -> tuple[bytes, str]:
        """Convert PIL image to bytes with mimetype.

        Args:
            image: PIL Image object.

        Returns:
            Tuple of (image_bytes, mimetype).
        """
        buffer = io.BytesIO()
        # Determine format based on image mode
        img_format = "PNG" if image.mode in ("RGBA", "LA", "P") else "JPEG"
        image.save(buffer, format=img_format)
        mimetype = f"image/{img_format.lower()}"
        return buffer.getvalue(), mimetype

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
