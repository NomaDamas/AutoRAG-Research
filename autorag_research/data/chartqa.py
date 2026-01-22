"""ChartQA Ingestor using VisRAG filtered dataset.

Uses the VisRAG-Ret-Test-ChartQA dataset which filters context-dependent questions
using GPT-4o judge, preserving only ~10% of the original ChartQA dataset for
better retrieval evaluation quality.

Dataset: https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-ChartQA

The dataset follows BEIR-style structure with separate subsets:
- corpus: Chart images with corpus-id
- queries: Filtered questions with query-id, query, answer
- qrels: Query-document relevance judgments
"""

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
from autorag_research.orm.models import image, or_all

logger = logging.getLogger("AutoRAG-Research")

RANDOM_SEED = 42
DATASET_NAME = "openbmb/VisRAG-Ret-Test-ChartQA"


class ChartQAIngestor(MultiModalEmbeddingDataIngestor):
    """Ingestor for VisRAG filtered ChartQA dataset.

    This ingestor uses the VisRAG-Ret-Test-ChartQA dataset which contains
    filtered chart QA pairs optimized for retrieval evaluation.

    Key characteristics:
    - Uses string primary keys (corpus-id format)
    - Separate corpus (500 images), queries (63), and qrels
    - Each query may map to one or more corpus images
    - Answer field provides generation ground truth
    """

    def __init__(
        self,
        embedding_model: MultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
    ):
        super().__init__(embedding_model, late_interaction_embedding_model)
        self._corpus_ds: list[dict[str, Any]] | None = None
        self._queries_ds: list[dict[str, Any]] | None = None
        self._qrels_ds: list[dict[str, Any]] | None = None

    def _load_datasets(self) -> None:
        """Load all dataset subsets lazily."""
        if self._corpus_ds is None:
            logger.info(f"Loading corpus from {DATASET_NAME}...")
            ds = load_dataset(DATASET_NAME, name="corpus", split="train")
            self._corpus_ds = list(ds)  # ty: ignore[invalid-argument-type]
        if self._queries_ds is None:
            logger.info(f"Loading queries from {DATASET_NAME}...")
            ds = load_dataset(DATASET_NAME, name="queries", split="train")
            self._queries_ds = list(ds)  # ty: ignore[invalid-argument-type]
        if self._qrels_ds is None:
            logger.info(f"Loading qrels from {DATASET_NAME}...")
            ds = load_dataset(DATASET_NAME, name="qrels", split="train")
            self._qrels_ds = list(ds)  # ty: ignore[invalid-argument-type]

    def _build_qrels_dict(self) -> dict[str, dict[str, int]]:
        """Build qrels dictionary from qrels dataset.

        Returns:
            Dict mapping query-id to dict of corpus-id -> score.
        """
        if self._qrels_ds is None:
            return {}
        qrels: dict[str, dict[str, int]] = {}
        for row in self._qrels_ds:
            qid = str(row["query-id"])
            cid = str(row["corpus-id"])
            score = int(row["score"])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][cid] = score
        return qrels

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        """VisRAG ChartQA uses string IDs (e.g., 'chartqa/test/png/...')."""
        return "string"

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        corpus_limit: int | None = None,
    ) -> None:
        """Ingest data from VisRAG filtered ChartQA dataset.

        Args:
            subset: Ignored for VisRAG dataset (only 'train' split available).
            query_limit: Maximum number of queries to ingest.
            corpus_limit: Maximum number of corpus images to ingest.
                          Gold images from selected queries are always included.
        """
        super().ingest(subset, query_limit, corpus_limit)
        if self.service is None:
            raise ServiceNotSetError

        self._load_datasets()
        qrels = self._build_qrels_dict()
        rng = random.Random(RANDOM_SEED)  # noqa: S311

        # Step 1: Sample queries and collect gold corpus IDs
        query_ids, gold_corpus_ids = self._sample_queries(qrels, query_limit, rng)

        # Step 2: Filter corpus (gold IDs + random samples)
        corpus_ids = self._filter_corpus(gold_corpus_ids, corpus_limit, rng)
        corpus_ids_set = set(corpus_ids)

        # Step 3: Ingest queries
        self._ingest_queries(query_ids)

        # Step 4: Ingest corpus images
        self._ingest_corpus(corpus_ids)

        # Step 5: Ingest qrels
        self._ingest_qrels(query_ids, qrels, corpus_ids_set)

        logger.info("ChartQA ingestion complete.")

    def _sample_queries(
        self,
        qrels: dict[str, dict[str, int]],
        query_limit: int | None,
        rng: random.Random,
    ) -> tuple[list[str], set[str]]:
        """Sample queries and collect gold corpus IDs.

        Args:
            qrels: Query relevance judgments.
            query_limit: Maximum number of queries. None means no limit.
            rng: Random number generator.

        Returns:
            Tuple of (selected query IDs, gold corpus IDs from selected queries).
        """
        if self._queries_ds is None:
            return [], set()

        # Get all query IDs that have qrels
        query_ids = [str(row["query-id"]) for row in self._queries_ds if str(row["query-id"]) in qrels]

        if query_limit is not None and query_limit < len(query_ids):
            query_ids = rng.sample(query_ids, query_limit)
            logger.info(f"Sampled {len(query_ids)} queries from {len(self._queries_ds)} total")

        # Collect gold corpus IDs from selected queries
        gold_corpus_ids: set[str] = set()
        for qid in query_ids:
            if qid in qrels:
                gold_corpus_ids.update(cid for cid, score in qrels[qid].items() if score > 0)

        return query_ids, gold_corpus_ids

    def _filter_corpus(
        self,
        gold_corpus_ids: set[str],
        corpus_limit: int | None,
        rng: random.Random,
    ) -> list[str]:
        """Filter corpus to include gold IDs + random samples up to limit.

        Args:
            gold_corpus_ids: Corpus IDs that must be included.
            corpus_limit: Maximum corpus size. None means no limit.
            rng: Random number generator.

        Returns:
            List of corpus IDs to include.
        """
        if self._corpus_ds is None:
            return []

        all_corpus_ids = [str(row["corpus-id"]) for row in self._corpus_ds]

        if corpus_limit is None:
            return all_corpus_ids

        # Always include gold IDs
        selected = list(gold_corpus_ids & set(all_corpus_ids))
        remaining = [cid for cid in all_corpus_ids if cid not in gold_corpus_ids]

        # Add random samples if needed
        additional_needed = corpus_limit - len(selected)
        if additional_needed > 0 and remaining:
            additional = rng.sample(remaining, min(additional_needed, len(remaining)))
            selected.extend(additional)

        logger.info(
            f"Corpus subset: {len(gold_corpus_ids)} gold + "
            f"{len(selected) - len(gold_corpus_ids)} random = {len(selected)} total"
        )
        return selected

    def _ingest_queries(self, query_ids: list[str]) -> None:
        """Ingest selected queries."""
        if self.service is None:
            raise ServiceNotSetError
        if self._queries_ds is None:
            return

        # Build lookup dict for queries
        queries_dict = {str(row["query-id"]): row for row in self._queries_ds}

        logger.info(f"Ingesting {len(query_ids)} queries...")
        self.service.add_queries([
            {
                "id": qid,
                "contents": str(queries_dict[qid]["query"]),
                "generation_gt": [str(queries_dict[qid]["answer"])],
            }
            for qid in query_ids
        ])

    def _ingest_corpus(self, corpus_ids: list[str]) -> None:
        """Ingest selected corpus images."""
        if self.service is None:
            raise ServiceNotSetError
        if self._corpus_ds is None:
            return

        # Build lookup dict for corpus
        corpus_dict = {str(row["corpus-id"]): row for row in self._corpus_ds}

        logger.info(f"Ingesting {len(corpus_ids)} corpus images...")
        image_data = []
        for cid in corpus_ids:
            img: Image.Image = corpus_dict[cid]["image"]
            content, mimetype = self._pil_image_to_bytes(img)
            image_data.append({"id": cid, "contents": content, "mimetype": mimetype})

        self.service.add_image_chunks(image_data)

    def _ingest_qrels(
        self,
        query_ids: list[str],
        qrels: dict[str, dict[str, int]],
        corpus_ids_set: set[str],
    ) -> None:
        """Ingest query-corpus relevance relations."""
        if self.service is None:
            raise ServiceNotSetError

        for qid in query_ids:
            if qid not in qrels:
                continue
            # Filter to only include corpus IDs that exist in our subset
            gt_ids = [cid for cid, score in qrels[qid].items() if score > 0 and cid in corpus_ids_set]
            if not gt_ids:
                continue
            self.service.add_retrieval_gt(qid, or_all(gt_ids, image), chunk_type="image")

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
    def _pil_image_to_bytes(img: Image.Image) -> tuple[bytes, str]:
        """Convert a PIL Image to bytes with appropriate format."""
        buffer = io.BytesIO()
        img_format = "PNG" if img.mode in ("RGBA", "LA", "P") else "JPEG"
        img.save(buffer, format=img_format)
        mimetype = f"image/{img_format.lower()}"
        return buffer.getvalue(), mimetype
