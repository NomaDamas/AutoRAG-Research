"""VisRAG Dataset Ingestor for multi-modal retrieval evaluation.

This unified ingestor supports all 6 VisRAG benchmark datasets:
- ArxivQA: Academic paper figures with questions
- ChartQA: Chart comprehension questions
- MP-DocVQA: Multi-page document visual QA
- InfoVQA: Infographic visual QA
- PlotQA: Scientific plot questions
- SlideVQA: Slide presentation questions

All datasets follow the BEIR-style format with separate corpus, queries, and qrels.
HuggingFace path pattern: openbmb/VisRAG-Ret-Test-{DatasetName}

Reference: https://github.com/OpenBMB/VisRAG
"""

import io
import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from datasets import load_dataset
from llama_index.core.embeddings import MultiModalEmbedding
from PIL import Image

from autorag_research.data.base import MultiModalEmbeddingDataIngestor
from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding
from autorag_research.exceptions import EmbeddingNotSetError, ServiceNotSetError
from autorag_research.util import pil_image_to_bytes

RANDOM_SEED = 42
BATCH_SIZE = 1000

logger = logging.getLogger("AutoRAG-Research")


class VisRAGDatasetName(str, Enum):
    """Supported VisRAG benchmark datasets."""

    ARXIV_QA = "ArxivQA"
    CHART_QA = "ChartQA"
    MP_DOCVQA = "MP-DocVQA"
    INFO_VQA = "InfoVQA"
    PLOT_QA = "PlotQA"
    SLIDE_VQA = "SlideVQA"


@dataclass(frozen=True)
class _DatasetConfig:
    """Configuration for each VisRAG dataset variant."""

    hf_path: str  # HuggingFace dataset path
    has_options: bool  # Whether queries have multiple choice options
    supports_multi_answer: bool  # Whether answers can be a list


# Dataset-specific configurations
_DATASET_CONFIGS: dict[VisRAGDatasetName, _DatasetConfig] = {
    VisRAGDatasetName.ARXIV_QA: _DatasetConfig(
        hf_path="openbmb/VisRAG-Ret-Test-ArxivQA",
        has_options=True,
        supports_multi_answer=False,
    ),
    VisRAGDatasetName.CHART_QA: _DatasetConfig(
        hf_path="openbmb/VisRAG-Ret-Test-ChartQA",
        has_options=True,
        supports_multi_answer=False,
    ),
    VisRAGDatasetName.MP_DOCVQA: _DatasetConfig(
        hf_path="openbmb/VisRAG-Ret-Test-MP-DocVQA",
        has_options=False,
        supports_multi_answer=True,
    ),
    VisRAGDatasetName.INFO_VQA: _DatasetConfig(
        hf_path="openbmb/VisRAG-Ret-Test-InfographicsVQA",
        has_options=False,
        supports_multi_answer=True,
    ),
    VisRAGDatasetName.PLOT_QA: _DatasetConfig(
        hf_path="openbmb/VisRAG-Ret-Test-PlotQA",
        has_options=True,
        supports_multi_answer=False,
    ),
    VisRAGDatasetName.SLIDE_VQA: _DatasetConfig(
        hf_path="openbmb/VisRAG-Ret-Test-SlideVQA",
        has_options=False,
        supports_multi_answer=False,
    ),
}


def _format_query_with_options(question: str, options: list[str]) -> str:
    """Format query with question and multiple choice options."""
    options_text = "\n".join(options)
    return f"Given the following query and options, select the correct option.\n\nQuery: {question}\n\nOptions: {options_text}"


class VisRAGIngestor(MultiModalEmbeddingDataIngestor):
    """Unified ingestor for all VisRAG benchmark datasets.

    The datasets follow BEIR-style format with separate corpus, queries, and qrels:
    - corpus: Images from various document types
    - queries: Context-independent questions (filtered by GPT-4o)
    - qrels: Query-to-corpus relevance judgments

    Attributes:
        dataset_name: Which VisRAG dataset to use.
        embedding_model: MultiModal embedding model for single-vector embeddings.
        late_interaction_embedding_model: Multi-vector embedding model (e.g., ColPali).

    Example:
        >>> ingestor = VisRAGIngestor(VisRAGDatasetName.CHART_QA)
        >>> ingestor.set_service(service)
        >>> ingestor.ingest(query_limit=100, corpus_limit=500)
    """

    def __init__(
        self,
        dataset_name: VisRAGDatasetName,
        embedding_model: MultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
    ):
        super().__init__(embedding_model, late_interaction_embedding_model)
        self.dataset_name = dataset_name
        self._config = _DATASET_CONFIGS[dataset_name]
        self._corpus_ds = None
        self._queries_ds = None
        self._qrels_ds = None

    def detect_primary_key_type(self) -> Literal["bigint"] | Literal["string"]:
        return "string"

    def _load_corpus(self):
        """Load corpus subset (images)."""
        if self._corpus_ds is None:
            logger.info(f"Loading {self._config.hf_path} corpus...")
            self._corpus_ds = load_dataset(self._config.hf_path, name="corpus", split="train")
        return self._corpus_ds

    def _load_queries(self):
        """Load queries subset."""
        if self._queries_ds is None:
            logger.info(f"Loading {self._config.hf_path} queries...")
            self._queries_ds = load_dataset(self._config.hf_path, name="queries", split="train")
        return self._queries_ds

    def _load_qrels(self) -> dict[str, dict[str, int]]:
        """Load and parse qrels into dict format.

        Returns:
            Dict mapping query_id -> {corpus_id: score}
        """
        if self._qrels_ds is None:
            logger.info(f"Loading {self._config.hf_path} qrels...")
            self._qrels_ds = load_dataset(self._config.hf_path, name="qrels", split="train")

        qrels: dict[str, dict[str, int]] = {}
        for row in self._qrels_ds:
            query_id = row["query-id"]
            corpus_id = row["corpus-id"]
            score = row["score"]
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][corpus_id] = score
        return qrels

    def _sample_queries(
        self,
        query_index: dict[str, dict],
        qrels: dict[str, dict[str, int]],
        query_limit: int | None,
        rng: random.Random,
    ) -> list[str]:
        """Sample queries that have valid qrels.

        Args:
            query_index: Dict mapping query_id to row data.
            qrels: Dict mapping query_id -> {corpus_id: score}.
            query_limit: Maximum number of queries to sample.
            rng: Random number generator.

        Returns:
            List of selected query IDs.
        """
        all_query_ids = [qid for qid in query_index if qrels.get(qid)]

        if query_limit is not None and query_limit < len(all_query_ids):
            selected_query_ids = rng.sample(all_query_ids, query_limit)
        else:
            selected_query_ids = all_query_ids

        logger.info(f"Selected {len(selected_query_ids)} queries from {len(all_query_ids)} total")
        return selected_query_ids

    def _collect_gold_corpus_ids(
        self,
        query_ids: list[str],
        qrels: dict[str, dict[str, int]],
    ) -> set[str]:
        """Collect gold corpus IDs from selected queries.

        Args:
            query_ids: List of query IDs.
            qrels: Dict mapping query_id -> {corpus_id: score}.

        Returns:
            Set of gold corpus IDs.
        """
        gold_corpus_ids: set[str] = set()
        for qid in query_ids:
            for corpus_id, score in qrels[qid].items():
                if score > 0:
                    gold_corpus_ids.add(corpus_id)
        logger.info(f"Found {len(gold_corpus_ids)} gold corpus IDs")
        return gold_corpus_ids

    def _filter_corpus(
        self,
        all_corpus_ids: list[str],
        gold_corpus_ids: set[str],
        corpus_limit: int | None,
        rng: random.Random,
    ) -> list[str]:
        """Filter corpus to include gold IDs plus random samples.

        Args:
            all_corpus_ids: All corpus IDs.
            gold_corpus_ids: Gold corpus IDs that must be included.
            corpus_limit: Maximum corpus size.
            rng: Random number generator.

        Returns:
            List of selected corpus IDs.
        """
        if corpus_limit is not None and corpus_limit < len(all_corpus_ids):
            non_gold_ids = [cid for cid in all_corpus_ids if cid not in gold_corpus_ids]
            remaining_slots = max(0, corpus_limit - len(gold_corpus_ids))

            if remaining_slots > 0 and len(non_gold_ids) > remaining_slots:
                sampled_non_gold = rng.sample(non_gold_ids, remaining_slots)
            else:
                sampled_non_gold = non_gold_ids[:remaining_slots]

            selected = list(gold_corpus_ids) + sampled_non_gold
        else:
            selected = all_corpus_ids

        logger.info(f"Selected {len(selected)} corpus items from {len(all_corpus_ids)} total")
        return selected

    def _format_query(self, row: dict) -> str:
        """Format query text based on dataset configuration.

        Args:
            row: Query row from the dataset.

        Returns:
            Formatted query string.
        """
        query_text = row["query"]
        options = row.get("options")

        if self._config.has_options and options:
            return _format_query_with_options(query_text, options)
        return query_text

    def _extract_answers(self, row: dict) -> list[str]:
        """Extract answers based on dataset configuration.

        Args:
            row: Query row from the dataset.

        Returns:
            List of answer strings.
        """
        answer = row.get("answer")

        if answer is None:
            return []

        if self._config.supports_multi_answer and isinstance(answer, list):
            return [str(a) for a in answer if a]

        return [str(answer)]

    def _ingest_corpus(
        self,
        corpus_ids: list[str],
        corpus_index: dict[str, dict],
    ) -> tuple[dict[str, str], int]:
        """Ingest corpus images in batches.

        Args:
            corpus_ids: Corpus IDs to ingest.
            corpus_index: Dict mapping corpus_id to row data.

        Returns:
            Tuple of (corpus_id_to_pk mapping, skipped count).
        """
        if self.service is None:
            raise ServiceNotSetError

        corpus_id_to_pk: dict[str, str] = {}
        skipped = 0

        for batch_start in range(0, len(corpus_ids), BATCH_SIZE):
            batch_ids = corpus_ids[batch_start : batch_start + BATCH_SIZE]
            image_data, valid_ids, batch_skipped = self._process_image_batch(batch_ids, corpus_index)
            skipped += batch_skipped

            if image_data:
                pks = self.service.add_image_chunks(image_data)
                for cid, pk in zip(valid_ids, pks, strict=True):
                    corpus_id_to_pk[cid] = pk

            logger.info(f"Processed corpus {batch_start + len(batch_ids)}/{len(corpus_ids)} (skipped: {skipped})")

        return corpus_id_to_pk, skipped

    def _process_image_batch(
        self,
        batch_ids: list[str],
        corpus_index: dict[str, dict],
    ) -> tuple[list[dict], list[str], int]:
        """Process a batch of images for ingestion.

        Args:
            batch_ids: Batch of corpus IDs.
            corpus_index: Dict mapping corpus_id to row data.

        Returns:
            Tuple of (image_chunks_data, valid_ids, skipped_count).
        """
        image_chunks_data: list[dict] = []
        valid_ids: list[str] = []
        skipped = 0

        for corpus_id in batch_ids:
            row = corpus_index[corpus_id]
            try:
                img = row["image"]
                if isinstance(img, Image.Image):
                    img_bytes, mimetype = pil_image_to_bytes(img)
                else:
                    img_bytes, mimetype = pil_image_to_bytes(Image.open(io.BytesIO(img)))

                image_chunks_data.append({"id": corpus_id, "contents": img_bytes, "mimetype": mimetype})
                valid_ids.append(corpus_id)
            except Exception as e:
                logger.warning(f"Failed to process image {corpus_id}: {e}")
                skipped += 1

        return image_chunks_data, valid_ids, skipped

    def _ingest_queries(
        self,
        query_ids: list[str],
        query_index: dict[str, dict],
    ) -> dict[str, str]:
        """Ingest queries in batches.

        Args:
            query_ids: Query IDs to ingest.
            query_index: Dict mapping query_id to row data.

        Returns:
            Dict mapping query_id to database PK.
        """
        if self.service is None:
            raise ServiceNotSetError

        query_id_to_pk: dict[str, str] = {}

        for batch_start in range(0, len(query_ids), BATCH_SIZE):
            batch_ids = query_ids[batch_start : batch_start + BATCH_SIZE]
            queries_data: list[dict] = []

            for query_id in batch_ids:
                row = query_index[query_id]
                formatted_query = self._format_query(row)
                answers = self._extract_answers(row)
                queries_data.append({
                    "id": query_id,
                    "contents": formatted_query,
                    "generation_gt": answers if answers else None,
                })

            if queries_data:
                pks = self.service.add_queries(queries_data)
                for qid, pk in zip(batch_ids, pks, strict=True):
                    query_id_to_pk[qid] = pk

            logger.info(f"Processed queries {batch_start + len(batch_ids)}/{len(query_ids)}")

        return query_id_to_pk

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "train",
        query_limit: int | None = None,
        corpus_limit: int | None = None,
    ) -> None:
        """Ingest VisRAG dataset.

        Note: VisRAG datasets only have a 'train' split, so all subset
        values are mapped to 'train'.

        Args:
            subset: Dataset split (ignored - always uses 'train').
            query_limit: Maximum number of queries to ingest.
            corpus_limit: Maximum number of corpus images to ingest.
                         Gold IDs from selected queries are always included.
        """
        super().ingest(subset, query_limit, corpus_limit)

        if self.service is None:
            raise ServiceNotSetError

        # Load all dataset components
        corpus_ds = self._load_corpus()
        queries_ds = self._load_queries()
        qrels = self._load_qrels()
        rng = random.Random(RANDOM_SEED)  # noqa: S311

        # Build indexes
        query_index = {row["query-id"]: row for row in queries_ds}
        corpus_index = {row["corpus-id"]: row for row in corpus_ds}

        # Sample and filter
        selected_query_ids = self._sample_queries(query_index, qrels, query_limit, rng)
        gold_corpus_ids = self._collect_gold_corpus_ids(selected_query_ids, qrels)
        selected_corpus_ids = self._filter_corpus(list(corpus_index.keys()), gold_corpus_ids, corpus_limit, rng)

        # Ingest data
        corpus_id_to_pk, skipped = self._ingest_corpus(selected_corpus_ids, corpus_index)
        query_id_to_pk = self._ingest_queries(selected_query_ids, query_index)
        self._ingest_qrels(selected_query_ids, qrels, query_id_to_pk, corpus_id_to_pk, set(selected_corpus_ids))

        logger.info(
            f"[{self.dataset_name.value}] Ingestion complete: "
            f"{len(query_id_to_pk)} queries, {len(corpus_id_to_pk)} images, {skipped} skipped"
        )

    def _ingest_qrels(
        self,
        query_ids: list[str],
        qrels: dict[str, dict[str, int]],
        query_id_to_pk: dict[str, str],
        corpus_id_to_pk: dict[str, str],
        selected_corpus_ids_set: set[str],
    ) -> None:
        """Ingest qrels (retrieval ground truth).

        Args:
            query_ids: List of query IDs to process.
            qrels: Dict mapping query_id -> {corpus_id: score}.
            query_id_to_pk: Mapping from original query_id to database PK.
            corpus_id_to_pk: Mapping from original corpus_id to database PK.
            selected_corpus_ids_set: Set of corpus IDs that were actually ingested.
        """
        if self.service is None:
            raise ServiceNotSetError

        from autorag_research.orm.models.retrieval_gt import image, or_all

        qrels_items: list[tuple[str, list[str]]] = []

        for query_id in query_ids:
            if query_id not in query_id_to_pk:
                continue

            query_pk = query_id_to_pk[query_id]
            gold_pks: list[str] = []

            for corpus_id, score in qrels[query_id].items():
                if score > 0 and corpus_id in selected_corpus_ids_set and corpus_id in corpus_id_to_pk:
                    gold_pks.append(corpus_id_to_pk[corpus_id])

            if gold_pks:
                qrels_items.append((query_pk, gold_pks))

        if qrels_items:
            self.service.add_retrieval_gt_batch(
                [(qid, or_all(gold_ids, image)) for qid, gold_ids in qrels_items],  # ty: ignore[invalid-argument-type]
                chunk_type="image",
            )

        logger.info(f"Added {len(qrels_items)} qrel entries")

    def embed_all(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        """Embed all queries and image chunks with single-vector embeddings."""
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

    def embed_all_late_interaction(
        self,
        max_concurrency: int = 16,
        batch_size: int = 128,
    ) -> None:
        """Embed all queries and image chunks with multi-vector embeddings."""
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
