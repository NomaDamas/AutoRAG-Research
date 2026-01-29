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
from typing import Literal

import pandas as pd
from llama_index.core.embeddings import MultiModalEmbedding
from PIL import Image

from autorag_research.data.base import MultiModalEmbeddingDataIngestor
from autorag_research.data.registry import register_ingestor
from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding
from autorag_research.exceptions import ServiceNotSetError
from autorag_research.util import pil_image_to_bytes

RANDOM_SEED = 42
BATCH_SIZE = 1000

logger = logging.getLogger("AutoRAG-Research")


VisRAGDatasetName = Literal["ArxivQA", "ChartQA", "MP-DocVQA", "InfoVQA", "PlotQA", "SlideVQA"]


@dataclass(frozen=True)
class _DatasetConfig:
    """Configuration for each VisRAG dataset variant."""

    hf_path: str  # HuggingFace dataset path
    has_options: bool  # Whether queries have multiple choice options
    supports_multi_answer: bool  # Whether answers can be a list


# Dataset-specific configurations
_DATASET_CONFIGS: dict[VisRAGDatasetName, _DatasetConfig] = {
    "ArxivQA": _DatasetConfig(
        hf_path="openbmb/VisRAG-Ret-Test-ArxivQA",
        has_options=True,
        supports_multi_answer=False,
    ),
    "ChartQA": _DatasetConfig(
        hf_path="openbmb/VisRAG-Ret-Test-ChartQA",
        has_options=False,
        supports_multi_answer=False,
    ),
    "MP-DocVQA": _DatasetConfig(
        hf_path="openbmb/VisRAG-Ret-Test-MP-DocVQA",
        has_options=False,
        supports_multi_answer=True,
    ),
    "InfoVQA": _DatasetConfig(
        hf_path="openbmb/VisRAG-Ret-Test-InfoVQA",
        has_options=False,
        supports_multi_answer=True,
    ),
    "PlotQA": _DatasetConfig(
        hf_path="openbmb/VisRAG-Ret-Test-PlotQA",
        has_options=False,
        supports_multi_answer=False,
    ),
    "SlideVQA": _DatasetConfig(
        hf_path="openbmb/VisRAG-Ret-Test-SlideVQA",
        has_options=False,
        supports_multi_answer=False,
    ),
}


@register_ingestor(
    name="visrag",
    description="Datasets that used in the VisRAG paper for benchmarking",
)
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
        >>> ingestor = VisRAGIngestor("ChartQA")
        >>> ingestor.set_service(service)
        >>> ingestor.ingest(query_limit=100, min_corpus_cnt=500)
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
        self._df_cache: dict[str, pd.DataFrame] = {}
        self._corpus_ds: list[dict] | None = None  # Raw corpus for image access

    def detect_primary_key_type(self) -> Literal["bigint"] | Literal["string"]:
        return "string"

    def _load(self, name: Literal["corpus", "queries", "qrels"]) -> pd.DataFrame:
        """Load dataset subset as DataFrame with caching."""
        if name not in self._df_cache:
            logger.info(f"Loading {self._config.hf_path} {name}...")
            from datasets import load_dataset

            ds = load_dataset(self._config.hf_path, name=name, split="train")
            if name == "corpus":
                self._corpus_ds = list(ds)  # Keep raw data for image access
            self._df_cache[name] = ds.to_pandas()  # ty: ignore[possibly-missing-attribute,invalid-assignment]
        return self._df_cache[name]

    def _sample_queries(
        self,
        queries_df: pd.DataFrame,
        query_limit: int | None,
        rng: random.Random,
    ) -> list[str]:
        """Sample queries from the queries DataFrame.

        Args:
            queries_df: DataFrame with query-id column.
            query_limit: Maximum number of queries to sample.
            rng: Random number generator (Python random for reproducibility).

        Returns:
            List of selected query IDs.
        """
        all_query_ids = queries_df["query-id"].tolist()

        if query_limit is not None and query_limit < len(all_query_ids):
            selected_query_ids = rng.sample(all_query_ids, query_limit)
        else:
            selected_query_ids = all_query_ids

        logger.info(f"Selected {len(selected_query_ids)} queries from {len(all_query_ids)} total")
        return selected_query_ids

    def _collect_gold_corpus_ids(
        self,
        query_ids: list[str],
        qrels_df: pd.DataFrame,
    ) -> set[str]:
        """Collect gold corpus IDs from selected queries.

        Args:
            query_ids: List of query IDs.
            qrels_df: DataFrame with query-id, corpus-id, score columns.

        Returns:
            Set of gold corpus IDs.
        """
        gold_corpus_ids = set(qrels_df.loc[qrels_df["query-id"].isin(query_ids), "corpus-id"])
        logger.info(f"Found {len(gold_corpus_ids)} gold corpus IDs")
        return gold_corpus_ids

    def _filter_corpus(
        self,
        corpus_df: pd.DataFrame,
        gold_corpus_ids: set[str],
        min_corpus_cnt: int | None,
        rng: random.Random,
    ) -> list[str]:
        """Filter corpus to include gold IDs plus random samples.

        Args:
            corpus_df: DataFrame with corpus-id column.
            gold_corpus_ids: Gold corpus IDs that must be included.
            min_corpus_cnt: Maximum corpus size.
            rng: Random number generator (Python random for reproducibility).

        Returns:
            List of selected corpus IDs.
        """
        all_corpus_ids = corpus_df["corpus-id"].tolist()

        if min_corpus_cnt is not None and min_corpus_cnt < len(all_corpus_ids):
            non_gold_ids = [cid for cid in all_corpus_ids if cid not in gold_corpus_ids]
            remaining_slots = max(0, min_corpus_cnt - len(gold_corpus_ids))

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

        if self._config.has_options and options is not None and len(options) > 0:
            options_text = "\n".join(list(options))
            return f"Given the following query and options, select the correct option.\n\nQuery: {query_text}\n\nOptions: {options_text}"
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
        min_corpus_cnt: int | None = None,
    ) -> None:
        """Ingest VisRAG dataset.

        Note: VisRAG datasets only have a 'train' split, so all subset
        values are mapped to 'train'.

        Args:
            subset: Dataset split (ignored - always uses 'train').
            query_limit: Maximum number of queries to ingest.
            min_corpus_cnt: Maximum number of corpus images to ingest.
                         Gold IDs from selected queries are always included.
        """
        super().ingest(subset, query_limit, min_corpus_cnt)

        if self.service is None:
            raise ServiceNotSetError

        # Load all dataset components as DataFrames
        corpus_df = self._load("corpus")
        queries_df = self._load("queries")
        qrels_df = self._load("qrels")
        rng = random.Random(RANDOM_SEED)

        # Build indexes for row lookup
        query_index = {row["query-id"]: row.to_dict() for _, row in queries_df.iterrows()}
        corpus_index = {row["corpus-id"]: row for row in self._corpus_ds or []}  # Use raw data for images

        # Sample and filter using DataFrames
        selected_query_ids = self._sample_queries(queries_df, query_limit, rng)
        gold_corpus_ids = self._collect_gold_corpus_ids(selected_query_ids, qrels_df)
        selected_corpus_ids = self._filter_corpus(corpus_df, gold_corpus_ids, min_corpus_cnt, rng)

        # Ingest data
        corpus_id_to_pk, skipped = self._ingest_corpus(selected_corpus_ids, corpus_index)
        query_id_to_pk = self._ingest_queries(selected_query_ids, query_index)
        self._ingest_qrels(selected_query_ids, qrels_df, query_id_to_pk, corpus_id_to_pk, set(selected_corpus_ids))

        logger.info(
            f"[{self.dataset_name}] Ingestion complete: "
            f"{len(query_id_to_pk)} queries, {len(corpus_id_to_pk)} images, {skipped} skipped"
        )

    def _ingest_qrels(
        self,
        query_ids: list[str],
        qrels_df: pd.DataFrame,
        query_id_to_pk: dict[str, str],
        corpus_id_to_pk: dict[str, str],
        selected_corpus_ids_set: set[str],
    ) -> None:
        """Ingest qrels (retrieval ground truth).

        Args:
            query_ids: List of query IDs to process.
            qrels_df: DataFrame with query-id, corpus-id, score columns.
            query_id_to_pk: Mapping from original query_id to database PK.
            corpus_id_to_pk: Mapping from original corpus_id to database PK.
            selected_corpus_ids_set: Set of corpus IDs that were actually ingested.
        """
        if self.service is None:
            raise ServiceNotSetError

        from autorag_research.orm.models.retrieval_gt import image, or_all

        filtered_qrels = qrels_df[
            qrels_df["query-id"].isin(query_ids) & qrels_df["corpus-id"].isin(selected_corpus_ids_set)
        ]

        # Group by query_id to collect gold corpus PKs
        qrels_items: list[tuple[str, list[str]]] = []
        for query_id, group in filtered_qrels.groupby("query-id"):
            if query_id not in query_id_to_pk:
                continue

            query_pk = query_id_to_pk[query_id]
            gold_pks = [corpus_id_to_pk[cid] for cid in group["corpus-id"] if cid in corpus_id_to_pk]

            if gold_pks:
                qrels_items.append((query_pk, gold_pks))

        if qrels_items:
            self.service.add_retrieval_gt_batch(
                [(qid, or_all(gold_ids, image)) for qid, gold_ids in qrels_items],  # ty: ignore[invalid-argument-type]
                chunk_type="image",
            )

        logger.info(f"Added {len(qrels_items)} qrel entries")
