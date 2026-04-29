"""SDS KoPub VDR Dataset Ingestor for AutoRAG-Research.

SDS KoPub VDR is a Korean public-document visual document retrieval benchmark.
This ingestor uses the MTEB text-to-image retrieval version, which provides a
BEIR-like structure with separate corpus, queries, and qrels subsets.
"""

from __future__ import annotations

import logging
import operator
import random
from functools import reduce
from typing import Any, Literal

import pandas as pd
from datasets import load_dataset

from autorag_research.data.base import MultiModalEmbeddingDataIngestor
from autorag_research.data.registry import register_ingestor
from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding, SingleVectorMultiModalEmbedding
from autorag_research.exceptions import ServiceNotSetError
from autorag_research.orm.models import ImageId, RetrievalGT, TextId, image, or_all_mixed, text
from autorag_research.util import pil_image_to_bytes

logger = logging.getLogger("AutoRAG-Research")

DATASET_PATH = "mteb/SDSKoPubVDRT2ITRetrieval"
RANDOM_SEED = 42
BATCH_SIZE = 100
QrelsMode = Literal["image", "text", "mixed"]


@register_ingestor(
    name="sds_kopub_vdr",
    description="SDS KoPub VDR Korean public-document visual retrieval benchmark",
    hf_repo="sds-kopub-vdr-dumps",
)
class SDSKoPubVDRIngestor(MultiModalEmbeddingDataIngestor):
    """Ingestor for the SDS KoPub VDR MTEB retrieval dataset.

    The source dataset follows a BEIR-like multimodal retrieval structure:
    ``corpus`` rows contain page image/text pairs, ``queries`` rows contain
    text queries, and ``qrels`` rows map query IDs to relevant corpus page IDs.
    The source uses string IDs, so this ingestor requires a string primary-key
    schema.
    """

    def __init__(
        self,
        qrels_mode: QrelsMode = "image",
        embedding_model: SingleVectorMultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
    ):
        """Initialize SDS KoPub VDR ingestor.

        Args:
            qrels_mode: How qrels should map to chunks:
                - ``image``: image chunks only (default)
                - ``text``: corpus text chunks only
                - ``mixed``: either image or text chunk is relevant
            embedding_model: Optional single-vector multimodal embedding model.
            late_interaction_embedding_model: Optional multi-vector embedding model.
        """
        super().__init__(embedding_model, late_interaction_embedding_model)
        self.qrels_mode = qrels_mode

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        """The MTEB SDS KoPub VDR dataset uses string query and corpus IDs."""
        return "string"

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
    ) -> None:
        """Ingest SDS KoPub VDR into the database.

        Args:
            subset: Dataset split. SDS KoPub VDR currently supports only ``test``.
            query_limit: Maximum number of queries to ingest. ``None`` ingests all queries.
            min_corpus_cnt: Minimum corpus pages to ingest. Gold pages from selected queries
                are always included; additional pages are streamed in corpus order.
        """
        if self.service is None:
            raise ServiceNotSetError
        if subset != "test":
            raise ValueError("SDS KoPub VDR only has 'test' split.")  # noqa: TRY003

        qrels_df = self._load_qrels(subset)
        queries_df = self._load_queries(subset)

        selected_query_ids = self._select_queries(queries_df, qrels_df, query_limit)
        grouped_qrels = self._group_qrels(qrels_df, selected_query_ids)
        gold_corpus_ids = self._extract_gold_corpus_ids(grouped_qrels)
        additional_needed = self._calculate_additional_corpus_needed(gold_corpus_ids, min_corpus_cnt)

        ingested_image_ids, ingested_text_ids = self._ingest_corpus(subset, gold_corpus_ids, additional_needed)
        query_id_to_pk = self._ingest_queries(selected_query_ids, queries_df)
        self._ingest_qrels(grouped_qrels, query_id_to_pk, ingested_image_ids, ingested_text_ids, self.qrels_mode)

        logger.info(
            "SDS KoPub VDR ingestion complete: "
            f"{len(query_id_to_pk)} queries, {len(ingested_image_ids)} images, {len(ingested_text_ids)} text chunks"
        )

    def _load_qrels(self, subset: str) -> pd.DataFrame:
        qrels_df: pd.DataFrame = load_dataset(  # type: ignore[union-attr, assignment]
            DATASET_PATH, "qrels", streaming=False, split=subset
        ).to_pandas()
        qrels_df["query-id"] = qrels_df["query-id"].astype(str)
        qrels_df["corpus-id"] = qrels_df["corpus-id"].astype(str)
        qrels_df["score"] = qrels_df["score"].astype(int)
        return qrels_df

    def _load_queries(self, subset: str) -> pd.DataFrame:
        queries_df: pd.DataFrame = load_dataset(  # type: ignore[union-attr, assignment]
            DATASET_PATH, "queries", streaming=False, split=subset
        ).to_pandas()
        queries_df["id"] = queries_df["id"].astype(str)
        return queries_df.set_index("id")

    def _select_queries(
        self,
        queries_df: pd.DataFrame,
        qrels_df: pd.DataFrame,
        query_limit: int | None,
    ) -> list[str]:
        query_ids_with_qrels = set(qrels_df.loc[qrels_df["score"] > 0, "query-id"].tolist())
        query_ids = [query_id for query_id in queries_df.index.tolist() if query_id in query_ids_with_qrels]
        if query_limit is not None and query_limit < len(query_ids):
            rng = random.Random(RANDOM_SEED)
            return rng.sample(query_ids, query_limit)
        return query_ids

    def _group_qrels(
        self,
        qrels_df: pd.DataFrame,
        selected_query_ids: list[str],
    ) -> dict[str, list[tuple[str, int]]]:
        filtered_qrels = qrels_df[(qrels_df["query-id"].isin(selected_query_ids)) & (qrels_df["score"] > 0)]
        grouped: dict[str, list[tuple[str, int]]] = {}
        for query_id, group in filtered_qrels.groupby("query-id", sort=False):
            grouped[str(query_id)] = [(str(row["corpus-id"]), int(row["score"])) for _, row in group.iterrows()]
        return grouped

    def _extract_gold_corpus_ids(self, grouped_qrels: dict[str, list[tuple[str, int]]]) -> set[str]:
        return {corpus_id for corpus_score_pairs in grouped_qrels.values() for corpus_id, _ in corpus_score_pairs}

    def _calculate_additional_corpus_needed(self, gold_corpus_ids: set[str], min_corpus_cnt: int | None) -> int:
        if min_corpus_cnt is not None and min_corpus_cnt > len(gold_corpus_ids):
            return min_corpus_cnt - len(gold_corpus_ids)
        return 0

    def _ingest_corpus(
        self,
        subset: str,
        gold_corpus_ids: set[str],
        additional_needed: int,
    ) -> tuple[set[str], set[str]]:
        if self.service is None:
            raise ServiceNotSetError

        corpus_ds = load_dataset(DATASET_PATH, "corpus", streaming=True, split=subset)
        ingested_image_ids: set[str] = set()
        ingested_text_ids: set[str] = set()
        doc_id_to_db_id: dict[str, int | str] = {}
        page_key_to_db_id: dict[tuple[str, int], int | str] = {}
        image_batch: list[dict[str, Any]] = []

        for row in corpus_ds:
            corpus_id = str(row["id"])
            is_gold = corpus_id in gold_corpus_ids
            is_additional = (not is_gold) and additional_needed > 0
            if not is_gold and not is_additional:
                continue

            if is_additional:
                additional_needed -= 1

            doc_id = self._extract_doc_id(corpus_id)
            page_num = self._infer_page_num(corpus_id)
            img_bytes, mimetype = pil_image_to_bytes(row["image"])
            page_db_id = self._get_or_create_page(
                row, corpus_id, doc_id, page_num, img_bytes, mimetype, doc_id_to_db_id, page_key_to_db_id
            )

            text_contents = str(row.get("text") or "").strip()
            if text_contents:
                self._ingest_text_chunk(corpus_id, text_contents, page_db_id)
                ingested_text_ids.add(corpus_id)

            image_batch.append({
                "id": corpus_id,
                "contents": img_bytes,
                "mimetype": mimetype,
                "parent_page": page_db_id,
            })
            ingested_image_ids.add(corpus_id)

            if len(image_batch) >= BATCH_SIZE:
                self.service.add_image_chunks(image_batch)
                image_batch.clear()

        if image_batch:
            self.service.add_image_chunks(image_batch)

        return ingested_image_ids, ingested_text_ids

    def _extract_doc_id(self, corpus_id: str) -> str:
        return corpus_id.rsplit("_", maxsplit=1)[0]

    def _infer_page_num(self, corpus_id: str) -> int:
        suffix = corpus_id.rsplit("_", maxsplit=1)[-1]
        return int(suffix) if suffix.isdigit() else 1

    def _get_or_create_page(
        self,
        row: dict[str, Any],
        corpus_id: str,
        doc_id: str,
        page_num: int,
        img_bytes: bytes,
        mimetype: str,
        doc_id_to_db_id: dict[str, int | str],
        page_key_to_db_id: dict[tuple[str, int], int | str],
    ) -> int | str:
        if self.service is None:
            raise ServiceNotSetError

        if doc_id not in doc_id_to_db_id:
            file_id = self.service.add_files([{"type": "raw", "path": f"hf://{DATASET_PATH}/{doc_id}"}])[0]
            document_id = self.service.add_documents([
                {
                    "path": file_id,
                    "filename": doc_id,
                    "title": row.get("id"),
                    "author": None,
                    "doc_metadata": {
                        "source_dataset": DATASET_PATH,
                        "original_doc_id": doc_id,
                    },
                }
            ])[0]
            doc_id_to_db_id[doc_id] = document_id

        page_key = (doc_id, page_num)
        if page_key not in page_key_to_db_id:
            page_id = self.service.add_pages([
                {
                    "document_id": doc_id_to_db_id[doc_id],
                    "page_num": page_num,
                    "image_contents": img_bytes,
                    "mimetype": mimetype,
                    "page_metadata": {
                        "source_dataset": DATASET_PATH,
                        "corpus_id": corpus_id,
                    },
                }
            ])[0]
            page_key_to_db_id[page_key] = page_id

        return page_key_to_db_id[page_key]

    def _ingest_text_chunk(self, corpus_id: str, contents: str, page_db_id: int | str) -> None:
        if self.service is None:
            raise ServiceNotSetError
        self.service.add_chunks([{"id": corpus_id, "contents": contents, "is_table": False}])
        self.service.link_page_to_chunks(page_db_id, [corpus_id])

    def _ingest_queries(self, selected_query_ids: list[str], queries_df: pd.DataFrame) -> dict[str, str]:
        if self.service is None:
            raise ServiceNotSetError

        query_id_to_pk: dict[str, str] = {}
        queries_batch: list[dict[str, Any]] = []
        for query_id in selected_query_ids:
            query_row = queries_df.loc[query_id]
            queries_batch.append({"id": query_id, "contents": str(query_row["text"])})
            query_id_to_pk[query_id] = query_id

        if queries_batch:
            self.service.add_queries(queries_batch)

        return query_id_to_pk

    def _ingest_qrels(
        self,
        grouped_qrels: dict[str, list[tuple[str, int]]],
        query_id_to_pk: dict[str, str],
        ingested_image_ids: set[str],
        ingested_text_ids: set[str],
        qrels_mode: QrelsMode,
    ) -> None:
        if self.service is None:
            raise ServiceNotSetError

        qrels_items: list[tuple[int | str, Any]] = []
        for query_id, corpus_score_pairs in grouped_qrels.items():
            query_pk = query_id_to_pk.get(query_id)
            if query_pk is None:
                continue
            gt_expr = self._build_gt_expression(corpus_score_pairs, ingested_image_ids, ingested_text_ids, qrels_mode)
            if gt_expr is not None:
                qrels_items.append((query_pk, gt_expr))

        if qrels_items:
            self.service.add_retrieval_gt_batch(qrels_items, chunk_type=qrels_mode)

        logger.info(f"Added {len(qrels_items)} SDS KoPub VDR qrel entries")

    def _build_gt_expression(
        self,
        corpus_score_pairs: list[tuple[str, int]],
        ingested_image_ids: set[str],
        ingested_text_ids: set[str],
        qrels_mode: QrelsMode,
    ) -> RetrievalGT | None:
        if qrels_mode == "image":
            items = [
                image(corpus_id, score=score)
                for corpus_id, score in corpus_score_pairs
                if corpus_id in ingested_image_ids
            ]
        elif qrels_mode == "text":
            items = [
                text(corpus_id, score=score)
                for corpus_id, score in corpus_score_pairs
                if corpus_id in ingested_text_ids
            ]
        else:
            items = self._build_mixed_items(corpus_score_pairs, ingested_image_ids, ingested_text_ids)

        if not items:
            return None
        if len(items) == 1:
            return items[0]
        return reduce(operator.or_, items)

    def _build_mixed_items(
        self,
        corpus_score_pairs: list[tuple[str, int]],
        ingested_image_ids: set[str],
        ingested_text_ids: set[str],
    ) -> list[Any]:
        items: list[Any] = []
        for corpus_id, score in corpus_score_pairs:
            alternatives: list[TextId | ImageId] = []
            if corpus_id in ingested_image_ids:
                alternatives.append(ImageId(corpus_id, score=score))
            if corpus_id in ingested_text_ids:
                alternatives.append(TextId(corpus_id, score=score))
            if alternatives:
                items.append(or_all_mixed(alternatives))
        return items

    def embed_all(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        super().embed_all(max_concurrency=max_concurrency, batch_size=batch_size)
        if self.embedding_model is None or self.service is None:
            return
        self.service.embed_all_chunks(
            self.embedding_model.aembed_query, batch_size=batch_size, max_concurrency=max_concurrency
        )

    def embed_all_late_interaction(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        super().embed_all_late_interaction(max_concurrency=max_concurrency, batch_size=batch_size)
        if self.late_interaction_embedding_model is None or self.service is None:
            return
        self.service.embed_all_chunks_multi_vector(
            self.late_interaction_embedding_model.aembed_query,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
