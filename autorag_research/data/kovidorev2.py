"""KoViDoRe V2 Dataset Ingestor for AutoRAG-Research.

KoViDoRe V2 is a Korean visual document retrieval benchmark. The datasets
follow a multimodal BEIR structure
with separate corpus, queries, and qrels subsets:

- corpus: page images plus OCR/markdown metadata
- queries: Korean text questions with generation answers and query type metadata
- qrels: query-to-page relevance judgments with integer IDs
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
from autorag_research.orm.models import (
    ImageId,
    RetrievalGT,
    TextId,
    and_all_mixed,
    image,
    or_all_mixed,
    text,
)
from autorag_research.util import pil_image_to_bytes, to_list

logger = logging.getLogger("AutoRAG-Research")

RANDOM_SEED = 42
BATCH_SIZE = 100

KoViDoReV2DatasetName = Literal["cybersecurity", "economic", "energy", "hr"]
QrelsMode = Literal["image", "text", "mixed"]

_DATASET_PATHS: dict[str, str] = {
    "cybersecurity": "whybe-choi/kovidore-v2-cybersecurity-beir",
    "economic": "whybe-choi/kovidore-v2-economic-beir",
    "energy": "whybe-choi/kovidore-v2-energy-beir",
    "hr": "whybe-choi/kovidore-v2-hr-beir",
}


@register_ingestor(
    name="kovidorev2",
    description="KoViDoRe V2 Korean visual document retrieval benchmark",
    hf_repo="kovidorev2-dumps",
)
class KoViDoReV2Ingestor(MultiModalEmbeddingDataIngestor):
    """Ingestor for KoViDoRe V2 BEIR visual document retrieval datasets.

    KoViDoRe V2 BEIR datasets use integer ``query_id`` and ``corpus_id`` fields.
    Corpus rows include both page images and markdown
    text, so this ingestor stores image chunks and text chunks for each selected
    corpus page. Retrieval ground truth maps to image chunks by default, with
    optional text-only or mixed text/image mappings.
    """

    def __init__(
        self,
        dataset_name: KoViDoReV2DatasetName,
        qrels_mode: QrelsMode = "image",
        embedding_model: SingleVectorMultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
    ):
        """Initialize KoViDoRe V2 ingestor.

        Args:
            dataset_name: KoViDoRe V2 domain to ingest.
            qrels_mode: How qrels should map to chunks:
                - ``image``: image chunks only (default)
                - ``text``: markdown text chunks only
                - ``mixed``: either image or markdown chunk is relevant
            embedding_model: Optional single-vector multimodal embedding model.
            late_interaction_embedding_model: Optional multi-vector embedding model.
        """
        super().__init__(embedding_model, late_interaction_embedding_model)
        self.dataset_name = dataset_name
        self.qrels_mode = qrels_mode
        self.dataset_path = _DATASET_PATHS[dataset_name]

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        """KoViDoRe V2 BEIR datasets use integer query and corpus IDs."""
        return "bigint"

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
    ) -> None:
        """Ingest KoViDoRe V2 dataset into the database.

        Args:
            subset: Dataset split. KoViDoRe V2 currently supports only ``test``.
            query_limit: Maximum number of queries to ingest. ``None`` ingests all queries.
            min_corpus_cnt: Minimum corpus pages to ingest. Gold corpus pages for selected
                queries are always included; additional pages are streamed in dataset order
                until this minimum is reached.
        """
        if self.service is None:
            raise ServiceNotSetError
        if subset != "test":
            raise ValueError("KoViDoRe V2 datasets only have 'test' split.")  # noqa: TRY003

        qrels_df = self._load_qrels(subset)
        queries_df = self._load_queries(subset)

        selected_query_ids, query_types_map = self._select_queries(queries_df, qrels_df, query_limit)
        grouped_qrels = self._group_qrels(qrels_df, selected_query_ids)
        gold_corpus_ids = self._extract_gold_corpus_ids(grouped_qrels)
        additional_needed = self._calculate_additional_corpus_needed(gold_corpus_ids, min_corpus_cnt)

        ingested_image_ids, ingested_text_ids = self._ingest_corpus(subset, gold_corpus_ids, additional_needed)
        query_id_to_pk = self._ingest_queries(selected_query_ids, queries_df)
        self._ingest_qrels(
            grouped_qrels,
            query_id_to_pk,
            query_types_map,
            ingested_image_ids,
            ingested_text_ids,
            self.qrels_mode,
        )

        logger.info(
            f"KoViDoRe V2 ingestion complete for '{self.dataset_name}': "
            f"{len(query_id_to_pk)} queries, {len(ingested_image_ids)} images, {len(ingested_text_ids)} text chunks"
        )

    def _load_qrels(self, subset: str) -> pd.DataFrame:
        qrels_df: pd.DataFrame = load_dataset(  # type: ignore[union-attr, assignment]
            self.dataset_path, "qrels", streaming=False, split=subset
        ).to_pandas()
        qrels_df["query_id"] = qrels_df["query_id"].astype(int)
        qrels_df["corpus_id"] = qrels_df["corpus_id"].astype(int)
        qrels_df["score"] = qrels_df["score"].astype(int)
        return qrels_df

    def _load_queries(self, subset: str) -> pd.DataFrame:
        queries_df: pd.DataFrame = load_dataset(  # type: ignore[union-attr, assignment]
            self.dataset_path, "queries", streaming=False, split=subset
        ).to_pandas()
        queries_df["query_id"] = queries_df["query_id"].astype(int)
        return queries_df.set_index("query_id")

    def _select_queries(
        self,
        queries_df: pd.DataFrame,
        qrels_df: pd.DataFrame,
        query_limit: int | None,
    ) -> tuple[list[int], dict[int, list[str]]]:
        query_ids_with_qrels = set(qrels_df.loc[qrels_df["score"] > 0, "query_id"].tolist())
        available_query_ids = [query_id for query_id in queries_df.index.tolist() if query_id in query_ids_with_qrels]

        if query_limit is not None and query_limit < len(available_query_ids):
            rng = random.Random(RANDOM_SEED)
            selected_query_ids = rng.sample(available_query_ids, query_limit)
        else:
            selected_query_ids = available_query_ids

        query_types_map: dict[int, list[str]] = {}
        for query_id in selected_query_ids:
            query_types = queries_df.at[query_id, "query_types"] if "query_types" in queries_df.columns else []
            query_types_list = to_list(query_types)
            if not isinstance(query_types_list, list):
                query_types_list = []
            query_types_map[query_id] = [str(query_type) for query_type in query_types_list]

        logger.info(f"Selected {len(selected_query_ids)} KoViDoRe V2 queries from {len(available_query_ids)} available")
        return selected_query_ids, query_types_map

    def _group_qrels(
        self,
        qrels_df: pd.DataFrame,
        selected_query_ids: list[int],
    ) -> dict[int, list[tuple[int, int]]]:
        filtered_qrels = qrels_df[(qrels_df["query_id"].isin(selected_query_ids)) & (qrels_df["score"] > 0)]
        grouped: dict[int, list[tuple[int, int]]] = {}
        for query_id, group in filtered_qrels.groupby("query_id", sort=False):
            grouped[int(query_id)] = [(int(row["corpus_id"]), int(row["score"])) for _, row in group.iterrows()]
        return grouped

    def _extract_gold_corpus_ids(self, grouped_qrels: dict[int, list[tuple[int, int]]]) -> set[int]:
        return {corpus_id for corpus_score_pairs in grouped_qrels.values() for corpus_id, _ in corpus_score_pairs}

    def _calculate_additional_corpus_needed(self, gold_corpus_ids: set[int], min_corpus_cnt: int | None) -> int:
        if min_corpus_cnt is not None and min_corpus_cnt > len(gold_corpus_ids):
            return min_corpus_cnt - len(gold_corpus_ids)
        return 0

    def _ingest_corpus(
        self,
        subset: str,
        gold_corpus_ids: set[int],
        additional_needed: int,
    ) -> tuple[set[int], set[int]]:
        if self.service is None:
            raise ServiceNotSetError

        corpus_ds = load_dataset(self.dataset_path, "corpus", streaming=True, split=subset)
        ingested_image_ids: set[int] = set()
        ingested_text_ids: set[int] = set()
        doc_id_to_db_id: dict[str, int | str] = {}
        page_key_to_db_id: dict[tuple[str, int], int | str] = {}
        image_batch: list[dict[str, Any]] = []

        for row in corpus_ds:
            corpus_id = int(row["corpus_id"])
            is_gold = corpus_id in gold_corpus_ids
            is_additional = (not is_gold) and additional_needed > 0
            if not is_gold and not is_additional:
                continue

            if is_additional:
                additional_needed -= 1

            doc_id = str(row.get("doc_id") or corpus_id)
            page_num = int(row.get("page_number_in_doc") or 0)
            img_bytes, mimetype = pil_image_to_bytes(row["image"])
            page_db_id = self._get_or_create_page(
                row, doc_id, page_num, img_bytes, mimetype, doc_id_to_db_id, page_key_to_db_id
            )

            markdown = str(row.get("markdown") or "")
            if markdown.strip():
                self._ingest_text_chunk(corpus_id, markdown, page_db_id)
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

    def _get_or_create_page(
        self,
        row: dict[str, Any],
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
            file_id = self.service.add_files([{"type": "raw", "path": f"hf://{self.dataset_path}/{doc_id}"}])[0]
            document_id = self.service.add_documents([
                {
                    "path": file_id,
                    "filename": doc_id,
                    "title": doc_id,
                    "author": None,
                    "doc_metadata": {
                        "source_dataset": self.dataset_path,
                        "original_doc_id": doc_id,
                    },
                }
            ])[0]
            doc_id_to_db_id[doc_id] = document_id

        page_key = (doc_id, page_num)
        if page_key not in page_key_to_db_id:
            page_metadata = {
                "source_dataset": self.dataset_path,
                "corpus_id": int(row["corpus_id"]),
                "modality": row.get("modality"),
                "elements": row.get("elements"),
            }
            page_id = self.service.add_pages([
                {
                    "document_id": doc_id_to_db_id[doc_id],
                    "page_num": page_num,
                    "image_contents": img_bytes,
                    "mimetype": mimetype,
                    "page_metadata": page_metadata,
                }
            ])[0]
            page_key_to_db_id[page_key] = page_id

        return page_key_to_db_id[page_key]

    def _ingest_text_chunk(self, corpus_id: int, markdown: str, page_db_id: int | str) -> None:
        if self.service is None:
            raise ServiceNotSetError
        self.service.add_chunks([{"id": corpus_id, "contents": markdown, "is_table": False}])
        self.service.link_page_to_chunks(page_db_id, [corpus_id])

    def _ingest_queries(self, selected_query_ids: list[int], queries_df: pd.DataFrame) -> dict[int, int]:
        if self.service is None:
            raise ServiceNotSetError

        query_id_to_pk: dict[int, int] = {}
        queries_batch: list[dict[str, Any]] = []

        for query_id in selected_query_ids:
            query_row = queries_df.loc[query_id]
            answer = query_row.get("answer")
            answers = [str(answer)] if answer is not None and str(answer).strip() else None
            query_text = str(query_row["query"])

            queries_batch.append({"id": query_id, "contents": query_text, "generation_gt": answers})
            query_id_to_pk[query_id] = query_id

        if queries_batch:
            self.service.add_queries(queries_batch)

        return query_id_to_pk

    def _ingest_qrels(
        self,
        grouped_qrels: dict[int, list[tuple[int, int]]],
        query_id_to_pk: dict[int, int],
        query_types_map: dict[int, list[str]],
        ingested_image_ids: set[int],
        ingested_text_ids: set[int],
        qrels_mode: QrelsMode,
    ) -> None:
        if self.service is None:
            raise ServiceNotSetError

        qrels_items: list[tuple[int | str, Any]] = []
        for query_id, corpus_score_pairs in grouped_qrels.items():
            query_pk = query_id_to_pk.get(query_id)
            if query_pk is None:
                continue

            is_multi_hop = "multi-hop" in query_types_map.get(query_id, [])
            gt_expr = self._build_gt_expression(
                corpus_score_pairs, ingested_image_ids, ingested_text_ids, qrels_mode, is_multi_hop
            )
            if gt_expr is not None:
                qrels_items.append((query_pk, gt_expr))

        if qrels_items:
            self.service.add_retrieval_gt_batch(qrels_items, chunk_type=qrels_mode)

        logger.info(f"Added {len(qrels_items)} KoViDoRe V2 qrel entries")

    def _build_gt_expression(
        self,
        corpus_score_pairs: list[tuple[int, int]],
        ingested_image_ids: set[int],
        ingested_text_ids: set[int],
        qrels_mode: QrelsMode,
        is_multi_hop: bool,
    ) -> RetrievalGT | None:
        if qrels_mode == "image":
            image_items = [
                image(corpus_id, score=score)
                for corpus_id, score in corpus_score_pairs
                if corpus_id in ingested_image_ids
            ]
            return self._combine_items(image_items, is_multi_hop)

        if qrels_mode == "text":
            text_items = [
                text(corpus_id, score=score)
                for corpus_id, score in corpus_score_pairs
                if corpus_id in ingested_text_ids
            ]
            return self._combine_items(text_items, is_multi_hop)

        return self._build_mixed_gt_expression(corpus_score_pairs, ingested_image_ids, ingested_text_ids, is_multi_hop)

    def _build_mixed_gt_expression(
        self,
        corpus_score_pairs: list[tuple[int, int]],
        ingested_image_ids: set[int],
        ingested_text_ids: set[int],
        is_multi_hop: bool,
    ) -> RetrievalGT | None:
        alternatives_by_corpus = [
            self._mixed_alternatives(corpus_id, score, ingested_image_ids, ingested_text_ids)
            for corpus_id, score in corpus_score_pairs
        ]
        alternatives_by_corpus = [alternatives for alternatives in alternatives_by_corpus if alternatives]
        if not alternatives_by_corpus:
            return None

        if is_multi_hop:
            groups = [or_all_mixed(alternatives) for alternatives in alternatives_by_corpus]
            return and_all_mixed(groups)  # type: ignore[arg-type, return-value]

        items = [item for alternatives in alternatives_by_corpus for item in alternatives]
        return or_all_mixed(items)

    def _mixed_alternatives(
        self,
        corpus_id: int,
        score: int,
        ingested_image_ids: set[int],
        ingested_text_ids: set[int],
    ) -> list[TextId | ImageId]:
        alternatives: list[TextId | ImageId] = []
        if corpus_id in ingested_image_ids:
            alternatives.append(ImageId(corpus_id, score=score))
        if corpus_id in ingested_text_ids:
            alternatives.append(TextId(corpus_id, score=score))
        return alternatives

    def _combine_items(self, items: list[Any], is_multi_hop: bool) -> RetrievalGT | None:
        if not items:
            return None
        if len(items) == 1:
            return items[0]
        op = operator.and_ if is_multi_hop else operator.or_
        return reduce(op, items)

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
