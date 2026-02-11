from __future__ import annotations

import logging
import operator
import random
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from datasets import load_dataset

from autorag_research.data.base import MultiModalEmbeddingDataIngestor
from autorag_research.data.registry import register_ingestor
from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding
from autorag_research.exceptions import ServiceNotSetError
from autorag_research.orm.models import (
    ImageId,
    OrGroup,
    RetrievalGT,
    TextId,
    and_all_mixed,
    image,
    or_all_mixed,
    text,
)
from autorag_research.util import pil_image_to_bytes, to_list

if TYPE_CHECKING:
    from autorag_research.embeddings.base import SingleVectorMultiModalEmbedding

logger = logging.getLogger("AutoRAG-Research")

RANDOM_SEED = 42

# Type alias for qrels mapping mode (matches chunk_type in add_retrieval_gt)
QrelsMode = Literal["image", "text", "mixed"]


VIDOREV3_CONFIGS = Literal[
    "hr",
    "finance_en",
    "industrial",
    "pharmaceuticals",
    "computer_science",
    "energy",
    "physics",
    "finance_fr",
]


@register_ingestor(
    name="vidorev3",
    description="ViDoRe V3 visual document retrieval benchmark",
    hf_repo="vidorev3-dumps",
)
class ViDoReV3Ingestor(MultiModalEmbeddingDataIngestor):
    def __init__(
        self,
        config_name: VIDOREV3_CONFIGS,
        qrels_mode: QrelsMode = "image",
        embedding_model: SingleVectorMultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
    ):
        """
        ViDoReV3 dataset ingestor.

        Args:
            config_name: config names for different ViDoReV3 datasets.
            qrels_mode: How to map qrels to chunks. Options:
                - "image" (default): Map corpus_id to ImageChunk only.
                - "text": Map corpus_id to text Chunk only.
                - "mixed": Map to both text Chunk and ImageChunk (OR relation).
            embedding_model: Embedding model for single-vector embeddings.
            late_interaction_embedding_model: Embedding model for multi-vector embeddings.
        """
        super().__init__(embedding_model, late_interaction_embedding_model)
        self.config_name = config_name
        self.qrels_mode = qrels_mode

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        return "bigint"

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
    ) -> None:
        """Ingest ViDoReV3 dataset into the database.

        Args:
            subset: Dataset split to use. Only "test" is supported for ViDoReV3.
            query_limit: Maximum number of queries to ingest. If None, ingest all.
            min_corpus_cnt: Minimum number of corpus items to ingest. If more than
                gold corpus items are needed, additional items will be sampled.

        """
        if self.service is None:
            raise ServiceNotSetError
        if subset != "test":
            raise ValueError("ViDoReV3 datasets only have 'test' split.")  # noqa: TRY003

        dataset_path = f"vidore/vidore_v3_{self.config_name}"

        qrels_df = self._load_qrels(dataset_path, subset)
        queries_df = self._load_queries(dataset_path, subset)
        docs_metadata = self._load_documents_metadata(dataset_path, subset)
        corpus_mapping = self._load_corpus_mapping(dataset_path, subset)

        selected_query_ids, query_types_map = self._select_queries(queries_df, query_limit)
        gold_corpus_ids = self._extract_gold_corpus_ids(qrels_df, selected_query_ids)
        additional_needed = self._calculate_additional_corpus_needed(gold_corpus_ids, min_corpus_cnt)

        all_corpus_ids = set(corpus_mapping.index)
        corpus_ids_to_ingest = self._determine_corpus_ids_to_ingest(all_corpus_ids, gold_corpus_ids, additional_needed)

        required_doc_ids = self._get_required_doc_ids(corpus_mapping, corpus_ids_to_ingest)
        filtered_docs_metadata = docs_metadata[docs_metadata.index.isin(required_doc_ids)]

        doc_id_to_db_id = self._ingest_document_hierarchy(filtered_docs_metadata)

        ingested_corpus_ids = self._ingest_corpus(dataset_path, corpus_ids_to_ingest, doc_id_to_db_id)

        self._ingest_queries(selected_query_ids, queries_df)
        self._ingest_qrels(qrels_df, selected_query_ids, query_types_map, ingested_corpus_ids, self.qrels_mode)

        logger.info(
            f"ViDoReV3 ingestion complete for '{self.config_name}': "
            f"{len(selected_query_ids)} queries, {len(ingested_corpus_ids)} corpus items"
        )

    def _load_qrels(self, dataset_path: str, subset: str) -> pd.DataFrame:
        qrels_df: pd.DataFrame = load_dataset(dataset_path, "qrels", streaming=False, split=subset).to_pandas()  # type: ignore[union-attr, assignment]
        qrels_df["query_id"] = qrels_df["query_id"].astype(int)
        qrels_df["corpus_id"] = qrels_df["corpus_id"].astype(int)
        return qrels_df

    def _load_queries(self, dataset_path: str, subset: str) -> pd.DataFrame:
        queries_df: pd.DataFrame = load_dataset(dataset_path, "queries", streaming=False, split=subset).to_pandas()  # type: ignore[union-attr, assignment]
        queries_df["query_id"] = queries_df["query_id"].astype(int)
        return queries_df.set_index("query_id")

    def _load_documents_metadata(self, dataset_path: str, subset: str) -> pd.DataFrame:
        docs_df: pd.DataFrame = load_dataset(  # type: ignore[union-attr, assignment]
            dataset_path, "documents_metadata", streaming=False, split=subset
        ).to_pandas()
        return docs_df.set_index("doc_id")

    def _load_corpus_mapping(self, dataset_path: str, subset: str) -> pd.DataFrame:
        corpus_ds = load_dataset(dataset_path, "corpus", streaming=False, split=subset)
        corpus_df: pd.DataFrame = corpus_ds.select_columns(["corpus_id", "doc_id"]).to_pandas()  # type: ignore[union-attr, assignment]
        corpus_df["corpus_id"] = corpus_df["corpus_id"].astype(int)
        return corpus_df.set_index("corpus_id")

    def _select_queries(
        self,
        queries_df: pd.DataFrame,
        query_limit: int | None,
    ) -> tuple[list[int], dict[int, list[str]]]:
        if query_limit is not None and query_limit < len(queries_df):
            rng = random.Random(RANDOM_SEED)
            selected_query_ids = rng.sample(list(queries_df.index), query_limit)
        else:
            selected_query_ids = list(queries_df.index)

        query_types_map: dict[int, list[str]] = {}
        for query_id in selected_query_ids:
            query_types = queries_df.at[query_id, "query_types"]
            query_types_map[query_id] = query_types if isinstance(query_types, list) else []

        return selected_query_ids, query_types_map

    def _extract_gold_corpus_ids(self, qrels_df: pd.DataFrame, query_ids: list[int]) -> set[int]:
        filtered_qrels = qrels_df[qrels_df["query_id"].isin(query_ids)]
        return set(filtered_qrels["corpus_id"].tolist())

    def _calculate_additional_corpus_needed(self, gold_corpus_ids: set[int], min_corpus_cnt: int | None) -> int:
        if min_corpus_cnt is not None and min_corpus_cnt > len(gold_corpus_ids):
            return min_corpus_cnt - len(gold_corpus_ids)
        return 0

    def _determine_corpus_ids_to_ingest(
        self,
        all_corpus_ids: set[int],
        gold_ids: set[int],
        additional_needed: int,
    ) -> set[int]:
        corpus_ids_to_ingest = set(gold_ids)

        if additional_needed > 0:
            non_gold_ids = list(all_corpus_ids - gold_ids)
            rng = random.Random(RANDOM_SEED)
            rng.shuffle(non_gold_ids)
            corpus_ids_to_ingest.update(non_gold_ids[:additional_needed])

        return corpus_ids_to_ingest

    def _get_required_doc_ids(self, corpus_mapping: pd.DataFrame, corpus_ids: set[int]) -> set[str]:
        filtered_mapping = corpus_mapping[corpus_mapping.index.isin(corpus_ids)]
        return {str(doc_id) for doc_id in filtered_mapping["doc_id"].unique()}

    def _ingest_document_hierarchy(self, docs_metadata: pd.DataFrame) -> dict[str, int]:
        if self.service is None:
            raise ServiceNotSetError

        files_data = [{"type": "raw", "path": row.get("url", "") or ""} for _, row in docs_metadata.iterrows()]
        file_ids = self.service.add_files(files_data)

        docs_data = []
        for (doc_id, row), file_db_id in zip(docs_metadata.iterrows(), file_ids, strict=True):
            file_name = row.get("file_name", "")

            docs_data.append({
                "path": file_db_id,
                "filename": file_name,
                "title": None,
                "author": None,
                "doc_metadata": {
                    "url": row.get("url", ""),
                    "doc_type": row.get("doc_type"),
                    "doc_language": row.get("doc_language"),
                    "doc_year": row.get("doc_year"),
                    "visual_types": to_list(row.get("visual_types", [])),
                    "page_count": row.get("page_number"),
                    "license": row.get("license"),
                    "original_doc_id": str(doc_id),
                },
            })

        doc_db_ids = self.service.add_documents(docs_data)

        doc_id_to_db_id = {
            str(doc_id): db_id for (doc_id, _), db_id in zip(docs_metadata.iterrows(), doc_db_ids, strict=True)
        }

        return doc_id_to_db_id

    def _create_text_chunk(
        self,
        corpus_id: int,
        markdown: str,
    ) -> int:
        """Create a single text chunk from markdown content (whole page = one chunk).

        Args:
            corpus_id: The corpus ID to use as chunk ID
            markdown: The markdown text content

        Returns:
            The created chunk DB ID (same as corpus_id)
        """
        if self.service is None:
            raise ServiceNotSetError

        chunks_data = [{"id": corpus_id, "contents": markdown, "is_table": False}]
        self.service.add_chunks(chunks_data)
        return corpus_id

    def _ingest_corpus(
        self,
        dataset_path: str,
        corpus_ids_to_ingest: set[int],
        doc_id_to_db_id: dict[str, int],
    ) -> set[int]:
        """Ingest corpus items and return ingested IDs.

        Note: corpus_id is used as both ImageChunk ID and text Chunk ID.

        Returns:
            Set of successfully ingested corpus IDs
        """
        if self.service is None:
            raise ServiceNotSetError

        corpus_ds = load_dataset(dataset_path, "corpus", streaming=True, split="test")
        ingested_ids: set[int] = set()

        page_key_to_db_id: dict[tuple[str, int], int | str] = {}

        for row in corpus_ds:
            corpus_id = int(row["corpus_id"])

            if corpus_id not in corpus_ids_to_ingest:
                continue

            doc_id = str(row["doc_id"])
            page_num = int(row["page_number_in_doc"])
            markdown = row.get("markdown", "") or ""
            pil_image = row["image"]

            img_bytes, mimetype = pil_image_to_bytes(pil_image)

            doc_db_id = doc_id_to_db_id.get(doc_id)

            page_key = (doc_id, page_num)
            if page_key not in page_key_to_db_id:
                page_ids = self.service.add_pages([
                    {
                        "document_id": doc_db_id,
                        "page_num": page_num,
                        "image_contents": img_bytes,
                        "mimetype": mimetype,
                    }
                ])
                if page_ids:
                    page_key_to_db_id[page_key] = page_ids[0]

            page_db_id = page_key_to_db_id.get(page_key)

            if markdown.strip() and page_db_id is not None:
                chunk_id = self._create_text_chunk(corpus_id, markdown)
                self.service.link_page_to_chunks(page_db_id, [chunk_id])

            self.service.add_image_chunks([
                {"id": corpus_id, "contents": img_bytes, "mimetype": mimetype, "parent_page": page_db_id}
            ])

            ingested_ids.add(corpus_id)

        return ingested_ids

    def _ingest_queries(
        self,
        selected_query_ids: list[int],
        queries_df: pd.DataFrame,
    ) -> None:
        if self.service is None:
            raise ServiceNotSetError

        queries_batch: list[dict[str, Any]] = []

        for query_id in selected_query_ids:
            query_row = queries_df.loc[query_id]
            query_text = str(query_row["query"])
            answer = query_row.get("answer")

            query_data: dict[str, Any] = {
                "id": query_id,
                "contents": query_text,
                "generation_gt": [answer],
            }

            queries_batch.append(query_data)

        if queries_batch:
            self.service.add_queries(queries_batch)

    def _ingest_qrels(
        self,
        qrels_df: pd.DataFrame,
        selected_query_ids: list[int],
        query_types_map: dict[int, list[str]],
        ingested_corpus_ids: set[int],
        qrels_mode: QrelsMode,
    ) -> None:
        """Ingest query relevance relations based on qrels_mode.

        Note: corpus_id is used as both ImageChunk ID and text Chunk ID.
        Relations with score=0 are skipped (not relevant).

        Args:
            qrels_df: DataFrame with query_id, corpus_id, and score columns
            selected_query_ids: List of query IDs to process
            query_types_map: Mapping from query_id to list of query types
            ingested_corpus_ids: Set of successfully ingested corpus IDs
            qrels_mode: How to map qrels to chunks:
                - "image": Map corpus_id to ImageChunk only
                - "text": Map corpus_id to text Chunk only
                - "mixed": Map to both text Chunk and ImageChunk (OR relation)
        """
        if self.service is None:
            raise ServiceNotSetError

        # Filter to selected queries and exclude score=0 (not relevant)
        filtered_qrels = qrels_df[(qrels_df["query_id"].isin(selected_query_ids)) & (qrels_df["score"] > 0)]

        # Group by query_id and collect (corpus_id, score) pairs
        grouped = filtered_qrels.groupby("query_id").apply(
            lambda x: list(zip(x["corpus_id"], x["score"], strict=True)), include_groups=False
        )

        for query_id, corpus_score_pairs in grouped.items():
            # Filter to only ingested corpus IDs, keeping scores
            valid_pairs = [(cid, score) for cid, score in corpus_score_pairs if cid in ingested_corpus_ids]

            if not valid_pairs:
                continue

            query_types = query_types_map.get(query_id, [])
            is_multi_hop = "multi-hop" in query_types

            gt_expr = self._build_gt_expression_with_scores(valid_pairs, qrels_mode, is_multi_hop)

            if gt_expr is None:
                continue

            self.service.add_retrieval_gt(
                query_id,
                gt_expr,
                chunk_type=qrels_mode,
            )

    def _build_gt_expression_with_scores(
        self,
        corpus_score_pairs: list[tuple[int, int]],
        qrels_mode: QrelsMode,
        is_multi_hop: bool,
    ) -> RetrievalGT | None:
        """Build ground truth expression with graded relevance scores.

        Note: corpus_id is used as both ImageChunk ID and text Chunk ID.

        Args:
            corpus_score_pairs: List of (corpus_id, score) tuples
            qrels_mode: How to map qrels to chunks
            is_multi_hop: Whether this is a multi-hop query (AND semantics)

        Returns:
            Ground truth expression, or None if no valid chunks found
        """
        if qrels_mode == "image":
            return self._build_image_only_gt_with_scores(corpus_score_pairs, is_multi_hop)
        elif qrels_mode == "text":
            return self._build_text_only_gt_with_scores(corpus_score_pairs, is_multi_hop)
        else:  # mixed
            return self._build_both_gt_with_scores(corpus_score_pairs, is_multi_hop)

    def _build_image_only_gt_with_scores(
        self,
        corpus_score_pairs: list[tuple[int, int]],
        is_multi_hop: bool,
    ) -> RetrievalGT:
        """Build GT expression for image-only mode with graded relevance scores."""
        items = [image(corpus_id, score=score) for corpus_id, score in corpus_score_pairs]
        if is_multi_hop:
            return reduce(operator.and_, items)
        return reduce(operator.or_, items)

    def _build_text_only_gt_with_scores(
        self,
        corpus_score_pairs: list[tuple[int, int]],
        is_multi_hop: bool,
    ) -> RetrievalGT:
        """Build GT expression for text-only mode with graded relevance scores.

        Note: corpus_id is used as text Chunk ID.
        """
        items = [text(corpus_id, score=score) for corpus_id, score in corpus_score_pairs]
        if is_multi_hop:
            return reduce(operator.and_, items)
        return reduce(operator.or_, items)

    def _build_both_gt_with_scores(
        self,
        corpus_score_pairs: list[tuple[int, int]],
        is_multi_hop: bool,
    ) -> RetrievalGT:
        """Build GT expression for both mode (text and image chunks) with graded relevance scores.

        Note: corpus_id is used as both ImageChunk ID and text Chunk ID.
        """
        if is_multi_hop:
            # Multi-hop: (TextChunk OR ImageChunk) AND (TextChunk OR ImageChunk) ...
            and_groups: list[OrGroup | TextId | ImageId] = [
                or_all_mixed([ImageId(corpus_id, score=score), TextId(corpus_id, score=score)])
                for corpus_id, score in corpus_score_pairs
            ]
            return and_all_mixed(and_groups)  # type: ignore[return-value, arg-type]
        else:
            # Non-multi-hop: all text chunks and all image chunks are OR alternatives
            items: list[TextId | ImageId] = [ImageId(c_id, score=score) for c_id, score in corpus_score_pairs] + [
                TextId(c_id, score=score) for c_id, score in corpus_score_pairs
            ]
            return or_all_mixed(items)

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
