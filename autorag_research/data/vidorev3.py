import logging
import random
from typing import Any, Literal

import pandas as pd
from datasets import load_dataset
from hydra.utils import instantiate
from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import Document
from omegaconf import DictConfig

from autorag_research.data.base import MultiModalEmbeddingDataIngestor
from autorag_research.data.registry import register_ingestor
from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding
from autorag_research.exceptions import ServiceNotSetError
from autorag_research.orm.models import (
    ImageId,
    OrGroup,
    RetrievalGT,
    TextId,
    and_all,
    and_all_mixed,
    image,
    or_all,
    or_all_mixed,
    text,
)
from autorag_research.orm.models.retrieval_gt import _IntWrapper
from autorag_research.util import pil_image_to_bytes, to_list

logger = logging.getLogger("AutoRAG-Research")

RANDOM_SEED = 42
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128

# Type alias for qrels mapping mode
QrelsMode = Literal["image-only", "text-only", "both"]


def _resolve_chunker(chunker: NodeParser | DictConfig | dict | None) -> NodeParser:
    """Resolve chunker from config or use default.

    Args:
        chunker: A NodeParser instance, a Hydra DictConfig with _target_, a dict with _target_, or None.

    Returns:
        A NodeParser instance ready to use for chunking.
    """
    # Import here to avoid linter removing unused import at module level
    from llama_index.core.node_parser import SentenceSplitter

    if chunker is None:
        return SentenceSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )
    if isinstance(chunker, dict):
        chunker = DictConfig(chunker)
    if isinstance(chunker, DictConfig):
        return instantiate(chunker)
    return chunker


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


def chunk_markdown(markdown: str, chunker: NodeParser) -> list[str]:
    """Chunk markdown text using the provided NodeParser.

    Args:
        markdown: The markdown text to chunk.
        chunker: A LlamaIndex NodeParser instance for splitting the text.

    Returns:
        A list of chunk strings.
    """
    if not markdown or not markdown.strip():
        return []

    nodes = chunker.get_nodes_from_documents([Document(text=markdown)])
    return [node.get_content() for node in nodes if node.get_content().strip()]


@register_ingestor(
    name="vidorev3",
    description="ViDoRe V3 visual document retrieval benchmark",
)
class ViDoReV3Ingestor(MultiModalEmbeddingDataIngestor):
    def __init__(
        self,
        config_name: VIDOREV3_CONFIGS,
        chunker: NodeParser | DictConfig | dict | None = None,
        qrels_mode: QrelsMode = "image-only",
        embedding_model: MultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
    ):
        """
        ViDoReV3 dataset ingestor.

        Args:
            config_name: config names for different ViDoReV3 datasets.
            chunker: NodeParser instance, Hydra DictConfig with _target_, or None.
                If None, does not use chunker (use full captions as single chunks).
            qrels_mode: How to map qrels to chunks. Options:
                - "image-only" (default): Map corpus_id to ImageChunk only.
                - "text-only": Map corpus_id to text Chunk(s) only.
                - "both": Map to both text Chunks and ImageChunk (OR relation).
            embedding_model: Embedding model for single-vector embeddings.
            late_interaction_embedding_model: Embedding model for multi-vector embeddings.
        """
        super().__init__(embedding_model, late_interaction_embedding_model)
        self.config_name = config_name
        self.chunker = _resolve_chunker(chunker)
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

        ingested_corpus_ids, corpus_to_chunks = self._ingest_corpus(dataset_path, corpus_ids_to_ingest, doc_id_to_db_id)

        self._ingest_queries(selected_query_ids, queries_df)
        self._ingest_qrels(
            qrels_df, selected_query_ids, query_types_map, ingested_corpus_ids, corpus_to_chunks, self.qrels_mode
        )

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

    def _create_text_chunks(
        self,
        markdown: str,
        caption_db_id: int,
        chunker: NodeParser,
    ) -> list[int | str]:
        """Create text chunks from markdown content.

        Args:
            markdown: The markdown text to chunk
            caption_db_id: The parent caption DB ID
            chunker: NodeParser for chunking

        Returns:
            List of created chunk DB IDs
        """
        if self.service is None:
            raise ServiceNotSetError

        chunk_texts = chunk_markdown(markdown, chunker)
        if chunk_texts:
            chunks_data = [
                {"parent_caption": caption_db_id, "contents": chunk_text, "is_table": False}
                for chunk_text in chunk_texts
            ]
            return self.service.add_chunks(chunks_data)
        return []

    def _ingest_corpus(
        self,
        dataset_path: str,
        corpus_ids_to_ingest: set[int],
        doc_id_to_db_id: dict[str, int],
    ) -> tuple[set[int], dict[int, list[int | str]]]:
        """Ingest corpus items and return ingested IDs and corpus-to-chunk mapping.

        Returns:
            A tuple of (ingested_corpus_ids, corpus_to_chunks) where:
            - ingested_corpus_ids: Set of successfully ingested corpus IDs
            - corpus_to_chunks: Mapping from corpus_id to list of text chunk DB IDs
        """
        if self.service is None:
            raise ServiceNotSetError

        corpus_ds = load_dataset(dataset_path, "corpus", streaming=True, split="test")
        ingested_ids: set[int] = set()
        corpus_to_chunks: dict[int, list[int | str]] = {}

        page_key_to_db_id: dict[tuple[str, int], int] = {}

        for row in corpus_ds:  # type: ignore[union-attr]
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
                caption_ids = self.service.add_captions([{"page_id": page_db_id, "contents": markdown}])
                caption_db_id = caption_ids[0] if caption_ids else None

                if caption_db_id is not None:
                    chunk_ids = self._create_text_chunks(markdown, caption_db_id, self.chunker)
                    if chunk_ids:
                        corpus_to_chunks[corpus_id] = chunk_ids

            self.service.add_image_chunks([
                {"id": corpus_id, "contents": img_bytes, "mimetype": mimetype, "parent_page": page_db_id}
            ])

            ingested_ids.add(corpus_id)

        return ingested_ids, corpus_to_chunks

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
        corpus_to_chunks: dict[int, list[int | str]],
        qrels_mode: QrelsMode,
    ) -> None:
        """Ingest query relevance relations based on qrels_mode.

        Args:
            qrels_df: DataFrame with query_id and corpus_id columns
            selected_query_ids: List of query IDs to process
            query_types_map: Mapping from query_id to list of query types
            ingested_corpus_ids: Set of successfully ingested corpus IDs
            corpus_to_chunks: Mapping from corpus_id to list of text chunk DB IDs
            qrels_mode: How to map qrels to chunks:
                - "image-only": Map corpus_id to ImageChunk only
                - "text-only": Map corpus_id to text Chunk(s) only
                - "both": Map to both text Chunks and ImageChunk (OR relation)
        """
        if self.service is None:
            raise ServiceNotSetError

        filtered_qrels = qrels_df[qrels_df["query_id"].isin(selected_query_ids)]
        grouped = filtered_qrels.groupby("query_id")["corpus_id"].apply(list)

        for query_id, corpus_ids in grouped.items():
            valid_corpus_ids = [cid for cid in corpus_ids if cid in ingested_corpus_ids]

            if not valid_corpus_ids:
                continue

            query_types = query_types_map.get(query_id, [])
            is_multi_hop = "multi-hop" in query_types

            gt_expr = self._build_gt_expression(valid_corpus_ids, corpus_to_chunks, qrels_mode, is_multi_hop)

            if gt_expr is None:
                continue

            # Determine chunk_type based on qrels_mode
            chunk_type = "mixed" if qrels_mode == "both" else ("image" if qrels_mode == "image-only" else "text")

            self.service.add_retrieval_gt(
                query_id,  # type: ignore[arg-type]
                gt_expr,
                chunk_type=chunk_type,
            )

    def _build_gt_expression(
        self,
        corpus_ids: list[int],
        corpus_to_chunks: dict[int, list[int | str]],
        qrels_mode: QrelsMode,
        is_multi_hop: bool,
    ) -> RetrievalGT | None:
        """Build ground truth expression based on qrels_mode and multi-hop status.

        Args:
            corpus_ids: List of valid corpus IDs for the query
            corpus_to_chunks: Mapping from corpus_id to list of text chunk DB IDs
            qrels_mode: How to map qrels to chunks
            is_multi_hop: Whether this is a multi-hop query (AND semantics)

        Returns:
            Ground truth expression, or None if no valid chunks found
        """
        if qrels_mode == "image-only":
            return self._build_image_only_gt(corpus_ids, is_multi_hop)
        elif qrels_mode == "text-only":
            return self._build_text_only_gt(corpus_ids, corpus_to_chunks, is_multi_hop)
        else:  # both
            return self._build_both_gt(corpus_ids, corpus_to_chunks, is_multi_hop)

    def _build_image_only_gt(
        self,
        corpus_ids: list[int],
        is_multi_hop: bool,
    ) -> RetrievalGT:
        """Build GT expression for image-only mode."""
        ids: list[int | str] = list(corpus_ids)  # ty: ignore[invalid-assignment]
        return and_all(ids, image) if is_multi_hop else or_all(ids, image)

    def _build_text_only_gt(
        self,
        corpus_ids: list[int],
        corpus_to_chunks: dict[int, list[int | str]],
        is_multi_hop: bool,
    ) -> RetrievalGT | None:
        """Build GT expression for text-only mode."""
        if is_multi_hop:
            # Each corpus_id's chunks form one AND group, within each group any chunk is valid (OR)
            groups: list[OrGroup | TextId] = []
            for corpus_id in corpus_ids:
                chunk_ids = corpus_to_chunks.get(corpus_id, [])
                if chunk_ids:
                    groups.append(or_all(chunk_ids, text))  # type: ignore[arg-type]
            if not groups:
                return None
            return and_all_mixed([g._to_chunk_id() if isinstance(g, _IntWrapper) else g for g in groups])  # type: ignore[return-value, arg-type]
        else:
            all_chunk_ids: list[int | str] = []
            for corpus_id in corpus_ids:
                all_chunk_ids.extend(corpus_to_chunks.get(corpus_id, []))
            if not all_chunk_ids:
                return None
            return or_all(all_chunk_ids, text)

    def _build_both_gt(
        self,
        corpus_ids: list[int],
        corpus_to_chunks: dict[int, list[int | str]],
        is_multi_hop: bool,
    ) -> RetrievalGT | None:
        """Build GT expression for both mode (text and image chunks)."""
        if is_multi_hop:
            # Multi-hop: (TextChunks OR ImageChunk) AND (TextChunks OR ImageChunk) ...
            and_groups: list[OrGroup | TextId | ImageId] = []
            for corpus_id in corpus_ids:
                chunk_ids = corpus_to_chunks.get(corpus_id, [])
                items: list[TextId | ImageId] = [TextId(cid) for cid in chunk_ids]
                items.append(ImageId(corpus_id))
                and_groups.append(or_all_mixed(items))  # type: ignore[arg-type]
            if not and_groups:
                return None
            return and_all_mixed(and_groups)  # type: ignore[return-value, arg-type]
        else:
            # Non-multi-hop: all text chunks and all image chunks are OR alternatives
            items: list[TextId | ImageId] = []
            for corpus_id in corpus_ids:
                chunk_ids = corpus_to_chunks.get(corpus_id, [])
                items.extend(TextId(cid) for cid in chunk_ids)
                items.append(ImageId(corpus_id))
            if not items:
                return None
            return or_all_mixed(items)

    def embed_all(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        super().embed_all(max_concurrency=max_concurrency, batch_size=batch_size)
        if self.embedding_model is None or self.service is None:
            return
        self.service.embed_all_chunks(
            self.embedding_model.aget_text_embedding, batch_size=batch_size, max_concurrency=max_concurrency
        )

    def embed_all_late_interaction(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        super().embed_all_late_interaction(max_concurrency=max_concurrency, batch_size=batch_size)
        if self.late_interaction_embedding_model is None or self.service is None:
            return
        self.service.embed_all_chunks_multi_vector(
            self.late_interaction_embedding_model.aget_text_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
