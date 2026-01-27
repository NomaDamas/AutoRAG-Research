import logging
import random
from typing import Any, Literal

import pandas as pd
from datasets import load_dataset
from hydra.utils import instantiate
from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.core.node_parser import NodeParser, SentenceSplitter
from llama_index.core.schema import Document
from omegaconf import DictConfig

from autorag_research.data.base import MultiModalEmbeddingDataIngestor
from autorag_research.data.registry import register_ingestor
from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding
from autorag_research.exceptions import ServiceNotSetError
from autorag_research.orm.models import and_all, image, or_all
from autorag_research.util import pil_image_to_bytes

logger = logging.getLogger("AutoRAG-Research")

RANDOM_SEED = 42
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128


def _resolve_chunker(chunker: NodeParser | DictConfig | None) -> NodeParser:
    """Resolve chunker from config or use default.

    Args:
        chunker: A NodeParser instance, a Hydra DictConfig with _target_, or None.

    Returns:
        A NodeParser instance ready to use for chunking.
    """
    if chunker is None:
        return SentenceSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )
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
        embedding_model: MultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
    ):
        super().__init__(embedding_model, late_interaction_embedding_model)
        self.config_name = config_name

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        return "bigint"

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
        chunker: NodeParser | DictConfig | None = None,
    ) -> None:
        """Ingest ViDoReV3 dataset into the database.

        Args:
            subset: Dataset split to use. Only "test" is supported for ViDoReV3.
            query_limit: Maximum number of queries to ingest. If None, ingest all.
            min_corpus_cnt: Minimum number of corpus items to ingest. If more than
                gold corpus items are needed, additional items will be sampled.
            chunker: NodeParser instance, Hydra DictConfig with _target_, or None.
                If None, uses default SentenceSplitter(chunk_size=1024, chunk_overlap=128).
        """
        if self.service is None:
            raise ServiceNotSetError
        if subset != "test":
            raise ValueError("ViDoReV3 datasets only have 'test' split.")  # noqa: TRY003

        resolved_chunker = _resolve_chunker(chunker)

        dataset_path = f"vidore/vidore_v3_{self.config_name}"

        qrels_df = self._load_qrels(dataset_path, subset)
        queries_df = self._load_queries(dataset_path, subset)
        docs_metadata = self._load_documents_metadata(dataset_path, subset)
        corpus_mapping = self._load_corpus_mapping(dataset_path, subset)

        selected_queries_df, query_types_map = self._select_queries(qrels_df, queries_df, query_limit)
        gold_corpus_ids = self._extract_gold_corpus_ids(selected_queries_df)
        additional_needed = self._calculate_additional_corpus_needed(gold_corpus_ids, min_corpus_cnt)

        corpus_ids_to_ingest = self._determine_corpus_ids_to_ingest(
            corpus_mapping, gold_corpus_ids, additional_needed
        )

        required_doc_ids = self._get_required_doc_ids(corpus_mapping, corpus_ids_to_ingest)
        filtered_docs_metadata = docs_metadata[docs_metadata.index.isin(required_doc_ids)]

        doc_id_to_db_id = self._ingest_document_hierarchy(filtered_docs_metadata)

        ingested_corpus_ids = self._ingest_corpus(
            dataset_path, corpus_ids_to_ingest, resolved_chunker, doc_id_to_db_id
        )

        self._ingest_queries(selected_queries_df, queries_df)
        self._ingest_qrels(selected_queries_df, query_types_map, ingested_corpus_ids)

        logger.info(
            f"ViDoReV3 ingestion complete for '{self.config_name}': "
            f"{len(selected_queries_df)} queries, {len(ingested_corpus_ids)} corpus items"
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
        corpus_df: pd.DataFrame = load_dataset(  # type: ignore[union-attr, assignment]
            dataset_path, "corpus", streaming=False, split=subset
        ).to_pandas()
        corpus_df["corpus_id"] = corpus_df["corpus_id"].astype(int)
        return corpus_df[["corpus_id", "doc_id"]].set_index("corpus_id")

    def _select_queries(
        self,
        qrels_df: pd.DataFrame,
        queries_df: pd.DataFrame,
        query_limit: int | None,
    ) -> tuple[pd.DataFrame, dict[int, list[str]]]:
        grouped = qrels_df.groupby("query_id").agg(
            corpus_ids=pd.NamedAgg(column="corpus_id", aggfunc=list),
        )

        if query_limit is not None and query_limit < len(grouped):
            rng = random.Random(RANDOM_SEED)  # noqa: S311
            sampled_indices = rng.sample(list(grouped.index), query_limit)
            selected_queries_df = grouped.loc[sampled_indices]
        else:
            selected_queries_df = grouped

        query_types_map: dict[int, list[str]] = {}
        for query_id in selected_queries_df.index:
            if query_id in queries_df.index:
                query_types = queries_df.at[query_id, "query_types"]
                query_types_map[query_id] = query_types if isinstance(query_types, list) else []

        return selected_queries_df, query_types_map

    def _extract_gold_corpus_ids(self, selected_queries_df: pd.DataFrame) -> set[int]:
        gold_corpus_ids: set[int] = set()
        for corpus_list in selected_queries_df["corpus_ids"]:
            gold_corpus_ids.update(corpus_list)
        return gold_corpus_ids

    def _calculate_additional_corpus_needed(
        self, gold_corpus_ids: set[int], min_corpus_cnt: int | None
    ) -> int:
        if min_corpus_cnt is not None and min_corpus_cnt > len(gold_corpus_ids):
            return min_corpus_cnt - len(gold_corpus_ids)
        return 0

    def _determine_corpus_ids_to_ingest(
        self,
        corpus_mapping: pd.DataFrame,
        gold_ids: set[int],
        additional_needed: int,
    ) -> set[int]:
        corpus_ids_to_ingest = set(gold_ids)

        if additional_needed > 0:
            gold_doc_ids = {
                str(corpus_mapping.loc[cid, "doc_id"])
                for cid in gold_ids
                if cid in corpus_mapping.index
            }

            all_corpus_ids = set(corpus_mapping.index)
            non_gold_ids = list(all_corpus_ids - gold_ids)

            same_doc_ids = [
                cid for cid in non_gold_ids
                if str(corpus_mapping.loc[cid, "doc_id"]) in gold_doc_ids
            ]
            other_doc_ids = [
                cid for cid in non_gold_ids
                if str(corpus_mapping.loc[cid, "doc_id"]) not in gold_doc_ids
            ]

            rng = random.Random(RANDOM_SEED)  # noqa: S311
            rng.shuffle(same_doc_ids)
            rng.shuffle(other_doc_ids)

            candidates = same_doc_ids + other_doc_ids
            additional_ids = candidates[:additional_needed]
            corpus_ids_to_ingest.update(additional_ids)

        return corpus_ids_to_ingest

    def _get_required_doc_ids(self, corpus_mapping: pd.DataFrame, corpus_ids: set[int]) -> set[str]:
        filtered_mapping = corpus_mapping[corpus_mapping.index.isin(corpus_ids)]
        return {str(doc_id) for doc_id in filtered_mapping["doc_id"].unique()}

    def _ingest_document_hierarchy(self, docs_metadata: pd.DataFrame) -> dict[str, int]:
        if self.service is None:
            raise ServiceNotSetError

        doc_id_to_db_id: dict[str, int] = {}

        for doc_id, row in docs_metadata.iterrows():
            doc_id_str = str(doc_id)
            url = row.get("url", "")

            file_ids = self.service.add_files([{
                "type": "raw",
                "path": url if url else "",
            }])
            file_db_id = file_ids[0] if file_ids else None

            file_name = row.get("file_name", "")
            title = file_name.replace(".pdf", "").replace("_", " ") if file_name else None

            visual_types = row.get("visual_types")
            if hasattr(visual_types, "tolist"):
                visual_types = visual_types.tolist()

            doc_ids = self.service.add_documents([{
                "path": file_db_id,
                "filename": file_name,
                "title": title,
                "author": None,
                "doc_metadata": {
                    "url": url,
                    "doc_type": row.get("doc_type"),
                    "doc_language": row.get("doc_language"),
                    "doc_year": row.get("doc_year"),
                    "visual_types": visual_types,
                    "page_count": row.get("page_number"),
                    "license": row.get("license"),
                    "original_doc_id": doc_id_str,
                },
            }])

            if doc_ids:
                doc_id_to_db_id[doc_id_str] = doc_ids[0]

        return doc_id_to_db_id

    def _ingest_corpus(
        self,
        dataset_path: str,
        corpus_ids_to_ingest: set[int],
        chunker: NodeParser,
        doc_id_to_db_id: dict[str, int],
    ) -> set[int]:
        if self.service is None:
            raise ServiceNotSetError

        corpus_ds = load_dataset(dataset_path, "corpus", streaming=True, split="test")
        ingested_ids: set[int] = set()

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
                page_ids = self.service.add_pages([{
                    "document_id": doc_db_id,
                    "page_num": page_num,
                    "image_contents": img_bytes,
                    "mimetype": mimetype,
                }])
                if page_ids:
                    page_key_to_db_id[page_key] = page_ids[0]

            page_db_id = page_key_to_db_id.get(page_key)

            if markdown.strip() and page_db_id is not None:
                caption_ids = self.service.add_captions([{
                    "page_id": page_db_id,
                    "contents": markdown,
                }])
                caption_db_id = caption_ids[0] if caption_ids else None

                if caption_db_id is not None:
                    chunk_texts = chunk_markdown(markdown, chunker)
                    if chunk_texts:
                        chunks_data = [
                            {
                                "parent_caption": caption_db_id,
                                "contents": chunk_text,
                                "is_table": False,
                            }
                            for chunk_text in chunk_texts
                        ]
                        self.service.add_chunks(chunks_data)

            self.service.add_image_chunks([{
                "id": corpus_id,
                "contents": img_bytes,
                "mimetype": mimetype,
                "parent_page": page_db_id,
            }])

            ingested_ids.add(corpus_id)

        return ingested_ids

    def _ingest_queries(
        self,
        selected_queries_df: pd.DataFrame,
        queries_df: pd.DataFrame,
    ) -> None:
        if self.service is None:
            raise ServiceNotSetError

        queries_batch: list[dict[str, Any]] = []

        for query_id in selected_queries_df.index:
            if query_id not in queries_df.index:
                continue

            query_row = queries_df.loc[query_id]
            query_text = str(query_row["query"])
            answer = query_row.get("answer")

            query_data: dict[str, Any] = {
                "id": query_id,
                "contents": query_text,
            }

            if answer:
                query_data["generation_gt"] = [answer]

            queries_batch.append(query_data)

        if queries_batch:
            self.service.add_queries(queries_batch)

    def _ingest_qrels(
        self,
        selected_queries_df: pd.DataFrame,
        query_types_map: dict[int, list[str]],
        ingested_corpus_ids: set[int],
    ) -> None:
        if self.service is None:
            raise ServiceNotSetError

        for query_id in selected_queries_df.index:
            corpus_ids = selected_queries_df.at[query_id, "corpus_ids"]
            valid_corpus_ids = [cid for cid in corpus_ids if cid in ingested_corpus_ids]

            if not valid_corpus_ids:
                continue

            query_types = query_types_map.get(query_id, [])
            is_multi_hop = "multi-hop" in query_types

            gt_expr = and_all(valid_corpus_ids, image) if is_multi_hop else or_all(valid_corpus_ids, image)

            self.service.add_retrieval_gt(
                query_id,  # type: ignore[arg-type]
                gt_expr,
                chunk_type="image",
            )

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
