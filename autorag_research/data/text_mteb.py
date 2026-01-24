"""MTEB dataset ingestor using the official mteb library."""

import logging
import random
from typing import Literal

import mteb
import pandas as pd
from llama_index.core.base.embeddings.base import BaseEmbedding

from autorag_research.data.base import TextEmbeddingDataIngestor
from autorag_research.data.registry import register_ingestor
from autorag_research.exceptions import ServiceNotSetError, UnsupportedDataSubsetError, UnsupportedMTEBTaskTypeError
from autorag_research.orm.models import or_all

logger = logging.getLogger("AutoRAG-Research")

RANDOM_SEED = 42

SUPPORTED_TASK_TYPES = {"Retrieval", "InstructionRetrieval"}


def _combine_title_text(row: pd.Series) -> str:
    """Combine title and text from a corpus row."""
    title = row.get("title", "") or ""
    text = row.get("text", "") or ""
    if title and title.strip():
        return f"{title.strip()}\n{text.strip()}"
    return text.strip()


@register_ingestor(
    name="mteb",
    description="MTEB (Massive Text Embedding Benchmark) retrieval tasks",
)
class TextMTEBDatasetIngestor(TextEmbeddingDataIngestor):
    """Text-only ingestor for MTEB retrieval datasets using official mteb library.

    Supported task types:
        - Retrieval: Standard text retrieval (e.g., NFCorpus, SciFact, MSMARCO)
        - InstructionRetrieval: Retrieval with query instructions (e.g., IFIRNFCorpus)

    Note:
        - Any2AnyRetrieval and Any2AnyMultilingualRetrieval are NOT supported
          as they involve cross-modal retrieval which is out of scope for AutoRAG-Research.
        - Image-based tasks (t2i, i2i, etc.) are NOT supported.

    For available task names, see:
        https://embeddings-benchmark.github.io/mteb/overview/available_tasks/retrieval/
        https://embeddings-benchmark.github.io/mteb/overview/available_tasks/instructionretrieval/

    Example usage:
        # Standard retrieval
        ingestor = TextMTEBDatasetIngestor(embedding_model, "NFCorpus")
        ingestor.set_service(service)
        ingestor.ingest(subset="test")

        # Instruction retrieval (instruction prepended by default)
        ingestor = TextMTEBDatasetIngestor(embedding_model, "IFIRNFCorpus")
        ingestor.ingest(subset="test")
    """

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        task_name: str,
        score_threshold: int = 1,
        include_instruction: bool = True,
    ):
        """Initialize the MTEB dataset ingestor.

        Args:
            embedding_model: LlamaIndex embedding model to use.
            task_name: MTEB task name (e.g., "NFCorpus", "SciFact", "MSMARCO").
                       Use exact name from MTEB registry.
            score_threshold: Minimum relevance score for a document to be
                           considered relevant. Default is 1.
                           0 - not relevant
                           1 - somewhat relevant
                           2 - very relevant
            include_instruction: Whether to prepend instructions to queries for
                               InstructionRetrieval tasks. Default is True.
        """
        super().__init__(embedding_model)
        self.task_name = task_name
        self.score_threshold = score_threshold
        self.include_instruction = include_instruction
        self._task = None

    def _get_task(self):
        """Get or load the MTEB task.

        Returns an MTEB retrieval task with loaded data.
        """
        if self._task is None:
            self._task = mteb.get_task(self.task_name)
            self._task.load_data()
        return self._task

    def _validate_task(self) -> None:
        """Validate task type.

        Raises:
            UnsupportedMTEBTaskTypeError: If the task type is not supported.
        """
        task = self._get_task()
        task_type = task.metadata.type

        if task_type not in SUPPORTED_TASK_TYPES:
            raise UnsupportedMTEBTaskTypeError(self.task_name, task_type, list(SUPPORTED_TASK_TYPES))

    def _get_split_data(self, split: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
        """Get corpus, queries, and relevant_docs for a split.

        MTEB stores data in task.dataset['default'][split] with:
        - 'corpus': HuggingFace Dataset with columns: id, title, text
        - 'queries': HuggingFace Dataset with columns: id, text (and optionally instruction)
        - 'relevant_docs': dict[query_id, dict[doc_id, score]]

        This method converts HuggingFace Datasets to pandas DataFrames for efficient access.

        Args:
            split: Dataset split name (e.g., 'test', 'dev', 'train').

        Returns:
            Tuple of (corpus_df, queries_df, relevant_docs).
            - corpus_df: DataFrame indexed by id with columns [title, text]
            - queries_df: DataFrame indexed by id with column [text]
            - relevant_docs: dict[query_id, dict[doc_id, score]]
        """
        task = self._get_task()
        task_type = task.metadata.type
        is_instruction = task_type == "InstructionRetrieval"

        split_data = task.dataset["default"][split]

        # Convert corpus to DataFrame with id as index
        corpus_df = split_data["corpus"].to_pandas().set_index("id")
        # Ensure title column exists
        if "title" not in corpus_df.columns:
            corpus_df["title"] = ""

        # Convert queries to DataFrame with id as index
        queries_df = split_data["queries"].to_pandas().set_index("id")

        # Handle InstructionRetrieval: prepend instruction to query text
        if is_instruction and self.include_instruction and "instruction" in queries_df.columns:
            mask = queries_df["instruction"].notna() & (queries_df["instruction"] != "")
            queries_df.loc[mask, "text"] = queries_df.loc[mask, "instruction"] + "\n\n" + queries_df.loc[mask, "text"]

        # relevant_docs is already a dict
        relevant_docs: dict[str, dict[str, int]] = split_data["relevant_docs"]

        return corpus_df, queries_df, relevant_docs

    def _get_available_splits(self) -> list[str]:
        """Get list of available splits for this task.

        Returns:
            List of available split names.
        """
        task = self._get_task()
        return list(task.dataset["default"].keys())

    def _resolve_subset(self, requested: str) -> str:
        """Resolve the requested subset to an available split.

        Args:
            requested: Requested subset name (train, dev, test).

        Returns:
            The resolved subset name that exists in the dataset.

        Raises:
            UnsupportedDataSubsetError: If the requested subset is not available.
        """
        available = self._get_available_splits()
        if requested in available:
            return requested
        if len(available) == 1:
            logger.warning(f"Subset '{requested}' not available, using '{available[0]}' instead")
            return available[0]
        raise UnsupportedDataSubsetError([requested])

    def _get_gold_ids(self, relevant_docs: dict[str, dict[str, int]], query_id: str) -> list[str]:
        """Get gold document IDs for a query, filtering by score threshold.

        Args:
            relevant_docs: Relevance judgments for queries.
            query_id: The query ID to get gold documents for.

        Returns:
            List of document IDs with score >= score_threshold.
        """
        if query_id not in relevant_docs:
            return []
        return [doc_id for doc_id, score in relevant_docs[query_id].items() if score >= self.score_threshold]

    def _filter_corpus(
        self,
        corpus_df: pd.DataFrame,
        gold_ids: set[str],
        min_corpus_cnt: int | None,
        rng: random.Random,
    ) -> pd.Index:
        """Filter corpus: always include gold IDs, plus random to reach limit.

        Args:
            corpus_df: Full corpus DataFrame indexed by id.
            gold_ids: Set of gold document IDs that must be included.
            min_corpus_cnt: Maximum number of corpus items. None means no limit.
            rng: Random number generator for sampling.

        Returns:
            Index of corpus IDs to include.
        """
        if min_corpus_cnt is None:
            return corpus_df.index

        # Always include gold IDs that exist in corpus
        gold_in_corpus = corpus_df.index[corpus_df.index.isin(gold_ids)]
        non_gold = corpus_df.index[~corpus_df.index.isin(gold_ids)]

        # Calculate how many additional items we need
        remaining = max(0, min_corpus_cnt - len(gold_in_corpus))
        sampled_count = min(remaining, len(non_gold))
        sampled_indices = rng.sample(list(non_gold), sampled_count) if sampled_count > 0 else []

        logger.info(
            f"Corpus subset: {len(gold_in_corpus)} gold + {len(sampled_indices)} random = "
            f"{len(gold_in_corpus) + len(sampled_indices)} total (from {len(corpus_df)} total)"
        )
        return gold_in_corpus.append(pd.Index(sampled_indices))

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        """Detect the primary key type used in the dataset."""
        available = self._get_available_splits()
        split = available[0]
        corpus_df, queries_df, _ = self._get_split_data(split)

        all_ids = list(corpus_df.index) + list(queries_df.index)
        # If any ID is a string that can't be converted to int, use string type
        for id_value in all_ids:
            if isinstance(id_value, str):
                try:
                    int(id_value)
                except ValueError:
                    return "string"
        # If all IDs are numeric strings or integers, check if they're within bigint range
        return "string"  # Default to string for safety

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
    ) -> None:
        """Ingest data from MTEB dataset.

        Args:
            subset: Dataset split to ingest (train, dev, or test).
            query_limit: Maximum number of queries to ingest. None means no limit.
            min_corpus_cnt: Maximum number of corpus items to ingest.
                          When set, gold IDs from selected queries are always included,
                          plus random samples to reach the limit. None means no limit.

        Raises:
            ServiceNotSetError: If service is not set.
            UnsupportedMTEBTaskTypeError: If the task type is not supported.
        """
        if self.service is None:
            raise ServiceNotSetError

        # Validate task type before proceeding
        self._validate_task()

        split = self._resolve_subset(subset)
        corpus_df, queries_df, relevant_docs = self._get_split_data(split)

        rng = random.Random(RANDOM_SEED)  # noqa: S311

        # Step 1: Sample queries (only those with relevant docs above score_threshold)
        query_ids_with_relations = [qid for qid in queries_df.index if self._get_gold_ids(relevant_docs, qid)]
        query_ids = query_ids_with_relations
        if query_limit is not None and query_limit < len(query_ids):
            query_ids = rng.sample(query_ids, query_limit)
            logger.info(f"Sampled {len(query_ids)} queries from {len(query_ids_with_relations)} with relations")

        # Step 2: Collect gold IDs from selected queries
        gold_ids: set[str] = set()
        for qid in query_ids:
            gold_ids.update(self._get_gold_ids(relevant_docs, qid))
        logger.info(f"Total gold IDs: {len(gold_ids)}")

        # Step 3: Filter corpus (gold IDs + random sampling)
        corpus_ids = self._filter_corpus(corpus_df, gold_ids, min_corpus_cnt, rng)
        corpus_ids_set = set(corpus_ids)

        # Step 4: Ingest queries
        logger.info(f"Ingesting {len(query_ids)} queries from MTEB task '{self.task_name}'...")
        self.service.add_queries([{"id": qid, "contents": queries_df.loc[qid, "text"]} for qid in query_ids])

        # Step 5: Ingest corpus
        logger.info(f"Ingesting {len(corpus_ids)} corpus documents from MTEB task '{self.task_name}'...")
        corpus_subset = corpus_df.loc[corpus_ids]
        self.service.add_chunks([
            {"id": cid, "contents": _combine_title_text(row)} for cid, row in corpus_subset.iterrows()
        ])

        # Step 6: Ingest retrieval relations
        for qid in query_ids:
            gt_ids = [gid for gid in self._get_gold_ids(relevant_docs, qid) if gid in corpus_ids_set]
            if not gt_ids:
                continue
            self.service.add_retrieval_gt(qid, or_all(gt_ids), chunk_type="text")

        self.service.clean()
        logger.info(f"MTEB '{self.task_name}' ingestion complete.")

    def embed_all(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        """Embed all queries and chunks.

        Args:
            max_concurrency: Maximum concurrent embedding requests.
            batch_size: Batch size for embedding requests.
        """
        if self.service is None:
            raise ServiceNotSetError

        logger.info("Embedding all queries...")
        self.service.embed_all_queries(
            self.embedding_model.aget_query_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )

        logger.info("Embedding all chunks...")
        self.service.embed_all_chunks(
            self.embedding_model.aget_text_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )

        logger.info(f"MTEB '{self.task_name}' embedding complete.")
