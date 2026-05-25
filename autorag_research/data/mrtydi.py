import logging
import random
from typing import Any, Literal, get_args

from datasets import load_dataset
from langchain_core.embeddings import Embeddings

from autorag_research.data.base import TextEmbeddingDataIngestor
from autorag_research.data.registry import register_ingestor
from autorag_research.exceptions import ServiceNotSetError, UnsupportedLanguageError

logger = logging.getLogger("AutoRAG-Research")

RANDOM_SEED = 42
DEFAULT_BATCH_SIZE = 1000

# Mr. TyDi supported languages
MRTYDI_LANGUAGES = Literal[
    "arabic",
    "bengali",
    "english",
    "finnish",
    "indonesian",
    "japanese",
    "korean",
    "russian",
    "swahili",
    "telugu",
    "thai",
]

MRTYDI_BASE_URL = "https://huggingface.co/datasets/castorini/mr-tydi/resolve/main"
MRTYDI_CORPUS_BASE_URL = "https://huggingface.co/datasets/castorini/mr-tydi-corpus/resolve/main"


@register_ingestor(
    name="mrtydi",
    description="Mr. TyDi multilingual retrieval benchmark",
    hf_repo="mrtydi-dumps",
)
class MrTyDiIngestor(TextEmbeddingDataIngestor):
    """Ingestor for Mr. TyDi multilingual retrieval benchmark dataset.

    Mr. TyDi is a multi-lingual benchmark dataset for monolingual information retrieval,
    covering 11 typologically diverse languages.

    Dataset: https://huggingface.co/datasets/castorini/mr-tydi
    Corpus: https://huggingface.co/datasets/castorini/mr-tydi-corpus
    """

    def __init__(
        self,
        embedding_model: Embeddings,
        language: MRTYDI_LANGUAGES = "english",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """Initialize Mr. TyDi ingestor.

        Args:
            embedding_model: Embedding model for vectorization.
            language: Language to ingest. One of: arabic, bengali, english, finnish,
                     indonesian, japanese, korean, russian, swahili, telugu, thai.
            batch_size: Maximum number of rows to send to database insertion in one call.
        """
        super().__init__(embedding_model)
        valid_languages = get_args(MRTYDI_LANGUAGES)
        if language.lower() not in valid_languages:
            raise UnsupportedLanguageError(language.lower(), list(valid_languages))
        self.language = language.lower()
        self.language_dir = f"mrtydi-v1.1-{self.language}"
        self.batch_size = batch_size

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        """Mr. TyDi uses string primary keys (e.g., '26569#0')."""
        return "string"

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
    ) -> None:
        """Ingest Mr. TyDi dataset.

        Args:
            subset: Dataset split to ingest (train, dev, or test).
                Recommend to use only 'test' subset for benchmarking.
            query_limit: Maximum number of queries to ingest.
            min_corpus_cnt: Maximum number of corpus items to ingest.
                         Gold passages are always included.
        """
        if self.service is None:
            raise ServiceNotSetError

        rng = random.Random(RANDOM_SEED)

        # Step 1: Load queries and extract gold docids
        logger.info(f"Loading Mr. TyDi queries ({self.language}, {subset} split)...")
        queries_url = f"{MRTYDI_BASE_URL}/{self.language_dir}/{subset}.jsonl.gz"
        queries_dataset = load_dataset("json", data_files=queries_url, split="train", streaming=True)

        queries, qrels, gold_docids = self._process_queries(queries_dataset, query_limit, rng)

        # Step 2: Load corpus
        logger.info(f"Loading Mr. TyDi corpus ({self.language})...")
        corpus_url = f"{MRTYDI_CORPUS_BASE_URL}/{self.language_dir}/corpus.jsonl.gz"
        corpus_dataset = load_dataset("json", data_files=corpus_url, split="train", streaming=True)

        # Step 3: Ingest data with bounded corpus memory
        self._ingest_queries(queries)
        corpus_ids_set = self._ingest_corpus_streaming(corpus_dataset, gold_docids, min_corpus_cnt, rng)
        self._ingest_qrels(qrels, corpus_ids_set)

        self.service.clean()

    def _process_queries(
        self,
        dataset,
        query_limit: int | None,
        rng: random.Random,
    ) -> tuple[dict[str, str], dict[str, dict[str, int]], set[str]]:
        """Process query dataset to extract queries, qrels, and gold docids.

        Args:
            dataset: HuggingFace dataset with queries.
            query_limit: Maximum queries to include.
            rng: Random number generator for sampling.

        Returns:
            Tuple of (queries, qrels, gold_docids).
        """
        rows = self._iter_limited_or_sampled_rows(dataset, query_limit, rng)

        queries: dict[str, str] = {}
        qrels: dict[str, dict[str, int]] = {}
        gold_docids: set[str] = set()

        for row in rows:
            qid = str(row["query_id"])
            queries[qid] = row["query"]
            qrels[qid] = {}

            # Process positive passages (only docids, no content)
            for passage in row["positive_passages"]:
                docid = passage["docid"]
                qrels[qid][docid] = 1
                gold_docids.add(docid)

        logger.info(f"Extracted {len(queries)} queries with {len(gold_docids)} unique gold docids")
        return queries, qrels, gold_docids

    def _iter_limited_or_sampled_rows(
        self,
        dataset,
        query_limit: int | None,
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        """Return all rows or a bounded reservoir sample without materializing unlimited streams."""
        if query_limit is None:
            return list(dataset)

        reservoir: list[dict[str, Any]] = []
        for seen_count, row in enumerate(dataset, start=1):
            row_dict: dict[str, Any] = row
            if len(reservoir) < query_limit:
                reservoir.append(row_dict)
                continue
            replacement_index = rng.randrange(seen_count)
            if replacement_index < query_limit:
                reservoir[replacement_index] = row_dict

        logger.info(f"Sampled {len(reservoir)} queries from streaming Mr. TyDi queries")
        return reservoir

    def _process_corpus(
        self,
        dataset,
        gold_docids: set[str],
        min_corpus_cnt: int | None,
        rng: random.Random,
    ) -> dict[str, dict[str, str]]:
        """Process corpus dataset to extract documents.

        Args:
            dataset: HuggingFace dataset with corpus.
            gold_docids: Set of gold docids that must be included.
            min_corpus_cnt: Maximum corpus size.
            rng: Random number generator for sampling.

        Returns:
            Corpus dictionary mapping docid to {title, text}.
        """
        # Build corpus dict - only include gold docids + random samples if limit is set
        corpus: dict[str, dict[str, str]] = {}
        non_gold_docs: list[dict] = []

        for row in dataset:
            docid = row["docid"]
            doc_data = {"title": row["title"], "text": row["text"]}

            if docid in gold_docids:
                corpus[docid] = doc_data
            elif min_corpus_cnt is None:
                # No limit - include all docs
                corpus[docid] = doc_data
            else:
                # Limit specified - collect non-gold for later sampling
                non_gold_docs.append({"docid": docid, **doc_data})

        # If min_corpus_cnt is set, add random non-gold docs up to the threshold
        if min_corpus_cnt is not None:
            additional_needed = min_corpus_cnt - len(corpus)
            if additional_needed > 0 and non_gold_docs:
                sampled = rng.sample(non_gold_docs, min(additional_needed, len(non_gold_docs)))
                for doc in sampled:
                    corpus[doc["docid"]] = {"title": doc["title"], "text": doc["text"]}

            logger.info(
                f"Corpus subset: {len(gold_docids)} gold IDs + "
                f"{len(corpus) - len(gold_docids)} random = {len(corpus)} total"
            )
        else:
            logger.info(f"Loaded full corpus: {len(corpus)} documents")

        return corpus

    def _ingest_corpus_streaming(
        self,
        dataset,
        gold_docids: set[str],
        min_corpus_cnt: int | None,
        rng: random.Random,
    ) -> set[str]:
        """Stream corpus documents into the database with bounded memory.

        Gold documents are always inserted. When ``min_corpus_cnt`` is set,
        non-gold documents are selected with reservoir sampling so the candidate
        pool does not grow with corpus size. When no limit is set, every corpus
        row is inserted in database batches instead of first building a full
        in-memory dictionary.
        """
        if self.service is None:
            raise ServiceNotSetError

        corpus_ids_set: set[str] = set()
        batch: list[dict[str, str | int | None]] = []
        non_gold_sample: list[dict[str, str | int | None]] = []
        non_gold_seen = 0
        additional_needed = None if min_corpus_cnt is None else max(0, min_corpus_cnt - len(gold_docids))

        for row in dataset:
            docid = row["docid"]
            chunk = self._make_chunk(row)
            if docid in gold_docids or min_corpus_cnt is None:
                if docid in gold_docids:
                    corpus_ids_set.add(docid)
                batch.append(chunk)
                batch = self._flush_chunk_batch_if_full(batch)
            elif additional_needed and additional_needed > 0:
                non_gold_seen = self._update_reservoir_sample(
                    non_gold_sample, chunk, additional_needed, non_gold_seen, rng
                )

        self._flush_chunk_batch(batch)

        for start in range(0, len(non_gold_sample), self.batch_size):
            sampled_batch = non_gold_sample[start : start + self.batch_size]
            self.service.add_chunks(sampled_batch)
            corpus_ids_set.update(str(chunk["id"]) for chunk in sampled_batch)

        logger.info(
            "Mr. TyDi corpus ingested with %s gold IDs and %s retrieval-eligible inserted chunks",
            len(gold_docids),
            len(corpus_ids_set),
        )
        return corpus_ids_set

    def _flush_chunk_batch_if_full(
        self,
        batch: list[dict[str, str | int | None]],
    ) -> list[dict[str, str | int | None]]:
        """Flush and reset a chunk batch when it reaches the configured size."""
        if len(batch) < self.batch_size:
            return batch
        self._flush_chunk_batch(batch)
        return []

    def _flush_chunk_batch(self, batch: list[dict[str, str | int | None]]) -> None:
        """Send one non-empty chunk batch to the ingestion service."""
        if self.service is None:
            raise ServiceNotSetError
        if batch:
            self.service.add_chunks(batch)

    @staticmethod
    def _update_reservoir_sample(
        reservoir: list[dict[str, str | int | None]],
        chunk: dict[str, str | int | None],
        capacity: int,
        seen_count: int,
        rng: random.Random,
    ) -> int:
        """Update a fixed-size reservoir sample and return the new seen count."""
        next_seen_count = seen_count + 1
        if len(reservoir) < capacity:
            reservoir.append(chunk)
            return next_seen_count
        replacement_index = rng.randrange(next_seen_count)
        if replacement_index < capacity:
            reservoir[replacement_index] = chunk
        return next_seen_count

    @staticmethod
    def _make_chunk(row: dict[str, Any]) -> dict[str, str | int | None]:
        """Build one text chunk row from a Mr. TyDi corpus row."""
        return {
            "id": row["docid"],
            "contents": (row.get("title", "") + " " + row["text"]).strip(),
        }

    def _ingest_queries(self, queries: dict[str, str]) -> None:
        """Ingest queries into the database."""
        if self.service is None:
            raise ServiceNotSetError
        logger.info(f"Ingesting {len(queries)} queries from Mr. TyDi ({self.language})...")
        self.service.add_queries([{"id": qid, "contents": text} for qid, text in queries.items()])

    def _ingest_corpus(self, corpus: dict[str, dict[str, str]]) -> None:
        """Ingest corpus documents into the database."""
        if self.service is None:
            raise ServiceNotSetError
        logger.info(f"Ingesting {len(corpus)} corpus documents from Mr. TyDi ({self.language})...")
        self.service.add_chunks([
            {
                "id": cid,
                "contents": (doc.get("title", "") + " " + doc["text"]).strip(),
            }
            for cid, doc in corpus.items()
        ])

    def _ingest_qrels(self, qrels: dict[str, dict[str, int]], corpus_ids_set: set[str]) -> None:
        """Ingest query-document relevance relations."""
        if self.service is None:
            raise ServiceNotSetError

        from autorag_research.orm.models import or_all

        for qid, doc_dict in qrels.items():
            # Filter to only include corpus IDs that exist in our filtered corpus
            gt_ids = [docid for docid, score in doc_dict.items() if score > 0 and docid in corpus_ids_set]
            if not gt_ids:
                continue
            self.service.add_retrieval_gt(qid, or_all(gt_ids), chunk_type="text")
