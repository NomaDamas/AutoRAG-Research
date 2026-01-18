import logging
import random
from typing import Literal

from datasets import load_dataset
from llama_index.core.base.embeddings.base import BaseEmbedding

from autorag_research.data.base import TextEmbeddingDataIngestor
from autorag_research.exceptions import ServiceNotSetError, UnsupportedLanguageError

logger = logging.getLogger("AutoRAG-Research")

RANDOM_SEED = 42

LANGUAGE_CONFIGS = {
    "arabic": "ar",
    "bengali": "bn",
    "english": "en",
    "finnish": "fi",
    "indonesian": "id",
    "japanese": "ja",
    "korean": "ko",
    "russian": "ru",
    "swahili": "sw",
    "telugu": "te",
    "thai": "th",
}


class MrTyDiIngestor(TextEmbeddingDataIngestor):
    """Ingestor for Mr. TyDi multilingual retrieval benchmark dataset.

    Mr. TyDi is a multi-lingual benchmark dataset for monolingual information retrieval,
    covering 11 typologically diverse languages.

    Dataset: https://huggingface.co/datasets/castorini/mr-tydi
    """

    def __init__(self, embedding_model: BaseEmbedding, language: str = "english"):
        """Initialize Mr. TyDi ingestor.

        Args:
            embedding_model: Embedding model for vectorization.
            language: Language to ingest. One of: arabic, bengali, english, finnish,
                     indonesian, japanese, korean, russian, swahili, telugu, thai.
        """
        super().__init__(embedding_model)
        if language.lower() not in LANGUAGE_CONFIGS:
            raise UnsupportedLanguageError(language.lower(), list(LANGUAGE_CONFIGS.keys()))
        self.language = language.lower()
        self.language_code = LANGUAGE_CONFIGS[self.language]

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        """Mr. TyDi uses string primary keys (e.g., '26569#0')."""
        return "string"

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        corpus_limit: int | None = None,
    ) -> None:
        """Ingest Mr. TyDi dataset.

        Args:
            subset: Dataset split to ingest (train, dev, or test).
            query_limit: Maximum number of queries to ingest.
            corpus_limit: Maximum number of corpus items to ingest.
                         Gold passages are always included.
        """
        if self.service is None:
            raise ServiceNotSetError

        logger.info(f"Loading Mr. TyDi dataset ({self.language}, {subset} split)...")
        dataset = load_dataset("castorini/mr-tydi", self.language_code, split=subset)

        rng = random.Random(RANDOM_SEED)  # noqa: S311

        # Step 1: Sample queries and extract data
        queries, qrels, corpus = self._process_dataset(dataset, query_limit, corpus_limit, rng)

        # Step 2: Ingest data
        self._ingest_queries(queries)
        self._ingest_corpus(corpus)
        self._ingest_qrels(qrels, set(corpus.keys()))

        self.service.clean()

    def _process_dataset(
        self,
        dataset,
        query_limit: int | None,
        corpus_limit: int | None,
        rng: random.Random,
    ) -> tuple[dict, dict, dict]:
        """Process dataset to extract queries, qrels, and corpus.

        Args:
            dataset: HuggingFace dataset object.
            query_limit: Maximum queries to include.
            corpus_limit: Maximum corpus items to include.
            rng: Random number generator for sampling.

        Returns:
            Tuple of (queries, qrels, corpus) dictionaries.
        """
        # Convert to list for sampling
        data_list = list(dataset)

        # Sample queries if limit specified
        if query_limit is not None and query_limit < len(data_list):
            data_list = rng.sample(data_list, query_limit)
            logger.info(f"Sampled {len(data_list)} queries from {len(dataset)} total")

        # Extract queries, qrels, and corpus
        queries: dict[str, str] = {}
        qrels: dict[str, dict[str, int]] = {}
        corpus: dict[str, dict[str, str]] = {}
        gold_corpus_ids: set[str] = set()

        for row in data_list:
            qid = str(row["query_id"])
            queries[qid] = row["query"]
            qrels[qid] = {}

            # Process positive passages (relevant documents)
            for passage in row["positive_passages"]:
                docid = passage["docid"]
                qrels[qid][docid] = 1
                gold_corpus_ids.add(docid)
                if docid not in corpus:
                    corpus[docid] = {
                        "title": passage["title"],
                        "text": passage["text"],
                    }

            # Process negative passages (add to corpus but not qrels)
            for passage in row["negative_passages"]:
                docid = passage["docid"]
                if docid not in corpus:
                    corpus[docid] = {
                        "title": passage["title"],
                        "text": passage["text"],
                    }

        # Apply corpus limit if specified
        if corpus_limit is not None and len(corpus) > corpus_limit:
            corpus = self._filter_corpus(corpus, gold_corpus_ids, corpus_limit, rng)

        return queries, qrels, corpus

    def _filter_corpus(
        self,
        corpus: dict,
        gold_corpus_ids: set[str],
        corpus_limit: int,
        rng: random.Random,
    ) -> dict:
        """Filter corpus to include gold IDs + random samples up to limit.

        Args:
            corpus: Full corpus dictionary.
            gold_corpus_ids: Set of gold (relevant) corpus IDs that must be included.
            corpus_limit: Maximum corpus size.
            rng: Random number generator.

        Returns:
            Filtered corpus dictionary.
        """
        # Always include gold IDs
        gold_in_corpus = gold_corpus_ids & set(corpus.keys())
        selected_ids = list(gold_in_corpus)
        remaining_ids = [cid for cid in corpus if cid not in gold_in_corpus]

        # Add random samples if we need more
        additional_needed = corpus_limit - len(selected_ids)
        if additional_needed > 0 and remaining_ids:
            additional_ids = rng.sample(
                remaining_ids,
                min(additional_needed, len(remaining_ids)),
            )
            selected_ids.extend(additional_ids)

        logger.info(
            f"Corpus subset: {len(gold_in_corpus)} gold IDs + "
            f"{len(selected_ids) - len(gold_in_corpus)} random = {len(selected_ids)} total"
        )

        return {cid: corpus[cid] for cid in selected_ids}

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

    def embed_all(self, max_concurrency: int = 16, batch_size: int = 128) -> None:
        """Embed all queries and chunks.

        Args:
            max_concurrency: Maximum concurrent embedding requests.
            batch_size: Batch size for embedding.
        """
        if self.service is None:
            raise ServiceNotSetError
        self.service.embed_all_queries(
            self.embedding_model.aget_query_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
        self.service.embed_all_chunks(
            self.embedding_model.aget_text_embedding,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
