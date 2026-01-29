import logging
import random
from typing import Literal, get_args

from datasets import load_dataset
from llama_index.core.base.embeddings.base import BaseEmbedding

from autorag_research.data.base import TextEmbeddingDataIngestor
from autorag_research.data.registry import register_ingestor
from autorag_research.exceptions import ServiceNotSetError, UnsupportedLanguageError

logger = logging.getLogger("AutoRAG-Research")

RANDOM_SEED = 42

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

    def __init__(self, embedding_model: BaseEmbedding, language: MRTYDI_LANGUAGES = "english"):
        """Initialize Mr. TyDi ingestor.

        Args:
            embedding_model: Embedding model for vectorization.
            language: Language to ingest. One of: arabic, bengali, english, finnish,
                     indonesian, japanese, korean, russian, swahili, telugu, thai.
        """
        super().__init__(embedding_model)
        valid_languages = get_args(MRTYDI_LANGUAGES)
        if language.lower() not in valid_languages:
            raise UnsupportedLanguageError(language.lower(), list(valid_languages))
        self.language = language.lower()
        self.language_dir = f"mrtydi-v1.1-{self.language}"

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
        queries_dataset = load_dataset("json", data_files=queries_url, split="train")

        queries, qrels, gold_docids = self._process_queries(queries_dataset, query_limit, rng)

        # Step 2: Load corpus
        logger.info(f"Loading Mr. TyDi corpus ({self.language})...")
        corpus_url = f"{MRTYDI_CORPUS_BASE_URL}/{self.language_dir}/corpus.jsonl.gz"
        corpus_dataset = load_dataset("json", data_files=corpus_url, split="train")

        corpus = self._process_corpus(corpus_dataset, gold_docids, min_corpus_cnt, rng)

        # Step 3: Ingest data
        self._ingest_queries(queries)
        self._ingest_corpus(corpus)
        self._ingest_qrels(qrels, set(corpus.keys()))

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
        data_list = list(dataset)

        # Sample queries if limit specified
        if query_limit is not None and query_limit < len(data_list):
            data_list = rng.sample(data_list, query_limit)
            logger.info(f"Sampled {len(data_list)} queries from {len(dataset)} total")

        queries: dict[str, str] = {}
        qrels: dict[str, dict[str, int]] = {}
        gold_docids: set[str] = set()

        for row in data_list:
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
