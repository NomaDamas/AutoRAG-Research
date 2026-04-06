import logging
import re
from html import unescape
from html.parser import HTMLParser
from typing import Any, Literal

from datasets import load_dataset
from langchain_core.embeddings import Embeddings

from autorag_research.data.base import TextEmbeddingDataIngestor
from autorag_research.data.registry import register_ingestor
from autorag_research.data.util import make_id
from autorag_research.exceptions import ServiceNotSetError, UnsupportedDataSubsetError

logger = logging.getLogger("AutoRAG-Research")

CRAG_DATA_URL = "https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2"
CRAG_SUBSET_TO_SPLIT = {
    "train": 0,
    "dev": 0,
    "test": 1,
}
DEFAULT_BATCH_SIZE = 100


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data:
            self._parts.append(data)

    def get_text(self) -> str:
        return _normalize_text(" ".join(self._parts))


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", unescape(value)).strip()


def _resolve_subset(subset: str) -> int:
    if subset not in CRAG_SUBSET_TO_SPLIT:
        raise UnsupportedDataSubsetError([subset])
    return CRAG_SUBSET_TO_SPLIT[subset]


def _make_query_id(subset: str, interaction_id: str) -> str:
    return make_id("crag", subset, interaction_id)


def _make_chunk_id(subset: str, interaction_id: str, result_index: int) -> str:
    return make_id("crag", subset, interaction_id, result_index)


def _extract_page_text(page_result: str | None) -> str:
    if not page_result:
        return ""

    parser = _HTMLTextExtractor()
    parser.feed(page_result)
    parser.close()
    extracted = parser.get_text()
    if extracted:
        return extracted

    stripped = re.sub(r"<[^>]+>", " ", page_result)
    return _normalize_text(stripped)


def _build_generation_gt(answer: str | None, alt_ans: list[str] | None) -> list[str] | None:
    deduped_answers: list[str] = []
    seen: set[str] = set()

    for candidate in [answer, *(alt_ans or [])]:
        normalized = _normalize_text(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_answers.append(normalized)

    return deduped_answers or None


def _format_search_result_contents(search_result: dict[str, Any]) -> str:
    title = _normalize_text(search_result.get("page_name"))
    url = _normalize_text(search_result.get("page_url"))
    snippet = _normalize_text(search_result.get("page_snippet"))
    last_modified = _normalize_text(search_result.get("page_last_modified"))
    page_text = _extract_page_text(search_result.get("page_result"))
    content = page_text or snippet

    sections = [
        ("Title", title),
        ("URL", url),
        ("Snippet", snippet),
        ("Last Modified", last_modified),
        ("Content", content),
    ]
    return "\n".join(f"{label}: {value}" for label, value in sections if value)


@register_ingestor(
    name="crag",
    description="CRAG benchmark for generation-oriented RAG evaluation using search-result pages",
)
class CRAGIngestor(TextEmbeddingDataIngestor):
    def __init__(self, embedding_model: Embeddings, batch_size: int = DEFAULT_BATCH_SIZE):
        super().__init__(embedding_model)
        self.batch_size = batch_size

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        return "string"

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
    ) -> None:
        if self.service is None:
            raise ServiceNotSetError

        target_split = _resolve_subset(subset)
        if subset == "train":
            logger.warning(
                "CRAG does not publish a train split in the supported source file; using the dev split instead."
            )
        if min_corpus_cnt is not None:
            logger.warning(
                "min_corpus_cnt is ineffective for CRAG. Each query ships with its own search results rather than a shared corpus."
            )

        dataset = load_dataset("json", data_files=CRAG_DATA_URL, split="train", streaming=True)

        batch: list[dict[str, Any]] = []
        total_processed = 0

        for example in dataset:
            example_dict: dict[str, Any] = example
            if example_dict.get("split") != target_split:
                continue

            batch.append(example_dict)

            if len(batch) >= self.batch_size:
                self._process_batch(subset, batch)
                total_processed += len(batch)
                batch = []
                logger.info(f"[crag:{subset}] Processed {total_processed} examples...")

            if query_limit is not None and total_processed + len(batch) >= query_limit:
                remaining = query_limit - total_processed
                batch = batch[:remaining]
                break

        if batch:
            self._process_batch(subset, batch)
            total_processed += len(batch)

        logger.info(f"[crag:{subset}] Total examples processed: {total_processed}")
        self.service.clean()
        logger.info("CRAG ingestion complete.")

    def _process_batch(self, subset: str, examples: list[dict[str, Any]]) -> None:
        if self.service is None:
            raise ServiceNotSetError

        queries: list[dict[str, str | list[str] | None]] = []
        chunks: list[dict[str, str | int | bool | None]] = []

        for example in examples:
            interaction_id = str(example["interaction_id"])
            queries.append({
                "id": _make_query_id(subset, interaction_id),
                "contents": example["query"],
                "generation_gt": _build_generation_gt(example.get("answer"), example.get("alt_ans")),
            })

            for result_index, search_result in enumerate(example.get("search_results", [])):
                chunk_contents = _format_search_result_contents(search_result)
                if not chunk_contents:
                    continue
                chunks.append({
                    "id": _make_chunk_id(subset, interaction_id, result_index),
                    "contents": chunk_contents,
                })

        if chunks:
            self.service.add_chunks(chunks)
        if queries:
            self.service.add_queries(queries)
