"""Qasper dataset ingestor for AutoRAG-Research.

Qasper is a question-answering dataset over NLP research papers from arXiv.
Each paper has multiple questions with answers from multiple annotators.

Dataset: https://huggingface.co/datasets/allenai/qasper

Two operation modes:
- answerable: Only answerable questions with retrieval GT (for Retrieval + Generation eval)
- full: All questions including unanswerable (for Generation eval only)
"""

import logging
import random
from typing import Any, Literal

from datasets import load_dataset
from llama_index.core.base.embeddings.base import BaseEmbedding

from autorag_research.data.base import TextEmbeddingDataIngestor
from autorag_research.data.registry import register_ingestor
from autorag_research.data.util import make_id
from autorag_research.exceptions import ServiceNotSetError
from autorag_research.orm.models import or_all
from autorag_research.util import normalize_string

logger = logging.getLogger("AutoRAG-Research")

RANDOM_SEED = 42

QASPER_QA_MODE = Literal["answerable", "full"]

SPLIT_MAPPING = {
    "train": "train",
    "dev": "validation",
    "test": "test",
}


def _iter_paper_chunks(paper_id: str, paragraphs: list[list[str]]):
    """Iterate over paper paragraphs and yield (chunk_id, paragraph) pairs.

    Args:
        paper_id: arXiv paper ID.
        paragraphs: List of section paragraph lists from full_text["paragraphs"].

    Yields:
        Tuple of (chunk_id, stripped_paragraph) for each non-empty paragraph.
    """
    for section_idx, section_paragraphs in enumerate(paragraphs):
        for para_idx, paragraph in enumerate(section_paragraphs):
            if stripped := (paragraph or "").strip():
                chunk_id = make_id("qasper", paper_id, f"s{section_idx}", f"p{para_idx}")
                yield chunk_id, stripped


def _build_evidence_to_chunk_map(paper_id: str, full_text: dict) -> dict[str, str]:
    """Build a mapping from normalized evidence text to chunk IDs.

    This enables matching evidence strings to the chunks they came from.

    Args:
        paper_id: arXiv paper ID.
        full_text: Paper's full_text dict with section_name and paragraphs.

    Returns:
        Dict mapping normalized paragraph text to chunk ID.
    """
    return {
        normalize_string(para): chunk_id
        for chunk_id, para in _iter_paper_chunks(paper_id, full_text.get("paragraphs", []))
    }


def _find_chunk_ids_for_evidence(
    evidence_list: list[str],
    evidence_map: dict[str, str],
) -> list[str]:
    """Find chunk IDs that correspond to evidence passages.

    Supports both exact and partial matching since evidence may be
    a subset of or slightly differ from the original paragraph.

    Args:
        evidence_list: List of evidence text strings.
        evidence_map: Mapping from normalized text to chunk ID.

    Returns:
        List of unique chunk IDs matching the evidence.
    """
    chunk_ids: set[str] = set()

    for evidence in evidence_list:
        if not evidence or not evidence.strip():
            continue

        normalized = normalize_string(evidence)

        # Try exact match first
        if normalized in evidence_map:
            chunk_ids.add(evidence_map[normalized])
            continue

        # Try partial matching (evidence is substring of chunk or vice versa)
        for key, chunk_id in evidence_map.items():
            if normalized in key or key in normalized:
                chunk_ids.add(chunk_id)
                break

    return list(chunk_ids)


def _extract_generation_gt(answers: list[dict]) -> list[str]:
    """Extract generation ground truth from all annotators' answers.

    Collects unique answers from all annotators, handling:
    - extractive_spans: Direct text spans from paper
    - free_form_answer: Synthesized/summarized answers
    - yes_no: Boolean answers converted to "yes"/"no"

    Unanswerable answers are skipped.

    Args:
        answers: List of annotator answer dicts from qas.answers.

    Returns:
        List of unique ground truth answer strings.
    """
    gt_answers: set[str] = set()

    for annotator in answers:
        answer_data = annotator.get("answer", {})

        # Skip unanswerable
        if answer_data.get("unanswerable"):
            continue

        # Extractive spans
        extractive_spans = answer_data.get("extractive_spans", [])
        if extractive_spans:
            for span in extractive_spans:
                if span and span.strip():
                    gt_answers.add(span)

        # Free-form answer
        free_form = answer_data.get("free_form_answer", "")
        if free_form and free_form.strip():
            gt_answers.add(free_form)

        # Yes/No answer
        yes_no = answer_data.get("yes_no")
        if yes_no is not None:
            gt_answers.add("yes" if yes_no else "no")

    return list(gt_answers)


def _is_answerable(answers: list[dict]) -> bool:
    """Check if any annotator marked the question as answerable.

    A question is considered answerable if at least one annotator
    did not mark it as unanswerable.

    Args:
        answers: List of annotator answer dicts.

    Returns:
        True if any annotator says answerable, False if all say unanswerable.
    """
    return any(not annotator.get("answer", {}).get("unanswerable") for annotator in answers)


def _collect_all_evidence(answers: list[dict]) -> list[str]:
    """Collect all evidence passages from all annotators.

    Args:
        answers: List of annotator answer dicts.

    Returns:
        List of all evidence strings (may contain duplicates).
    """
    all_evidence: list[str] = []
    for annotator in answers:
        answer_data = annotator.get("answer", {})
        evidence = answer_data.get("evidence", [])
        all_evidence.extend(evidence)
    return all_evidence


def _process_question(
    paper_id: str,
    question: str,
    question_id: str,
    answers: list[dict],
    evidence_map: dict[str, str],
    qa_mode: str,
) -> dict | None:
    """Process a single question and return query metadata if valid.
       (A function created to reduce function complexity as requested by Ruff)

    Args:
        paper_id: Paper ID.
        question: Question text.
        question_id: Question ID.
        answers: Raw answers list from dataset.
        evidence_map: Mapping from normalized text to chunk ID.
        qa_mode: "answerable" or "full" mode.

    Returns:
        Query metadata dict or None if question should be skipped.
    """
    if not answers:
        return None

    annotator_answers = [{"answer": a} for a in answers]
    is_answerable = _is_answerable(annotator_answers)

    if qa_mode == "answerable" and not is_answerable:
        return None

    query_id = make_id("qasper", paper_id, question_id)
    generation_gt = _extract_generation_gt(annotator_answers)

    if not is_answerable:
        generation_gt = ["unanswerable"]

    gold_chunk_ids: list[str] = []
    if qa_mode == "answerable" and is_answerable:
        all_evidence = _collect_all_evidence(annotator_answers)
        gold_chunk_ids = _find_chunk_ids_for_evidence(all_evidence, evidence_map)

        if not gold_chunk_ids:
            logger.debug(f"Skipping query {question_id} in paper {paper_id}: no evidence matched any paragraph")
            return None

    return {
        "query_id": query_id,
        "paper_id": paper_id,
        "question": question,
        "question_id": question_id,
        "generation_gt": generation_gt,
        "gold_chunk_ids": gold_chunk_ids,
        "annotator_answers": annotator_answers,
    }


@register_ingestor(
    name="qasper",
    description="Qasper QA dataset over NLP research papers",
)
class QasperIngestor(TextEmbeddingDataIngestor):
    """Ingestor for the Qasper dataset.

    Qasper contains NLP research papers with question-answer pairs.
    Each paper becomes multiple chunks (paragraphs), and each question
    becomes a query with evidence-based retrieval ground truth.

    Two modes:
    - answerable: Only includes answerable questions with retrieval GT
    - full: Includes all questions (unanswerable get "unanswerable" as GT)
    """

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        qa_mode: QASPER_QA_MODE = "answerable",
        batch_size: int = 100,
    ):
        """Initialize Qasper ingestor.

        Args:
            embedding_model: Embedding model for text.
            qa_mode: Operation mode - "answerable" (default) or "full".
            batch_size: Batch size for processing.
        """
        super().__init__(embedding_model)
        self.qa_mode = qa_mode
        self.batch_size = batch_size

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        """Qasper uses string IDs (arXiv paper IDs like '1909.00694')."""
        return "string"

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
    ) -> None:
        """Ingest Qasper dataset.

        Args:
            subset: Dataset split (train/dev/test).
            query_limit: Maximum number of queries to ingest.
            min_corpus_cnt: Maximum number of chunks to ingest.
                           Gold chunks from selected queries are always included.
        """
        if self.service is None:
            raise ServiceNotSetError

        hf_split = SPLIT_MAPPING[subset]
        logger.info(f"Loading Qasper dataset split: {hf_split}")

        ds = load_dataset("allenai/qasper", split=hf_split, trust_remote_code=True)

        # Single pass: collect all queries and paper chunks
        query_metadata, paper_chunks = self._collect_all_data(ds)
        logger.info(f"Found {len(query_metadata)} {'answerable' if self.qa_mode == 'answerable' else 'total'} queries")

        # Sample queries if limit specified
        rng = random.Random(RANDOM_SEED)  # noqa: S311
        if query_limit is not None and query_limit < len(query_metadata):
            query_metadata = rng.sample(query_metadata, query_limit)
            logger.info(f"Sampled {len(query_metadata)} queries")

        # Collect gold chunk IDs from selected queries
        gold_chunk_ids: set[str] = set()
        for qm in query_metadata:
            gold_chunk_ids.update(qm["gold_chunk_ids"])
        logger.info(f"Gold chunks from selected queries: {len(gold_chunk_ids)}")

        # Collect chunks from papers with selected queries
        paper_ids_needed = {qm["paper_id"] for qm in query_metadata}
        all_chunks: dict[str, str] = {}
        for paper_id in paper_ids_needed:
            all_chunks.update(paper_chunks.get(paper_id, {}))
        logger.info(f"Total chunks from selected papers: {len(all_chunks)}")

        # Filter chunks if min_corpus_cnt specified
        if min_corpus_cnt is not None:
            selected_chunk_ids = self._filter_chunks(
                list(all_chunks.keys()),
                gold_chunk_ids,
                min_corpus_cnt,
                rng,
            )
            all_chunks = {cid: all_chunks[cid] for cid in selected_chunk_ids if cid in all_chunks}
            logger.info(f"Selected {len(all_chunks)} chunks after filtering")

        # Ingest chunks
        self._ingest_chunks(all_chunks)

        # Ingest queries and relations
        chunk_ids_set = set(all_chunks.keys())
        self._ingest_queries_and_relations(query_metadata, chunk_ids_set)

        self.service.clean()
        logger.info("Qasper ingestion complete.")

    def _collect_all_data(self, ds: Any) -> tuple[list[dict], dict[str, dict[str, str]]]:
        """Collect query metadata and paper chunks in a single pass.

        Performs a single iteration over the dataset to collect both
        query metadata and chunk content, avoiding redundant I/O.

        Args:
            ds: HuggingFace dataset.

        Returns:
            Tuple of:
            - List of query metadata dicts
            - Dict mapping paper_id to {chunk_id: content}
        """
        query_metadata: list[dict] = []
        paper_chunks: dict[str, dict[str, str]] = {}

        for paper in ds:
            paper_id = paper["id"]
            full_text = paper["full_text"]
            qas = paper["qas"]

            # Collect chunks for this paper
            paper_chunks[paper_id] = dict(_iter_paper_chunks(paper_id, full_text.get("paragraphs", [])))

            # Build evidence map for this paper
            evidence_map = _build_evidence_to_chunk_map(paper_id, full_text)

            # Validate and process questions
            questions = qas.get("question", [])
            question_ids = qas.get("question_id", [])
            answers_list = qas.get("answers", [])

            if len(questions) != len(question_ids):
                logger.warning(
                    f"Paper {paper_id}: question/question_id length mismatch "
                    f"({len(questions)} vs {len(question_ids)}). Skipping paper."
                )
                continue

            for i, (question, question_id) in enumerate(zip(questions, question_ids, strict=True)):
                if i >= len(answers_list):
                    continue

                answers = answers_list[i].get("answer", [])
                qm = _process_question(paper_id, question, question_id, answers, evidence_map, self.qa_mode)
                if qm:
                    query_metadata.append(qm)

        return query_metadata, paper_chunks

    def _filter_chunks(
        self,
        all_chunk_ids: list[str],
        gold_chunk_ids: set[str],
        min_corpus_cnt: int,
        rng: random.Random,
    ) -> list[str]:
        """Filter chunks to meet the minimum corpus count.

        Always includes gold chunks, then adds random samples to reach target.

        Args:
            all_chunk_ids: List of all available chunk IDs.
            gold_chunk_ids: Set of chunk IDs that must be included.
            min_corpus_cnt: Target number of chunks.
            rng: Random number generator.

        Returns:
            List of selected chunk IDs.
        """
        # Start with gold chunks
        selected = list(gold_chunk_ids & set(all_chunk_ids))

        if len(selected) >= min_corpus_cnt:
            return selected

        # Add random samples to reach target
        remaining = [cid for cid in all_chunk_ids if cid not in gold_chunk_ids]
        additional_needed = min_corpus_cnt - len(selected)
        additional = rng.sample(remaining, min(additional_needed, len(remaining)))
        selected.extend(additional)

        return selected

    def _ingest_chunks(self, chunks: dict[str, str]) -> None:
        """Ingest chunks into the database.

        Args:
            chunks: Dict mapping chunk_id to chunk content.
        """
        if self.service is None:
            raise ServiceNotSetError

        chunks_to_add = [{"id": cid, "contents": content} for cid, content in chunks.items()]

        # Process in batches
        for i in range(0, len(chunks_to_add), self.batch_size):
            batch = chunks_to_add[i : i + self.batch_size]
            self.service.add_chunks(batch)
            logger.info(f"Ingested {min(i + self.batch_size, len(chunks_to_add))}/{len(chunks_to_add)} chunks")

    def _ingest_queries_and_relations(
        self,
        query_metadata: list[dict],
        chunk_ids_set: set[str],
    ) -> None:
        """Ingest queries and their retrieval relations.

        Args:
            query_metadata: List of query metadata dicts.
            chunk_ids_set: Set of ingested chunk IDs for validation.
        """
        if self.service is None:
            raise ServiceNotSetError

        queries_to_add: list[dict] = []

        for qm in query_metadata:
            queries_to_add.append({
                "id": qm["query_id"],
                "contents": qm["question"],
                "generation_gt": qm["generation_gt"] if qm["generation_gt"] else None,
            })

        # Add queries in batches
        for i in range(0, len(queries_to_add), self.batch_size):
            batch = queries_to_add[i : i + self.batch_size]
            self.service.add_queries(batch)
            logger.info(f"Ingested {min(i + self.batch_size, len(queries_to_add))}/{len(queries_to_add)} queries")

        # Add retrieval relations (only in answerable mode)
        if self.qa_mode == "answerable":
            relations_added = 0
            for qm in query_metadata:
                # Filter to only include chunks that were actually ingested
                valid_chunk_ids = [cid for cid in qm["gold_chunk_ids"] if cid in chunk_ids_set]

                if valid_chunk_ids:
                    self.service.add_retrieval_gt(qm["query_id"], or_all(valid_chunk_ids), chunk_type="text")
                    relations_added += 1

            logger.info(f"Added retrieval relations for {relations_added} queries")
