"""ViDoRe V1 Unified Ingestor for visual document QA benchmark.

This unified ingestor supports all 10 ViDoRe V1 benchmark datasets:
- arxivqa_test_subsampled: Academic paper figures with multiple choice questions
- docvqa_test_subsampled: Document visual QA (test set held out)
- infovqa_test_subsampled: Infographic visual QA (test set held out)
- tabfquad_test_subsampled: Table-based French QA (no answer field)
- tatdqa_test: Table and text document QA (empty string answers)
- shiftproject_test: Project reports QA (JSON list answers)
- syntheticDocQA_artificial_intelligence_test: Synthetic AI docs (JSON list answers)
- syntheticDocQA_energy_test: Synthetic energy docs (JSON list answers)
- syntheticDocQA_government_reports_test: Synthetic govt docs (JSON list answers)
- syntheticDocQA_healthcare_industry_test: Synthetic healthcare docs (JSON list answers)

All datasets follow 1:1 query-to-image mapping where each row is a query-image pair.
HuggingFace path pattern: vidore/{dataset_name}

Reference: https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d
"""

from __future__ import annotations

import ast
import json
import logging
import random
from typing import TYPE_CHECKING, Any, Literal, get_args

from datasets import load_dataset

from autorag_research.data.base import MultiModalEmbeddingDataIngestor
from autorag_research.data.registry import register_ingestor
from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding

if TYPE_CHECKING:
    from autorag_research.embeddings.base import SingleVectorMultiModalEmbedding
from autorag_research.exceptions import InvalidDatasetNameError, ServiceNotSetError
from autorag_research.util import pil_image_to_bytes

RANDOM_SEED = 42

logger = logging.getLogger("AutoRAG-Research")

VIDORE_V1_DATASETS = Literal[
    "arxivqa_test_subsampled",
    "docvqa_test_subsampled",
    "infovqa_test_subsampled",
    "tabfquad_test_subsampled",
    "tatdqa_test",
    "shiftproject_test",
    "syntheticDocQA_artificial_intelligence_test",
    "syntheticDocQA_energy_test",
    "syntheticDocQA_government_reports_test",
    "syntheticDocQA_healthcare_industry_test",
]


@register_ingestor(
    name="vidore",
    description="ViDoRe v1 visual document QA benchmark",
    hf_repo="vidore-dumps",
)
class ViDoReIngestor(MultiModalEmbeddingDataIngestor):
    """Unified ingestor for all ViDoRe V1 benchmark datasets.

    The datasets follow a 1:1 query-to-image mapping where each row is a
    query-image pair. Supports all 10 V1 datasets through the dataset_name
    constructor argument.

    Attributes:
        dataset_name: Which ViDoRe V1 dataset to use.
        embedding_model: MultiModal embedding model for single-vector embeddings.
        late_interaction_embedding_model: Multi-vector embedding model (e.g., ColPali).

    Example:
        >>> ingestor = ViDoReIngestor("arxivqa_test_subsampled")
        >>> ingestor.set_service(service)
        >>> ingestor.ingest(query_limit=100)
    """

    def __init__(
        self,
        dataset_name: VIDORE_V1_DATASETS,
        embedding_model: SingleVectorMultiModalEmbedding | None = None,
        late_interaction_embedding_model: MultiVectorMultiModalEmbedding | None = None,
    ):
        """Initialize ViDoReIngestor.

        Args:
            dataset_name: Name of the ViDoRe V1 dataset to ingest.
            embedding_model: Optional single-vector embedding model.
            late_interaction_embedding_model: Optional multi-vector embedding model.

        Raises:
            InvalidDatasetNameError: If dataset_name is not a valid ViDoRe V1 dataset.
        """
        super().__init__(embedding_model, late_interaction_embedding_model)
        if dataset_name not in list(get_args(VIDORE_V1_DATASETS)):
            raise InvalidDatasetNameError(dataset_name)
        self.dataset_name = dataset_name

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        """Detect the primary key type used in the dataset.

        Returns:
            Always returns "string" for ViDoRe datasets.
        """
        return "string"

    def _parse_answer(self, answer: Any) -> list[str] | None:
        """Parse answer field based on dataset-specific format.

        Handles five distinct answer formats across ViDoRe V1 datasets:
        1. tabfquad: no answer field exists -> None
        2. null/None (docvqa, infovqa - test set held out) -> None
        3. Empty string (tatdqa) -> None
        4. Single letter answer (arxivqa - A/B/C/D) -> [answer]
        5. JSON-stringified list (shiftproject, syntheticDocQA_*) -> parsed list

        Args:
            answer: Raw answer value from dataset.

        Returns:
            List of answer strings, or None if no valid answer.
        """
        # Case 1: tabfquad - no answer field exists (handled by caller)
        if self.dataset_name == "tabfquad_test_subsampled":
            return None

        # Case 2: null/None (docvqa, infovqa - test set held out)
        if answer is None:
            return None

        # Case 3: Empty string (tatdqa)
        if isinstance(answer, str) and answer.strip() == "":
            return None

        # Case 4: Single letter answer (arxivqa)
        if self.dataset_name == "arxivqa_test_subsampled":
            return [answer]

        # Case 5: JSON-stringified list (shiftproject, syntheticDocQA_*)
        if isinstance(answer, str):
            try:
                # Try JSON parsing first (e.g., '["answer1", "answer2"]')
                parsed = json.loads(answer)
                if isinstance(parsed, list):
                    return [str(a) for a in parsed]
                return [str(parsed)]
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse failed for answer, trying ast.literal_eval: {e}")

            try:
                # Fallback to ast.literal_eval for Python literal strings
                parsed = ast.literal_eval(answer)
                if isinstance(parsed, list):
                    return [str(a) for a in parsed]
                return [str(parsed)]
            except (ValueError, SyntaxError) as e:
                # If all parsing fails, treat as plain string
                logger.debug(f"ast.literal_eval failed, using raw string: {e}")
                return [answer]

        # Fallback: convert to string
        return [str(answer)]

    def _format_query(self, query: str, options_str: str | None) -> str:
        """Format query text with optional multiple-choice options.

        For arxivqa_test_subsampled, the query includes multiple-choice options.
        For all other datasets, the query is returned unchanged.

        Args:
            query: Raw query text.
            options_str: String representation of options (for arxivqa).

        Returns:
            Formatted query string.
        """
        if self.dataset_name != "arxivqa_test_subsampled":
            return query

        if options_str:
            try:
                options_list = ast.literal_eval(options_str)
                options_formatted = "\n".join(options_list)
            except (ValueError, SyntaxError):
                logger.warning(f"Failed to parse options: {options_str[:50]}...")
            else:
                return (
                    f"Given the following query and options, select the correct option.\n\n"
                    f"Query: {query}\n\n"
                    f"Options: {options_formatted}"
                )

        return query

    def _compute_effective_limit(self, query_limit: int | None, min_corpus_cnt: int | None) -> int | None:
        """Compute effective limit from query_limit and min_corpus_cnt.

        For V1 datasets with 1:1 query-image mapping, both limits are equivalent.
        Returns the minimum of the two if both are set.
        """
        if query_limit is not None and min_corpus_cnt is not None:
            return min(query_limit, min_corpus_cnt)
        return query_limit or min_corpus_cnt

    def _select_indices(self, total_count: int, effective_limit: int | None) -> list[int]:
        """Select indices for sampling.

        If effective_limit is less than total_count, randomly sample indices
        using a fixed seed for reproducibility.
        """
        if effective_limit is not None and effective_limit < total_count:
            rng = random.Random(RANDOM_SEED)
            return sorted(rng.sample(range(total_count), effective_limit))
        return list(range(total_count))

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
    ) -> None:
        """Ingest ViDoRe V1 dataset.

        For V1 datasets, each row is a query-image pair (1:1 mapping).
        Both query_limit and min_corpus_cnt effectively limit the same thing.

        Args:
            subset: Dataset split (ignored - V1 datasets only have 'test').
            query_limit: Maximum number of queries to ingest.
            min_corpus_cnt: Maximum number of corpus images to ingest.
                          Since 1:1 mapping, this is equivalent to query_limit.
        """
        super().ingest(subset, query_limit, min_corpus_cnt)

        if self.service is None:
            raise ServiceNotSetError

        # V1 datasets only have 'test' split
        if subset != "test":
            logger.warning(f"ViDoRe V1 datasets only have 'test' split, ignoring subset='{subset}'")

        # Load dataset (streaming disabled - random sampling requires knowing total count)
        logger.info(f"Loading vidore/{self.dataset_name}...")
        ds = load_dataset(f"vidore/{self.dataset_name}", split="test")
        total_count = len(ds)  # ty: ignore[invalid-argument-type]
        logger.info(f"Loaded dataset with {total_count} rows")

        # Calculate effective limit and select indices
        effective_limit = self._compute_effective_limit(query_limit, min_corpus_cnt)
        selected_indices = self._select_indices(total_count, effective_limit)

        logger.info(f"Selected {len(selected_indices)} items from {total_count} total")

        ds_subset = ds.select(selected_indices)

        # Extract data from selected subset
        image_list = list(ds_subset["image"])
        queries = list(ds_subset["query"])

        # Get options for arxivqa (if available)
        options_list: list[str | None] = (
            list(ds_subset["options"]) if self.dataset_name == "arxivqa_test_subsampled" else [None] * len(queries)
        )

        # Format queries (special handling for arxivqa)
        queries = [self._format_query(q, opt) for q, opt in zip(queries, options_list, strict=True)]

        # Parse answers (dataset-specific)
        has_answer_field = self.dataset_name != "tabfquad_test_subsampled"
        if has_answer_field:
            raw_answers = list(ds_subset["answer"])
            answers: list[list[str] | None] = [self._parse_answer(ans) for ans in raw_answers]
        else:
            answers = [None] * len(queries)

        # Convert PIL images to bytes
        image_bytes_list = [pil_image_to_bytes(img) for img in image_list]

        # Add queries
        query_data: list[dict[str, Any]] = []
        for idx, (query, ans) in enumerate(zip(queries, answers, strict=True)):
            qd: dict[str, Any] = {
                "id": str(selected_indices[idx]),
                "contents": query,
            }
            if ans is not None:
                qd["generation_gt"] = ans
            query_data.append(qd)

        query_pk_list = self.service.add_queries(query_data)

        # Add image chunks (using same indices as IDs)
        image_chunk_data = [
            {
                "id": str(selected_indices[idx]),
                "contents": content,
                "mimetype": mimetype,
            }
            for idx, (content, mimetype) in enumerate(image_bytes_list)
        ]
        image_chunk_pk_list = self.service.add_image_chunks(image_chunk_data)

        # Create 1:1 retrieval relations
        self.service.add_retrieval_gt_batch(
            list(zip(query_pk_list, image_chunk_pk_list, strict=True)),
            chunk_type="image",
        )

        logger.info(
            f"[{self.dataset_name}] Ingestion complete: {len(query_pk_list)} queries, {len(image_chunk_pk_list)} images"
        )
