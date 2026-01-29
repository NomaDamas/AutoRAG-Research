"""Common test utilities for data ingestor integration tests.

This module provides reusable utilities for testing data ingestors against
real PostgreSQL databases using actual dataset subsets. It replaces mock-based
tests with deterministic integration tests.

Usage:
    from tests.autorag_research.data.ingestor_test_utils import (
        IngestorTestConfig,
        create_test_database,
        IngestorTestVerifier,
    )

    CONFIG = IngestorTestConfig(
        expected_query_count=10,
        expected_chunk_count=50,
        check_retrieval_relations=True,
        primary_key_type="string",
    )

    @pytest.mark.data
    def test_my_ingestor():
        with create_test_database(CONFIG) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)
            ingestor = MyIngestor(MockEmbedding(768))
            ingestor.set_service(service)
            ingestor.ingest(query_limit=10, min_corpus_cnt=50)

            verifier = IngestorTestVerifier(service, db.schema, CONFIG)
            verifier.verify_all()
"""

import hashlib
import io
import logging
import random
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from PIL import Image
from sqlalchemy.engine import Engine
from sqlalchemy.orm import scoped_session

from autorag_research.orm.connection import DBConnection
from autorag_research.orm.service.base_ingestion import BaseIngestionService
from tests.util import CheckResult, VerificationReport

# Environment variables are loaded by conftest.py via load_dotenv()

logger = logging.getLogger("AutoRAG-Research")

RANDOM_SEED = 42


@dataclass
class ExpectedContentHash:
    """Expected content hash for verification."""

    record_id: str | int
    content_hash: str  # MD5 hash of expected content
    record_type: Literal["query", "chunk", "image_chunk"]


@dataclass
class IngestorTestConfig:
    """Configuration for ingestor integration tests."""

    # Required counts
    expected_query_count: int
    expected_chunk_count: int | None = None  # For text datasets
    expected_image_chunk_count: int | None = None  # For multi-modal datasets

    # Count verification mode
    # When True, chunk/image_chunk counts are verified as >= expected (for subset tests where gold IDs are always included)
    # When False (default), counts are verified as == expected
    chunk_count_is_minimum: bool = False

    # Optional structure checks
    check_documents: bool = False
    expected_document_count: int | None = None
    check_pages: bool = False
    expected_page_count: int | None = None
    check_files: bool = False
    expected_file_count: int | None = None

    # Relation checks
    check_retrieval_relations: bool = True
    check_generation_gt: bool = False
    # When True, ALL queries must have generation_gt (not just some)
    generation_gt_required_for_all: bool = False

    # Content verification
    content_hashes: list[ExpectedContentHash] = field(default_factory=list)

    # Database settings
    db_name: str = "ingestor_test_db"
    primary_key_type: Literal["bigint", "string"] = "string"
    embedding_dim: int = 768


@dataclass
class TestDatabaseContext:
    """Context object yielded by create_test_database."""

    schema: Any
    engine: Engine
    session_factory: scoped_session


@contextmanager
def create_test_database(config: IngestorTestConfig) -> Generator[TestDatabaseContext, None, None]:
    """Create isolated test database, yield components, cleanup.

    Args:
        config: Test configuration with database settings.

    Yields:
        TestDatabaseContext with schema, engine, and session_factory.
    """
    conn = DBConnection.from_env()
    conn.database = config.db_name
    conn.create_database()

    # Create schema with explicit embedding_dim and primary_key_type from config
    schema = conn.create_schema(config.embedding_dim, config.primary_key_type)
    engine = conn.get_engine()
    session_factory = conn.get_session_factory()

    try:
        yield TestDatabaseContext(
            schema=schema,
            engine=engine,
            session_factory=session_factory,
        )
    finally:
        engine.dispose()
        conn.terminate_connections()
        conn.drop_database()


class IngestorTestVerifier:
    """Verifier for data ingestor integration tests.

    Provides verification methods for:
    - Count verification (queries, chunks, image_chunks, etc.)
    - Random sample format validation
    - Retrieval relation verification
    - Generation ground truth verification
    - Content hash verification
    - Detailed logging for CI inspection
    """

    def __init__(
        self,
        service: BaseIngestionService,
        schema: Any,
        config: IngestorTestConfig,
    ):
        """Initialize verifier.

        Args:
            service: The ingestion service used for verification.
            schema: The schema namespace from create_schema().
            config: Test configuration with expected values.
        """
        self.service = service
        self.schema = schema
        self.config = config
        self._rng = random.Random(RANDOM_SEED)

    def verify_all(self) -> VerificationReport:
        """Run all configured checks and return detailed report.

        Also logs sample content for CI inspection.

        Returns:
            VerificationReport with all check results.

        Raises:
            AssertionError: If any check fails.
        """
        report = VerificationReport()

        # 1. Count verification
        self._add_count_checks(report)

        # 2. Optional structure checks
        self._add_structure_checks(report)

        # 3. Random sample format validation
        self._add_format_checks(report)

        # 4. Relation checks
        self._add_relation_checks(report)

        # 5. Content hash verification
        if self.config.content_hashes:
            report.add_check("content_hashes", self._verify_content_hashes())

        # 6. Log sample content for CI
        self._log_sample_content()

        # Log summary
        logger.info(report.summary())

        # Assert all passed
        if not report.all_passed:
            failed_checks = [name for name, check in report.checks.items() if not check.passed]
            msg = f"Verification failed for checks: {failed_checks}\n{report.summary()}"
            raise AssertionError(msg)

        return report

    def _add_count_checks(self, report: VerificationReport) -> None:
        """Add count verification checks to report."""
        report.add_check("query_count", self._verify_query_count())
        if self.config.expected_chunk_count is not None:
            report.add_check("chunk_count", self._verify_chunk_count())
        if self.config.expected_image_chunk_count is not None:
            report.add_check("image_chunk_count", self._verify_image_chunk_count())

    def _add_structure_checks(self, report: VerificationReport) -> None:
        """Add optional structure checks to report."""
        if self.config.check_documents:
            report.add_check("document_count", self._verify_document_count())
        if self.config.check_pages:
            report.add_check("page_count", self._verify_page_count())
        if self.config.check_files:
            report.add_check("file_count", self._verify_file_count())

    def _add_format_checks(self, report: VerificationReport) -> None:
        """Add format validation checks to report."""
        report.add_check("query_format", self._verify_query_format_random_sample())
        if self.config.expected_chunk_count is not None:
            report.add_check("chunk_format", self._verify_chunk_format_random_sample())
        if self.config.expected_image_chunk_count is not None:
            report.add_check("image_chunk_format", self._verify_image_chunk_format_random_sample())

    def _add_relation_checks(self, report: VerificationReport) -> None:
        """Add relation verification checks to report."""
        if self.config.check_retrieval_relations:
            report.add_check("retrieval_relations", self._verify_retrieval_relations())
        if self.config.check_generation_gt:
            report.add_check("generation_gt", self._verify_generation_gt())

    # ==================== Count Verification ====================

    def _verify_query_count(self) -> CheckResult:
        """Verify query count matches expected."""
        stats = self.service.get_statistics()
        actual = stats["queries"]["total"] if isinstance(stats["queries"], dict) else stats["queries"]
        expected = self.config.expected_query_count
        passed = actual == expected
        return CheckResult(
            passed=passed,
            message=f"Expected {expected}, got {actual}",
            failures=[] if passed else [f"Query count mismatch: expected {expected}, got {actual}"],
        )

    def _verify_chunk_count(self) -> CheckResult:
        """Verify chunk count matches expected."""
        stats = self.service.get_statistics()
        actual = stats["chunks"]["total"]
        expected = self.config.expected_chunk_count
        if self.config.chunk_count_is_minimum:
            passed = actual >= expected  # type: ignore[operator]
            comparison = ">="
        else:
            passed = actual == expected
            comparison = "=="
        return CheckResult(
            passed=passed,
            message=f"Expected {comparison} {expected}, got {actual}",
            failures=[] if passed else [f"Chunk count mismatch: expected {comparison} {expected}, got {actual}"],
        )

    def _verify_image_chunk_count(self) -> CheckResult:
        """Verify image chunk count matches expected."""
        stats = self.service.get_statistics()
        actual = stats["image_chunks"]["total"]
        expected = self.config.expected_image_chunk_count
        if self.config.chunk_count_is_minimum:
            passed = actual >= expected  # type: ignore[operator]
            comparison = ">="
        else:
            passed = actual == expected
            comparison = "=="
        return CheckResult(
            passed=passed,
            message=f"Expected {comparison} {expected}, got {actual}",
            failures=[] if passed else [f"Image chunk count mismatch: expected {comparison} {expected}, got {actual}"],
        )

    def _verify_document_count(self) -> CheckResult:
        """Verify document count matches expected."""
        stats = self.service.get_statistics()
        actual = stats.get("documents", 0)
        expected = self.config.expected_document_count or 0
        if self.config.chunk_count_is_minimum:
            passed = actual >= expected
            comparison = ">="
        else:
            passed = actual == expected
            comparison = "=="
        return CheckResult(
            passed=passed,
            message=f"Expected {comparison} {expected}, got {actual}",
            failures=[] if passed else [f"Document count mismatch: expected {comparison} {expected}, got {actual}"],
        )

    def _verify_page_count(self) -> CheckResult:
        """Verify page count matches expected."""
        stats = self.service.get_statistics()
        actual = stats.get("pages", 0)
        expected = self.config.expected_page_count or 0
        if self.config.chunk_count_is_minimum:
            passed = actual >= expected
            comparison = ">="
        else:
            passed = actual == expected
            comparison = "=="
        return CheckResult(
            passed=passed,
            message=f"Expected {comparison} {expected}, got {actual}",
            failures=[] if passed else [f"Page count mismatch: expected {comparison} {expected}, got {actual}"],
        )

    def _verify_file_count(self) -> CheckResult:
        """Verify file count matches expected."""
        stats = self.service.get_statistics()
        actual = stats.get("files", 0)
        expected = self.config.expected_file_count or 0
        if self.config.chunk_count_is_minimum:
            passed = actual >= expected
            comparison = ">="
        else:
            passed = actual == expected
            comparison = "=="
        return CheckResult(
            passed=passed,
            message=f"Expected {comparison} {expected}, got {actual}",
            failures=[] if passed else [f"File count mismatch: expected {comparison} {expected}, got {actual}"],
        )

    # ==================== Format Validation ====================

    def _verify_query_format_random_sample(self, sample_size: int = 5) -> CheckResult:
        """Randomly sample queries and validate format/typing."""
        failures: list[str] = []

        with self.service._create_uow() as uow:
            # Get all query IDs and sample randomly
            all_queries = uow.queries.get_all(limit=1000)
            if not all_queries:
                return CheckResult(passed=False, message="No queries found", failures=["No queries to sample"])

            sample = self._rng.sample(all_queries, min(sample_size, len(all_queries)))

            for query in sample:
                # Check ID type
                expected_type = str if self.config.primary_key_type == "string" else int
                if not isinstance(query.id, expected_type):
                    failures.append(f"Query {query.id}: ID type mismatch, expected {expected_type.__name__}")

                # Check contents is non-empty string
                if not isinstance(query.contents, str) or not query.contents.strip():
                    failures.append(f"Query {query.id}: contents is empty or not a string")

                # Check generation_gt is list[str] or None
                if query.generation_gt is not None:
                    if not isinstance(query.generation_gt, list):
                        failures.append(f"Query {query.id}: generation_gt is not a list")
                    elif not all(isinstance(gt, str) for gt in query.generation_gt):
                        failures.append(f"Query {query.id}: generation_gt contains non-string values")

        passed = len(failures) == 0
        return CheckResult(
            passed=passed,
            message=f"Sampled {len(sample)} queries",
            failures=failures,
        )

    def _verify_chunk_format_random_sample(self, sample_size: int = 5) -> CheckResult:
        """Randomly sample chunks and validate format/typing."""
        failures: list[str] = []

        with self.service._create_uow() as uow:
            all_chunks = uow.chunks.get_all(limit=1000)
            if not all_chunks:
                return CheckResult(passed=False, message="No chunks found", failures=["No chunks to sample"])

            sample = self._rng.sample(all_chunks, min(sample_size, len(all_chunks)))

            for chunk in sample:
                # Check ID type
                expected_type = str if self.config.primary_key_type == "string" else int
                if not isinstance(chunk.id, expected_type):
                    failures.append(f"Chunk {chunk.id}: ID type mismatch, expected {expected_type.__name__}")

                # Check contents is non-empty string
                if not isinstance(chunk.contents, str) or not chunk.contents.strip():
                    failures.append(f"Chunk {chunk.id}: contents is empty or not a string")

        passed = len(failures) == 0
        return CheckResult(
            passed=passed,
            message=f"Sampled {len(sample)} chunks",
            failures=failures,
        )

    def _verify_image_chunk_format_random_sample(self, sample_size: int = 5) -> CheckResult:
        """Randomly sample image chunks and validate format/typing."""
        failures: list[str] = []

        with self.service._create_uow() as uow:
            all_image_chunks = uow.image_chunks.get_all(limit=1000)
            if not all_image_chunks:
                return CheckResult(
                    passed=False, message="No image chunks found", failures=["No image chunks to sample"]
                )

            sample = self._rng.sample(all_image_chunks, min(sample_size, len(all_image_chunks)))

            for img_chunk in sample:
                # Check ID type
                expected_type = str if self.config.primary_key_type == "string" else int
                if not isinstance(img_chunk.id, expected_type):
                    failures.append(f"ImageChunk {img_chunk.id}: ID type mismatch")

                # Check contents is bytes (binary image data)
                if not isinstance(img_chunk.contents, bytes) or len(img_chunk.contents) == 0:
                    failures.append(f"ImageChunk {img_chunk.id}: contents is empty or not bytes")
                    continue

                # Check mimetype is set
                if not img_chunk.mimetype:
                    failures.append(f"ImageChunk {img_chunk.id}: mimetype is not set")

                # Verify bytes are valid image data (can be opened by PIL)
                try:
                    img = Image.open(io.BytesIO(img_chunk.contents))
                    if img.size[0] <= 0 or img.size[1] <= 0:
                        failures.append(f"ImageChunk {img_chunk.id}: invalid image dimensions")
                except Exception as e:
                    failures.append(f"ImageChunk {img_chunk.id}: cannot open as image - {e}")

        passed = len(failures) == 0
        return CheckResult(
            passed=passed,
            message=f"Sampled {len(sample)} image chunks",
            failures=failures,
        )

    # ==================== Relation Verification ====================

    def _verify_retrieval_relations(self) -> CheckResult:
        """Verify every query has at least one retrieval relation."""
        failures: list[str] = []

        with self.service._create_uow() as uow:
            all_queries = uow.queries.get_all(limit=1000)
            queries_without_relations = []

            for query in all_queries:
                relations = uow.retrieval_relations.get_by_query_id(query.id)
                if not relations:
                    queries_without_relations.append(query.id)

            if queries_without_relations:
                failures.append(
                    f"{len(queries_without_relations)} queries without retrieval relations: "
                    f"{queries_without_relations[:5]}..."
                )

        passed = len(failures) == 0
        return CheckResult(
            passed=passed,
            message=f"All {len(all_queries)} queries have relations" if passed else "Some queries missing relations",
            failures=failures,
        )

    def _verify_generation_gt(self) -> CheckResult:
        """Verify queries that should have generation_gt have non-null values."""
        failures: list[str] = []

        with self.service._create_uow() as uow:
            all_queries = uow.queries.get_all(limit=1000)
            queries_with_gt = [q for q in all_queries if q.generation_gt is not None]

            if not queries_with_gt:
                failures.append("No queries have generation_gt set")
            else:
                # Verify generation_gt values are valid
                for query in queries_with_gt:
                    if not query.generation_gt or len(query.generation_gt) == 0:
                        failures.append(f"Query {query.id}: generation_gt is empty list")

            # If required for all, check that every query has generation_gt
            if self.config.generation_gt_required_for_all:
                queries_without_gt = [q for q in all_queries if q.generation_gt is None]
                if queries_without_gt:
                    failures.append(
                        f"{len(queries_without_gt)} queries missing generation_gt: "
                        f"{[q.id for q in queries_without_gt[:5]]}..."
                    )

        passed = len(failures) == 0
        return CheckResult(
            passed=passed,
            message=f"{len(queries_with_gt)}/{len(all_queries)} queries have generation_gt",
            failures=failures,
        )

    # ==================== Content Hash Verification ====================

    def _verify_content_hashes(self) -> CheckResult:
        """Verify content matches expected hashes."""
        failures: list[str] = []

        with self.service._create_uow() as uow:
            for exp in self.config.content_hashes:
                if exp.record_type == "query":
                    record = uow.queries.get_by_id(exp.record_id)
                    if record is None:
                        failures.append(f"Query {exp.record_id} not found")
                        continue
                    content = record.contents
                elif exp.record_type == "chunk":
                    record = uow.chunks.get_by_id(exp.record_id)
                    if record is None:
                        failures.append(f"Chunk {exp.record_id} not found")
                        continue
                    content = record.contents
                elif exp.record_type == "image_chunk":
                    record = uow.image_chunks.get_by_id(exp.record_id)
                    if record is None:
                        failures.append(f"ImageChunk {exp.record_id} not found")
                        continue
                    # For image chunks, hash the binary content
                    content = record.contents if isinstance(record.contents, bytes) else record.contents.encode()
                else:
                    failures.append(f"Unknown record type: {exp.record_type}")
                    continue

                # Compute hash (MD5 is fine for content verification, not cryptographic use)
                if isinstance(content, bytes):
                    actual_hash = hashlib.md5(content).hexdigest()  # noqa: S324
                else:
                    actual_hash = hashlib.md5(content.encode()).hexdigest()  # noqa: S324

                if actual_hash != exp.content_hash:
                    failures.append(
                        f"{exp.record_type} {exp.record_id}: hash mismatch - "
                        f"expected {exp.content_hash}, got {actual_hash}"
                    )

        passed = len(failures) == 0
        return CheckResult(
            passed=passed,
            message=f"Verified {len(self.config.content_hashes)} content hashes",
            failures=failures,
        )

    # ==================== Logging for CI ====================

    def _log_sample_content(self, n: int = 3) -> None:
        """Log sample content for CI inspection."""
        with self.service._create_uow() as uow:
            all_queries = uow.queries.get_all(limit=100)
            self._log_sample_queries(all_queries, n)
            self._log_sample_chunks(uow, n)
            self._log_sample_image_chunks(uow, n)
            self._log_sample_relations(uow, all_queries)

    def _log_sample_queries(self, all_queries: list, n: int) -> None:
        """Log sample queries."""
        if not all_queries:
            return
        sample_queries = self._rng.sample(all_queries, min(n, len(all_queries)))
        logger.info("=== Sample Queries ===")
        for q in sample_queries:
            content_preview = q.contents[:200] + "..." if len(q.contents) > 200 else q.contents
            logger.info(f"Query [{q.id}]: {content_preview}")
            if q.generation_gt:
                gt_preview = str(q.generation_gt[:2]) + "..." if len(q.generation_gt) > 2 else str(q.generation_gt)
                logger.info(f"  Generation GT: {gt_preview}")

    def _log_sample_chunks(self, uow: Any, n: int) -> None:
        """Log sample text chunks."""
        if not self.config.expected_chunk_count:
            return
        all_chunks = uow.chunks.get_all(limit=100)
        if not all_chunks:
            return
        sample_chunks = self._rng.sample(all_chunks, min(n, len(all_chunks)))
        logger.info("=== Sample Chunks ===")
        for c in sample_chunks:
            content_preview = c.contents[:200] + "..." if len(c.contents) > 200 else c.contents
            logger.info(f"Chunk [{c.id}]: {content_preview}")

    def _log_sample_image_chunks(self, uow: Any, n: int) -> None:
        """Log sample image chunks."""
        if not self.config.expected_image_chunk_count:
            return
        all_image_chunks = uow.image_chunks.get_all(limit=100)
        if not all_image_chunks:
            return
        sample_image_chunks = self._rng.sample(all_image_chunks, min(n, len(all_image_chunks)))
        logger.info("=== Sample Image Chunks ===")
        for ic in sample_image_chunks:
            content_size = len(ic.contents) if ic.contents else 0
            logger.info(f"ImageChunk [{ic.id}]: {content_size} bytes, mimetype={ic.mimetype}")

    def _log_sample_relations(self, uow: Any, all_queries: list) -> None:
        """Log sample retrieval relations."""
        if not self.config.check_retrieval_relations or not all_queries:
            return
        sample_query = all_queries[0]
        relations = uow.retrieval_relations.get_by_query_id(sample_query.id)
        logger.info(f"=== Sample Retrieval Relations for Query [{sample_query.id}] ===")
        for rel in relations[:5]:
            chunk_id = rel.chunk_id or rel.image_chunk_id
            chunk_type = "chunk" if rel.chunk_id else "image_chunk"
            logger.info(f"  -> {chunk_type} [{chunk_id}] (group={rel.group_index}, order={rel.group_order})")
