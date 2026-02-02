"""Common test utilities for pipeline integration tests.

This module provides reusable utilities for testing retrieval and generation pipelines
against real PostgreSQL databases. It follows the same patterns as ingestor_test_utils.py.

Usage:
    from tests.autorag_research.pipelines.pipeline_test_utils import (
        PipelineTestConfig,
        PipelineTestVerifier,
        create_mock_llm,
    )

    # Retrieval Pipeline (uses real BM25RetrievalPipeline with bm25_index_path fixture)
    config = PipelineTestConfig(
        pipeline_type="retrieval",
        expected_total_queries=5,
        expected_min_results=15,
    )
    verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
    verifier.verify_all()

    # Generation Pipeline (uses real retrieval + mock LLM)
    config = PipelineTestConfig(
        pipeline_type="generation",
        expected_total_queries=5,
        check_token_usage=True,
        check_execution_time=True,
    )
    verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
    verifier.verify_all()
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.orm import scoped_session

from tests.util import CheckResult, VerificationReport

logger = logging.getLogger("AutoRAG-Research")


@dataclass
class PipelineTestConfig:
    """Configuration for pipeline integration tests."""

    # Pipeline type
    pipeline_type: Literal["retrieval", "generation"]

    # Common
    expected_total_queries: int = 5  # Seed data default

    # Retrieval pipeline specific
    expected_min_results: int | None = None

    # Generation pipeline specific
    check_token_usage: bool = False
    check_execution_time: bool = False
    expected_token_usage_keys: list[str] = field(
        default_factory=lambda: ["prompt_tokens", "completion_tokens", "total_tokens"]
    )

    # Persistence check
    check_persistence: bool = True

    # Schema for dynamic schema support
    schema: Any | None = None


class PipelineTestVerifier:
    """Verifier for pipeline integration tests.

    Provides verification methods for:
    - Return structure validation (required keys)
    - pipeline_id matching
    - total_queries count validation
    - total_results validation (retrieval)
    - token_usage structure validation (generation)
    - avg_execution_time_ms validation (generation)
    - Persistence verification (DB records exist)
    """

    def __init__(
        self,
        run_result: dict[str, Any],
        pipeline_id: int,
        session_factory: scoped_session,
        config: PipelineTestConfig,
    ):
        """Initialize verifier.

        Args:
            run_result: The result dictionary from pipeline.run().
            pipeline_id: The expected pipeline ID.
            session_factory: SQLAlchemy session factory for DB queries.
            config: Test configuration with expected values.
        """
        self.run_result = run_result
        self.pipeline_id = pipeline_id
        self.session_factory = session_factory
        self.config = config

    def verify_all(self) -> VerificationReport:
        """Run all configured checks and return detailed report.

        Returns:
            VerificationReport with all check results.

        Raises:
            AssertionError: If any check fails.
        """
        report = VerificationReport()

        # 1. Common checks
        report.add_check("return_structure", self._verify_return_structure())
        report.add_check("pipeline_id", self._verify_pipeline_id())
        report.add_check("total_queries", self._verify_total_queries())

        # 2. Type-specific checks
        if self.config.pipeline_type == "retrieval":
            if self.config.expected_min_results is not None:
                report.add_check("total_results", self._verify_total_results())
        elif self.config.pipeline_type == "generation":
            if self.config.check_token_usage:
                report.add_check("token_usage", self._verify_token_usage())
            if self.config.check_execution_time:
                report.add_check("execution_time", self._verify_execution_time())

        # 3. Persistence checks
        if self.config.check_persistence:
            report.add_check("persistence", self._verify_persistence())

        # Log summary
        logger.info(report.summary("Pipeline Verification Report"))

        # Assert all passed
        if not report.all_passed:
            failed_checks = [name for name, check in report.checks.items() if not check.passed]
            msg = f"Verification failed for checks: {failed_checks}\n{report.summary('Pipeline Verification Report')}"
            raise AssertionError(msg)

        return report

    # ==================== Common Checks ====================

    def _verify_return_structure(self) -> CheckResult:
        """Verify run() returns dict with required keys."""
        failures: list[str] = []

        if not isinstance(self.run_result, dict):
            return CheckResult(
                passed=False,
                message="Result is not a dict",
                failures=[f"Expected dict, got {type(self.run_result).__name__}"],
            )

        # Common required keys
        required_keys = ["pipeline_id", "total_queries"]

        # Type-specific required keys
        if self.config.pipeline_type == "retrieval":
            required_keys.append("total_results")
        elif self.config.pipeline_type == "generation":
            required_keys.extend(["token_usage", "avg_execution_time_ms"])

        for key in required_keys:
            if key not in self.run_result:
                failures.append(f"Missing required key: {key}")

        passed = len(failures) == 0
        return CheckResult(
            passed=passed,
            message=f"Has all {len(required_keys)} required keys" if passed else "Missing required keys",
            failures=failures,
        )

    def _verify_pipeline_id(self) -> CheckResult:
        """Verify returned pipeline_id matches expected."""
        actual = self.run_result.get("pipeline_id")
        expected = self.pipeline_id
        passed = actual == expected
        return CheckResult(
            passed=passed,
            message=f"Expected {expected}, got {actual}",
            failures=[] if passed else [f"pipeline_id mismatch: expected {expected}, got {actual}"],
        )

    def _verify_total_queries(self) -> CheckResult:
        """Verify total_queries matches expected."""
        actual = self.run_result.get("total_queries")
        expected = self.config.expected_total_queries
        passed = actual == expected
        return CheckResult(
            passed=passed,
            message=f"Expected {expected}, got {actual}",
            failures=[] if passed else [f"total_queries mismatch: expected {expected}, got {actual}"],
        )

    # ==================== Retrieval-Specific Checks ====================

    def _verify_total_results(self) -> CheckResult:
        """Verify total_results is >= expected minimum."""
        actual = self.run_result.get("total_results", 0)
        expected_min = self.config.expected_min_results or 0
        passed = actual >= expected_min
        return CheckResult(
            passed=passed,
            message=f"Expected >= {expected_min}, got {actual}",
            failures=[] if passed else [f"total_results too low: expected >= {expected_min}, got {actual}"],
        )

    # ==================== Generation-Specific Checks ====================

    def _verify_token_usage(self) -> CheckResult:
        """Verify token_usage dict has expected keys."""
        failures: list[str] = []

        token_usage = self.run_result.get("token_usage")
        if token_usage is None:
            return CheckResult(
                passed=False,
                message="token_usage is None",
                failures=["token_usage is None"],
            )

        if not isinstance(token_usage, dict):
            return CheckResult(
                passed=False,
                message="token_usage is not a dict",
                failures=[f"Expected dict, got {type(token_usage).__name__}"],
            )

        for key in self.config.expected_token_usage_keys:
            if key not in token_usage:
                failures.append(f"Missing token_usage key: {key}")

        # Verify values are numeric and non-negative
        for key, value in token_usage.items():
            if key in self.config.expected_token_usage_keys:
                if not isinstance(value, (int, float)):
                    failures.append(f"token_usage[{key}] is not numeric: {type(value).__name__}")
                elif value < 0:
                    failures.append(f"token_usage[{key}] is negative: {value}")

        passed = len(failures) == 0
        return CheckResult(
            passed=passed,
            message=f"Has all {len(self.config.expected_token_usage_keys)} expected keys"
            if passed
            else "Invalid token_usage",
            failures=failures,
        )

    def _verify_execution_time(self) -> CheckResult:
        """Verify avg_execution_time_ms is a valid positive number."""
        failures: list[str] = []

        exec_time = self.run_result.get("avg_execution_time_ms")
        if exec_time is None:
            failures.append("avg_execution_time_ms is None")
        elif not isinstance(exec_time, (int, float)):
            failures.append(f"avg_execution_time_ms is not numeric: {type(exec_time).__name__}")
        elif exec_time < 0:
            failures.append(f"avg_execution_time_ms is negative: {exec_time}")

        passed = len(failures) == 0
        return CheckResult(
            passed=passed,
            message=f"Valid execution time: {exec_time}" if passed else "Invalid execution time",
            failures=failures,
        )

    # ==================== Persistence Checks ====================

    def _verify_persistence(self) -> CheckResult:
        """Verify results are persisted in the database."""
        failures: list[str] = []

        session = self.session_factory()
        try:
            if self.config.pipeline_type == "retrieval":
                failures.extend(self._check_chunk_retrieved_results(session))
            elif self.config.pipeline_type == "generation":
                failures.extend(self._check_executor_results(session))
        finally:
            session.close()

        passed = len(failures) == 0
        return CheckResult(
            passed=passed,
            message="Results persisted correctly" if passed else "Persistence verification failed",
            failures=failures,
        )

    def _check_chunk_retrieved_results(self, session) -> list[str]:
        """Check ChunkRetrievedResult records exist for the pipeline."""
        failures: list[str] = []

        if self.config.schema is not None:
            from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository

            repo = ChunkRetrievedResultRepository(session, self.config.schema.ChunkRetrievedResult)
        else:
            from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository

            repo = ChunkRetrievedResultRepository(session)

        results = repo.get_by_pipeline(self.pipeline_id)
        expected_total = self.run_result.get("total_results", 0)

        # If expected_total is 0, having 0 results is correct
        if expected_total == 0 and len(results) == 0:
            return failures  # No failure - 0 expected, 0 found

        if len(results) == 0:
            failures.append("No ChunkRetrievedResult records found for pipeline")
        elif len(results) != expected_total:
            failures.append(f"ChunkRetrievedResult count mismatch: expected {expected_total}, got {len(results)}")

        return failures

    def _check_executor_results(self, session) -> list[str]:
        """Check ExecutorResult records exist for the pipeline."""
        failures: list[str] = []

        if self.config.schema is not None:
            from autorag_research.orm.repository.executor_result import ExecutorResultRepository

            repo = ExecutorResultRepository(session, self.config.schema.ExecutorResult)
        else:
            from autorag_research.orm.repository.executor_result import ExecutorResultRepository

            repo = ExecutorResultRepository(session)

        results = repo.get_by_pipeline_id(self.pipeline_id)
        expected_count = self.config.expected_total_queries

        if len(results) == 0:
            failures.append("No ExecutorResult records found for pipeline")
        elif len(results) != expected_count:
            failures.append(f"ExecutorResult count mismatch: expected {expected_count}, got {len(results)}")

        return failures


# ==================== Mock Factory Functions ====================


def create_mock_llm(
    response_text: str = "This is a generated answer.",
    token_usage: dict[str, int] | None = None,
) -> MagicMock:
    """Create a mock LLM that returns predictable responses (LangChain style).

    Note: We mock LLM because API calls are expensive and require API keys.
    For retrieval pipelines, use real BM25RetrievalPipeline with the
    bm25_index_path fixture from conftest.py.

    Args:
        response_text: The text to return from invoke()/ainvoke().
        token_usage: Token usage dict. Defaults to standard values.

    Returns:
        MagicMock configured as a LangChain BaseLanguageModel.
    """

    if token_usage is None:
        token_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    mock = MagicMock()
    mock_response = MagicMock()
    # LangChain uses 'content' attribute for response text
    mock_response.content = response_text
    mock_response.__str__ = lambda x: response_text
    # LangChain uses 'usage_metadata' for token counts
    mock_response.usage_metadata = {
        "input_tokens": token_usage["prompt_tokens"],
        "output_tokens": token_usage["completion_tokens"],
        "total_tokens": token_usage["total_tokens"],
    }
    mock.invoke.return_value = mock_response
    # Add async ainvoke for async pipeline support
    mock.ainvoke = AsyncMock(return_value=mock_response)
    return mock


def create_mock_retrieval_pipeline(
    pipeline_id: int = 1,
    default_results: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Create a mock retrieval pipeline for testing generation pipelines.

    Args:
        pipeline_id: The pipeline ID to return.
        default_results: Default retrieval results. If None, returns seed data chunk IDs (1-6).

    Returns:
        MagicMock configured as a BaseRetrievalPipeline.
        The mock.retrieve is an AsyncMock with side_effect, so call_count is available.
    """
    mock = MagicMock()
    mock.pipeline_id = pipeline_id

    if default_results is not None:
        mock.retrieve = AsyncMock(return_value=default_results)
    else:
        # Default: return chunk IDs that exist in seed data (1-6)
        # Use side_effect to preserve call_count tracking on the AsyncMock
        async def mock_retrieve(query_text: str, top_k: int):
            return [{"doc_id": i, "score": 0.9 - i * 0.1} for i in range(1, min(top_k + 1, 7))]

        mock.retrieve = AsyncMock(side_effect=mock_retrieve)

    return mock


def cleanup_pipeline_results_factory(session_factory):
    """Factory for creating cleanup fixtures for pipeline results.

    Usage in tests:
        @pytest.fixture
        def cleanup(self, session_factory):
            yield from cleanup_pipeline_results_factory(session_factory)

    Args:
        session_factory: SQLAlchemy session factory.

    Yields:
        List to append pipeline IDs for cleanup.
    """
    from autorag_research.orm.repository.executor_result import ExecutorResultRepository

    created_pipeline_ids: list[int] = []
    yield created_pipeline_ids

    session = session_factory()
    try:
        repo = ExecutorResultRepository(session)
        for pipeline_id in created_pipeline_ids:
            repo.delete_by_pipeline(pipeline_id)
        session.commit()
    finally:
        session.close()
