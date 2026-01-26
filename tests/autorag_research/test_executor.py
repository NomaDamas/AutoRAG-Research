"""Tests for the Executor class."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from autorag_research.config import (
    BaseGenerationPipelineConfig,
    BaseMetricConfig,
    BaseRetrievalPipelineConfig,
    ExecutorConfig,
    MetricType,
)
from autorag_research.evaluation.metrics.retrieval import NDCGConfig, RecallConfig
from autorag_research.executor import Executor, ExecutorResult, MetricResult
from autorag_research.pipelines.retrieval.bm25 import BM25PipelineConfig


class TestExecutorWithRealDB:
    """Test suite for Executor class with real database."""

    @pytest.fixture
    def cleanup_pipelines(self, session_factory):
        """Clean up created pipelines after tests."""
        created_pipeline_ids = []

        yield created_pipeline_ids

        # Clean up after test
        from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
        from autorag_research.orm.repository.evaluator_result import EvaluatorResultRepository
        from autorag_research.orm.repository.pipeline import PipelineRepository

        session = session_factory()
        try:
            result_repo = ChunkRetrievedResultRepository(session)
            eval_repo = EvaluatorResultRepository(session)
            pipeline_repo = PipelineRepository(session)

            for pipeline_id in created_pipeline_ids:
                # Delete chunk results
                result_repo.delete_by_pipeline(pipeline_id)

                # Delete evaluation results (iterate and delete each)
                eval_results = eval_repo.get_by_pipeline_id(pipeline_id)
                for eval_result in eval_results:
                    eval_repo.delete(eval_result)
            session.commit()

            # Delete pipelines
            for pipeline_id in created_pipeline_ids:
                pipeline_repo.delete_by_id(pipeline_id)
            session.commit()
        finally:
            session.close()

    @pytest.fixture
    def mock_bm25_module(self):
        """Create a mock BM25DBModule for testing."""

        def mock_run(queries: list[str], top_k: int) -> list[list[dict]]:
            results = []
            for _ in queries:
                results.append([{"doc_id": i + 1, "score": 0.9 - i * 0.1} for i in range(top_k)])
            return results

        mock = MagicMock()
        mock.run = mock_run
        return mock

    def test_run_successful_pipeline(self, session_factory, cleanup_pipelines, mock_bm25_module):
        """Test successful pipeline execution with real database."""
        config = ExecutorConfig(
            pipelines=[
                BM25PipelineConfig(
                    name="test_successful_pipeline",
                    tokenizer="bert",
                    top_k=3,
                ),
            ],
            metrics=[RecallConfig(), NDCGConfig()],
            max_retries=2,
        )

        executor = Executor(session_factory, config)

        with patch(
            "autorag_research.nodes.retrieval.bm25.BM25DBModule",
            return_value=mock_bm25_module,
        ):
            result = executor.run()

        # Track created pipelines for cleanup
        for pr in result.pipeline_results:
            if pr.pipeline_id > 0:
                cleanup_pipelines.append(pr.pipeline_id)

        assert isinstance(result, ExecutorResult)
        assert result.total_pipelines_run == 1
        assert result.total_pipelines_succeeded == 1
        assert len(result.pipeline_results) == 1
        assert result.pipeline_results[0].success is True
        assert result.pipeline_results[0].pipeline_name == "test_successful_pipeline"
        assert result.total_metrics_evaluated == 2
        assert all(mr.success for mr in result.metric_results)
        assert all(mr.average is not None for mr in result.metric_results)

    def test_multiple_pipelines_execution(self, session_factory, cleanup_pipelines, mock_bm25_module):
        """Test execution of multiple pipelines."""
        config = ExecutorConfig(
            pipelines=[
                BM25PipelineConfig(
                    name="test_multi_pipeline_1",
                    tokenizer="bert",
                    top_k=3,
                ),
                BM25PipelineConfig(
                    name="test_multi_pipeline_2",
                    tokenizer="bert",
                    top_k=3,
                ),
            ],
            metrics=[RecallConfig()],
            max_retries=2,
        )

        executor = Executor(session_factory, config)

        with patch(
            "autorag_research.nodes.retrieval.bm25.BM25DBModule",
            return_value=mock_bm25_module,
        ):
            result = executor.run()

        # Track created pipelines for cleanup
        for pr in result.pipeline_results:
            if pr.pipeline_id > 0:
                cleanup_pipelines.append(pr.pipeline_id)

        assert result.total_pipelines_run == 2
        assert result.total_pipelines_succeeded == 2
        # Each pipeline should have metrics evaluated
        assert result.total_metrics_evaluated == 2

    def test_pipeline_failure_with_retry(self, session_factory, cleanup_pipelines):
        """Test pipeline failure triggers retry and eventually fails."""
        config = ExecutorConfig(
            pipelines=[
                BM25PipelineConfig(
                    name="test_failure_pipeline",
                    tokenizer="bert",
                ),
            ],
            metrics=[RecallConfig()],
            max_retries=2,
        )

        executor = Executor(session_factory, config)

        # Mock BM25DBModule to always fail
        mock_module = MagicMock()
        mock_module.run.side_effect = Exception("BM25 error")

        with patch(
            "autorag_research.nodes.retrieval.bm25.BM25DBModule",
            return_value=mock_module,
        ):
            result = executor.run()

        # Track created pipelines for cleanup (even failed ones create DB entries)
        for pr in result.pipeline_results:
            if pr.pipeline_id > 0:
                cleanup_pipelines.append(pr.pipeline_id)

        assert result.total_pipelines_run == 1
        assert result.total_pipelines_succeeded == 0
        assert result.pipeline_results[0].success is False
        assert result.pipeline_results[0].retries_used == config.max_retries


class TestMetricEvaluationRules:
    """Test suite for metric evaluation rules."""

    @pytest.fixture
    def mock_retrieval_metric_config(self):
        """Create mock retrieval metric config."""

        @dataclass
        class MockRetrievalMetric(BaseMetricConfig):
            metric_type: MetricType = field(default=MetricType.RETRIEVAL, init=False)

            def get_metric_name(self) -> str:
                return "mock_retrieval"

            def get_metric_func(self) -> Callable:
                return lambda x: 0.8

        return MockRetrievalMetric()

    @pytest.fixture
    def mock_generation_metric_config(self):
        """Create mock generation metric config."""

        @dataclass
        class MockGenerationMetric(BaseMetricConfig):
            metric_type: MetricType = field(default=MetricType.GENERATION, init=False)

            def get_metric_name(self) -> str:
                return "mock_generation"

            def get_metric_func(self) -> Callable:
                return lambda x: 0.7

        return MockGenerationMetric()

    def test_retrieval_pipeline_skips_generation_metrics(
        self,
        session_factory,
        mock_retrieval_metric_config,
        mock_generation_metric_config,
    ):
        """Test that retrieval pipelines only evaluate retrieval metrics."""

        # Create mock pipeline config
        @dataclass
        class MockRetrievalPipeline(BaseRetrievalPipelineConfig):
            def get_pipeline_class(self) -> type:
                return MagicMock

            def get_pipeline_kwargs(self) -> dict[str, Any]:
                return {}

            def get_run_kwargs(self) -> dict[str, Any]:
                return {}

        config = ExecutorConfig(
            pipelines=[MockRetrievalPipeline(name="test_retrieval")],
            metrics=[mock_retrieval_metric_config, mock_generation_metric_config],
            max_retries=1,
        )

        executor = Executor(session_factory, config)

        # Mock successful pipeline run
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "pipeline_id": 999,
            "total_queries": 10,
            "total_results": 100,
        }

        config.pipelines[0].get_pipeline_class = lambda: lambda **kw: mock_pipeline
        executor._verify_pipeline_completion = MagicMock(return_value=True)

        # Mock evaluation to track which metrics are called
        evaluated_metrics = []

        def track_evaluate(pipeline_id, metric_config):
            evaluated_metrics.append(metric_config.get_metric_name())
            return MetricResult(
                metric_name=metric_config.get_metric_name(),
                metric_type=metric_config.metric_type,
                pipeline_id=pipeline_id,
                queries_evaluated=10,
                average=0.8,
                success=True,
            )

        executor._evaluate_metric = track_evaluate

        result = executor.run()

        # Should only evaluate retrieval metric, not generation
        assert "mock_retrieval" in evaluated_metrics
        assert "mock_generation" not in evaluated_metrics
        assert result.total_metrics_evaluated == 1

    def test_generation_pipeline_evaluates_all_metrics(
        self,
        session_factory,
        mock_retrieval_metric_config,
        mock_generation_metric_config,
    ):
        """Test that generation pipelines evaluate both retrieval and generation metrics."""

        # Create mock generation pipeline config
        @dataclass
        class MockGenerationPipeline(BaseGenerationPipelineConfig):
            def get_pipeline_class(self) -> type:
                return MagicMock

            def get_pipeline_kwargs(self) -> dict[str, Any]:
                return {}

            def get_run_kwargs(self) -> dict[str, Any]:
                return {}

        config = ExecutorConfig(
            pipelines=[MockGenerationPipeline(name="test_generation")],
            metrics=[mock_retrieval_metric_config, mock_generation_metric_config],
            max_retries=1,
        )

        executor = Executor(session_factory, config)

        # Mock successful pipeline run
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "pipeline_id": 999,
            "total_queries": 10,
            "total_results": 100,
        }

        config.pipelines[0].get_pipeline_class = lambda: lambda **kw: mock_pipeline
        executor._verify_pipeline_completion = MagicMock(return_value=True)

        # Mock evaluation to track which metrics are called
        evaluated_metrics = []

        def track_evaluate(pipeline_id, metric_config):
            evaluated_metrics.append(metric_config.get_metric_name())
            return MetricResult(
                metric_name=metric_config.get_metric_name(),
                metric_type=metric_config.metric_type,
                pipeline_id=pipeline_id,
                queries_evaluated=10,
                average=0.75,
                success=True,
            )

        executor._evaluate_metric = track_evaluate

        result = executor.run()

        # Should evaluate BOTH metrics for generation pipeline
        assert "mock_retrieval" in evaluated_metrics
        assert "mock_generation" in evaluated_metrics
        assert result.total_metrics_evaluated == 2
