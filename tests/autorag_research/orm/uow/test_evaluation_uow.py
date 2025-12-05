import pytest

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.repository.evaluator_result import EvaluatorResultRepository
from autorag_research.orm.repository.executor_result import ExecutorResultRepository
from autorag_research.orm.repository.metric import MetricRepository
from autorag_research.orm.repository.pipeline import PipelineRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.repository.retrieval_relation import RetrievalRelationRepository
from autorag_research.orm.uow import RetrievalEvaluationUnitOfWork
from autorag_research.orm.uow.evaluation_uow import GenerationEvaluationUnitOfWork


@pytest.mark.parametrize(
    ("repo_property", "expected_class"),
    [
        ("queries", QueryRepository),
        ("pipelines", PipelineRepository),
        ("metrics", MetricRepository),
        ("retrieval_relations", RetrievalRelationRepository),
        ("chunk_results", ChunkRetrievedResultRepository),
        ("evaluation_results", EvaluatorResultRepository),
    ],
)
def test_retrieval_evaluation_uow_repository_returns_correct_type(session_factory, repo_property, expected_class):
    with RetrievalEvaluationUnitOfWork(session_factory) as uow:
        repo = getattr(uow, repo_property)
        assert isinstance(repo, expected_class)


@pytest.mark.parametrize(
    ("repo_property", "expected_class"),
    [
        ("queries", QueryRepository),
        ("pipelines", PipelineRepository),
        ("metrics", MetricRepository),
        ("executor_results", ExecutorResultRepository),
        ("evaluation_results", EvaluatorResultRepository),
    ],
)
def test_generation_evaluation_uow_repository_returns_correct_type(session_factory, repo_property, expected_class):
    with GenerationEvaluationUnitOfWork(session_factory) as uow:
        repo = getattr(uow, repo_property)
        assert isinstance(repo, expected_class)


def test_retrieval_evaluation_uow_can_use_existing_seed_data(session_factory):
    with RetrievalEvaluationUnitOfWork(session_factory) as uow:
        query = uow.queries.get_by_id(1)
        pipeline = uow.pipelines.get_by_id(1)
        metric = uow.metrics.get_by_id(1)

        assert query is not None
        assert pipeline is not None
        assert metric is not None


def test_generation_evaluation_uow_can_use_existing_seed_data(session_factory):
    with GenerationEvaluationUnitOfWork(session_factory) as uow:
        query = uow.queries.get_by_id(1)
        pipeline = uow.pipelines.get_by_id(1)
        metric = uow.metrics.get_by_id(1)

        assert query is not None
        assert pipeline is not None
        assert metric is not None
