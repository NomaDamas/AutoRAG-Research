import pytest

from autorag_research.orm.service.generation_evaluation import GenerationEvaluationService
from autorag_research.schema import MetricInput


class TestGenerationEvaluationService:
    @pytest.fixture
    def service(self, session_factory):
        return GenerationEvaluationService(session_factory)

    def test_set_metric_and_properties(self, service):
        def dummy_metric(x):
            return 1.0

        service.set_metric(metric_id=2, metric_func=dummy_metric)

        assert service.metric_id == 2
        assert service.metric_func == dummy_metric
        assert service._async_metric_func is not None

    def test_get_metric_existing(self, service):
        metric = service.get_metric("bleu", "generation")

        assert metric is not None
        assert metric.id == 2
        assert metric.name == "bleu"

    def test_get_metric_not_found(self, service):
        metric = service.get_metric("nonexistent_metric")
        assert metric is None

    def test_get_or_create_metric_existing(self, service):
        metric_id = service.get_or_create_metric("bleu", "generation")
        assert metric_id == 2

    def test_get_or_create_metric_new(self, service):
        metric_id = service.get_or_create_metric("test_gen_metric", "generation")

        assert metric_id is not None
        assert isinstance(metric_id, int)

        with service._create_uow() as uow:
            uow.metrics.delete_by_id(metric_id)
            uow.commit()

    def test_count_total_query_ids(self, service):
        count = service._count_total_query_ids()
        assert count == 5

    def test_fetch_query_ids_batch(self, service):
        batch = service._fetch_query_ids_batch(limit=3, offset=0)

        assert len(batch) == 3
        assert batch == [1, 2, 3]

    def test_fetch_query_ids_batch_with_offset(self, service):
        batch = service._fetch_query_ids_batch(limit=2, offset=3)

        assert len(batch) == 2
        assert batch == [4, 5]

    def test_iter_query_id_batches(self, service):
        batches = list(service._iter_query_id_batches(batch_size=2))

        assert len(batches) == 3
        assert batches[0] == [1, 2]
        assert batches[1] == [3, 4]
        assert batches[2] == [5]

    def test_get_execution_results(self, service):
        results = service._get_execution_results(pipeline_id=1, query_ids=[2])

        assert 2 in results
        assert "generated_text" in results[2]
        assert "generation_gt" in results[2]
        assert results[2]["generated_text"] == "Generated text 1"
        assert results[2]["generation_gt"] == ["beta"]

    def test_get_execution_results_no_executor_result(self, service):
        results = service._get_execution_results(pipeline_id=1, query_ids=[3])
        assert 3 not in results

    def test_filter_missing_query_ids(self, service):
        missing = service._filter_missing_query_ids(pipeline_id=1, metric_id=2, query_ids=[1, 2, 3])

        assert 2 not in missing
        assert 1 in missing
        assert 3 in missing

    def test_prepare_metric_input(self, service):
        execution_result = {
            "generated_text": "This is generated",
            "generation_gt": ["ground", "truth"],
        }

        metric_input = service._prepare_metric_input(pipeline_id=1, query_id=1, execution_result=execution_result)

        assert isinstance(metric_input, MetricInput)
        assert metric_input.generated_texts == "This is generated"
        assert metric_input.generation_gt == ["ground", "truth"]

    def test_save_evaluation_results(self, service):
        results = [(3, 0.65), (4, 0.70)]
        service._save_evaluation_results(pipeline_id=1, metric_id=2, results=results)

        with service._create_uow() as uow:
            saved = uow.evaluation_results.get_by_query_and_pipeline(query_id=3, pipeline_id=1)
            assert any(r.metric_id == 2 and r.metric_result == 0.65 for r in saved)

            uow.evaluation_results.delete_by_composite_key(query_id=3, pipeline_id=1, metric_id=2)
            uow.evaluation_results.delete_by_composite_key(query_id=4, pipeline_id=1, metric_id=2)
            uow.commit()

    def test_is_evaluation_complete_false(self, service):
        is_complete = service.is_evaluation_complete(pipeline_id=1, metric_id=2)
        assert is_complete is False

    def test_is_evaluation_complete_true(self, service):
        new_metric_id = service.get_or_create_metric("test_complete_gen_metric", "generation")

        results = [(i, 0.5) for i in range(1, 6)]
        service._save_evaluation_results(pipeline_id=1, metric_id=new_metric_id, results=results)

        is_complete = service.is_evaluation_complete(pipeline_id=1, metric_id=new_metric_id)
        assert is_complete is True

        with service._create_uow() as uow:
            for i in range(1, 6):
                uow.evaluation_results.delete_by_composite_key(query_id=i, pipeline_id=1, metric_id=new_metric_id)
            uow.metrics.delete_by_id(new_metric_id)
            uow.commit()

    def test_has_results_for_queries_all_have_results(self, service):
        # Seed: query 1 and 2 have ExecutorResult for pipeline 1
        assert service._has_results_for_queries(pipeline_id=1, query_ids=[1]) is True
        assert service._has_results_for_queries(pipeline_id=1, query_ids=[2]) is True
        assert service._has_results_for_queries(pipeline_id=1, query_ids=[1, 2]) is True

    def test_has_results_for_queries_missing_results(self, service):
        # Seed: query 3, 4, 5 have no ExecutorResult for pipeline 1
        assert service._has_results_for_queries(pipeline_id=1, query_ids=[3]) is False
        assert service._has_results_for_queries(pipeline_id=1, query_ids=[1, 3]) is False
        assert service._has_results_for_queries(pipeline_id=1, query_ids=[3, 4, 5]) is False

    def test_verify_pipeline_completion_incomplete(self, service):
        # Seed: pipeline 1 has ExecutorResult only for query 1 and 2
        # Missing: queries 3, 4, 5
        assert service.verify_pipeline_completion(pipeline_id=1) is False

    def test_verify_pipeline_completion_complete(self, service):
        # Add executor results for all queries to make pipeline complete
        with service._create_uow() as uow:
            schema_classes = service._get_schema_classes()
            executor_result_cls = schema_classes["ExecutorResult"]

            # Add results for queries 3, 4, 5 (query 1 and 2 already have results)
            new_results = [
                executor_result_cls(
                    query_id=3, pipeline_id=1, generation_result="gen3", token_usage=100, execution_time=1000
                ),
                executor_result_cls(
                    query_id=4, pipeline_id=1, generation_result="gen4", token_usage=100, execution_time=1000
                ),
                executor_result_cls(
                    query_id=5, pipeline_id=1, generation_result="gen5", token_usage=100, execution_time=1000
                ),
            ]
            for result in new_results:
                uow.executor_results.add(result)
            uow.commit()

        assert service.verify_pipeline_completion(pipeline_id=1) is True

        # Cleanup
        with service._create_uow() as uow:
            uow.executor_results.delete_by_composite_key(query_id=3, pipeline_id=1)
            uow.executor_results.delete_by_composite_key(query_id=4, pipeline_id=1)
            uow.executor_results.delete_by_composite_key(query_id=5, pipeline_id=1)
            uow.commit()
