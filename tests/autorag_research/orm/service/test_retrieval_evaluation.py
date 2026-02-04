from dataclasses import dataclass

import pytest

from autorag_research.orm.service.retrieval_evaluation import (
    RetrievalEvaluationService,
    build_retrieval_gt_from_relations,
)
from autorag_research.schema import MetricInput


@dataclass
class MockRetrievalRelation:
    """Mock RetrievalRelation for testing."""

    group_index: int
    group_order: int
    chunk_id: int | None = None
    image_chunk_id: int | None = None
    score: int | None = None


class TestBuildRetrievalGtFromRelations:
    def test_empty_relations(self):
        """Empty input returns empty list and empty scores."""
        result, scores = build_retrieval_gt_from_relations([])
        assert result == []
        assert scores == {}

    def test_single_group_single_chunk(self):
        """Single group with single chunk item."""
        relations = [MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1)]
        result, scores = build_retrieval_gt_from_relations(relations)
        assert result == [["chunk_1"]]
        assert scores == {"chunk_1": 1}

    def test_single_group_single_image_chunk(self):
        """Single group with single image chunk item."""
        relations = [MockRetrievalRelation(group_index=0, group_order=0, image_chunk_id=1)]
        result, scores = build_retrieval_gt_from_relations(relations)
        assert result == [["image_chunk_1"]]
        assert scores == {"image_chunk_1": 1}

    def test_single_group_multiple_items_or_condition(self):
        """Multiple items in same group = OR condition."""
        relations = [
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1),
            MockRetrievalRelation(group_index=0, group_order=1, chunk_id=2),
            MockRetrievalRelation(group_index=0, group_order=2, chunk_id=3),
        ]
        result, scores = build_retrieval_gt_from_relations(relations)
        # All in same inner list = OR condition
        assert result == [["chunk_1", "chunk_2", "chunk_3"]]
        assert scores == {"chunk_1": 1, "chunk_2": 1, "chunk_3": 1}

    def test_multiple_groups_single_items_and_condition(self):
        """Multiple groups with single items = AND condition."""
        relations = [
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1),
            MockRetrievalRelation(group_index=1, group_order=0, chunk_id=2),
            MockRetrievalRelation(group_index=2, group_order=0, chunk_id=3),
        ]
        result, scores = build_retrieval_gt_from_relations(relations)
        # Each in different inner list = AND condition
        assert result == [["chunk_1"], ["chunk_2"], ["chunk_3"]]
        assert scores == {"chunk_1": 1, "chunk_2": 1, "chunk_3": 1}

    def test_mixed_and_or_conditions(self):
        """Mixed AND/OR: (chunk_1 OR chunk_2) AND chunk_3."""
        relations = [
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1),
            MockRetrievalRelation(group_index=0, group_order=1, chunk_id=2),
            MockRetrievalRelation(group_index=1, group_order=0, chunk_id=3),
        ]
        result, scores = build_retrieval_gt_from_relations(relations)
        assert result == [["chunk_1", "chunk_2"], ["chunk_3"]]
        assert scores == {"chunk_1": 1, "chunk_2": 1, "chunk_3": 1}

    def test_mixed_chunk_and_image_chunk(self):
        """Mixed chunk and image_chunk in same group = OR condition."""
        relations = [
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1),
            MockRetrievalRelation(group_index=0, group_order=1, image_chunk_id=2),
            MockRetrievalRelation(group_index=1, group_order=0, chunk_id=3),
        ]
        result, scores = build_retrieval_gt_from_relations(relations)
        assert result == [["chunk_1", "image_chunk_2"], ["chunk_3"]]
        assert scores == {"chunk_1": 1, "image_chunk_2": 1, "chunk_3": 1}

    def test_group_order_sorting(self):
        """Items within group are sorted by group_order."""
        relations = [
            MockRetrievalRelation(group_index=0, group_order=2, chunk_id=3),
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1),
            MockRetrievalRelation(group_index=0, group_order=1, chunk_id=2),
        ]
        result, scores = build_retrieval_gt_from_relations(relations)
        # Should be sorted by group_order: 0, 1, 2 -> chunk_ids: 1, 2, 3
        assert result == [["chunk_1", "chunk_2", "chunk_3"]]
        assert scores == {"chunk_1": 1, "chunk_2": 1, "chunk_3": 1}

    def test_group_index_sorting(self):
        """Groups are sorted by group_index."""
        relations = [
            MockRetrievalRelation(group_index=2, group_order=0, chunk_id=30),
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=10),
            MockRetrievalRelation(group_index=1, group_order=0, chunk_id=20),
        ]
        result, scores = build_retrieval_gt_from_relations(relations)
        assert result == [["chunk_10"], ["chunk_20"], ["chunk_30"]]
        assert scores == {"chunk_10": 1, "chunk_20": 1, "chunk_30": 1}

    def test_none_chunk_ids_ignored(self):
        """Relations with both None chunk_id and image_chunk_id are ignored."""
        relations = [
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1),
            MockRetrievalRelation(group_index=0, group_order=1, chunk_id=None, image_chunk_id=None),
            MockRetrievalRelation(group_index=0, group_order=2, chunk_id=2),
        ]
        result, scores = build_retrieval_gt_from_relations(relations)
        assert result == [["chunk_1", "chunk_2"]]
        assert scores == {"chunk_1": 1, "chunk_2": 1}

    def test_all_none_ids(self):
        """All None IDs returns empty list and empty scores."""
        relations = [
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=None, image_chunk_id=None),
            MockRetrievalRelation(group_index=1, group_order=0, chunk_id=None, image_chunk_id=None),
        ]
        result, scores = build_retrieval_gt_from_relations(relations)
        assert result == []
        assert scores == {}

    def test_complex_scenario_with_mixed_types(self):
        """Complex: (chunk_1 OR image_chunk_2 OR chunk_3) AND (image_chunk_4 OR chunk_5) AND chunk_6."""
        relations = [
            # Group 0: OR condition with mixed types
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1),
            MockRetrievalRelation(group_index=0, group_order=1, image_chunk_id=2),
            MockRetrievalRelation(group_index=0, group_order=2, chunk_id=3),
            # Group 1: OR condition with mixed types
            MockRetrievalRelation(group_index=1, group_order=0, image_chunk_id=4),
            MockRetrievalRelation(group_index=1, group_order=1, chunk_id=5),
            # Group 2: single item
            MockRetrievalRelation(group_index=2, group_order=0, chunk_id=6),
        ]
        result, scores = build_retrieval_gt_from_relations(relations)
        assert result == [["chunk_1", "image_chunk_2", "chunk_3"], ["image_chunk_4", "chunk_5"], ["chunk_6"]]
        assert scores == {
            "chunk_1": 1,
            "image_chunk_2": 1,
            "chunk_3": 1,
            "image_chunk_4": 1,
            "chunk_5": 1,
            "chunk_6": 1,
        }

    def test_only_image_chunks(self):
        """All image chunks."""
        relations = [
            MockRetrievalRelation(group_index=0, group_order=0, image_chunk_id=1),
            MockRetrievalRelation(group_index=0, group_order=1, image_chunk_id=2),
            MockRetrievalRelation(group_index=1, group_order=0, image_chunk_id=3),
        ]
        result, scores = build_retrieval_gt_from_relations(relations)
        assert result == [["image_chunk_1", "image_chunk_2"], ["image_chunk_3"]]
        assert scores == {"image_chunk_1": 1, "image_chunk_2": 1, "image_chunk_3": 1}

    def test_with_graded_relevance_scores(self):
        """Test that graded relevance scores are correctly extracted."""
        relations = [
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1, score=2),
            MockRetrievalRelation(group_index=0, group_order=1, chunk_id=2, score=1),
            MockRetrievalRelation(group_index=1, group_order=0, chunk_id=3, score=0),
        ]
        result, scores = build_retrieval_gt_from_relations(relations)
        assert result == [["chunk_1", "chunk_2"], ["chunk_3"]]
        assert scores == {"chunk_1": 2, "chunk_2": 1, "chunk_3": 0}

    def test_with_none_score_defaults_to_one(self):
        """Test that None score defaults to 1."""
        relations = [
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1, score=None),
            MockRetrievalRelation(group_index=0, group_order=1, chunk_id=2, score=2),
        ]
        result, scores = build_retrieval_gt_from_relations(relations)
        assert result == [["chunk_1", "chunk_2"]]
        assert scores == {"chunk_1": 1, "chunk_2": 2}

    def test_mixed_scores_with_image_chunks(self):
        """Test graded scores with mixed chunk types."""
        relations = [
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1, score=2),
            MockRetrievalRelation(group_index=0, group_order=1, image_chunk_id=2, score=1),
            MockRetrievalRelation(group_index=1, group_order=0, image_chunk_id=3, score=2),
        ]
        result, scores = build_retrieval_gt_from_relations(relations)
        assert result == [["chunk_1", "image_chunk_2"], ["image_chunk_3"]]
        assert scores == {"chunk_1": 2, "image_chunk_2": 1, "image_chunk_3": 2}


class TestRetrievalEvaluationService:
    @pytest.fixture
    def service(self, session_factory):
        return RetrievalEvaluationService(session_factory)

    def test_set_metric_and_properties(self, service):
        def dummy_metric(x):
            return 1.0

        service.set_metric(metric_id=1, metric_func=dummy_metric)

        assert service.metric_id == 1
        assert service.metric_func == dummy_metric

    def test_get_metric_existing(self, service):
        # Seed data: metric id=1 name='retrieval@k' type='retrieval'
        metric = service.get_metric("retrieval@k", "retrieval")

        assert metric is not None
        assert metric.id == 1
        assert metric.name == "retrieval@k"

    def test_get_metric_not_found(self, service):
        metric = service.get_metric("nonexistent_metric")
        assert metric is None

    def test_get_or_create_metric_existing(self, service):
        # Seed data: metric id=1 name='retrieval@k' type='retrieval'
        metric_id = service.get_or_create_metric("retrieval@k", "retrieval")
        assert metric_id == 1

    def test_get_or_create_metric_new(self, service):
        metric_id = service.get_or_create_metric("test_new_metric", "retrieval")

        assert metric_id is not None
        assert isinstance(metric_id, int)

        # Cleanup
        with service._create_uow() as uow:
            uow.metrics.delete_by_id(metric_id)
            uow.commit()

    def test_count_total_query_ids(self, service):
        # Seed data has 5 queries
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
        # Seed: query 1, pipeline 1 has ChunkRetrievedResult with chunk_id=1
        # Seed: query 1 has RetrievalRelation with chunk_id=1, score=2
        results = service._get_execution_results(pipeline_id=1, query_ids=[1])

        assert 1 in results
        assert "retrieved_ids" in results[1]
        assert "retrieval_gt" in results[1]
        assert "relevance_scores" in results[1]
        # IDs are now prefixed with 'chunk_' or 'image_chunk_'
        assert results[1]["retrieved_ids"] == ["chunk_1"]
        assert results[1]["retrieval_gt"] == [["chunk_1"]]
        assert results[1]["relevance_scores"] == {"chunk_1": 2}

    def test_get_execution_results_no_chunk_results(self, service):
        # Query 3 has no ChunkRetrievedResult for pipeline 1
        results = service._get_execution_results(pipeline_id=1, query_ids=[2])

        assert 2 in results
        assert results[2]["retrieved_ids"] == []

    def test_filter_missing_query_ids(self, service):
        # Seed: (query_id=1, pipeline_id=1, metric_id=1) exists
        # Query 2 with pipeline 1, metric 1 does not exist
        missing = service._filter_missing_query_ids(pipeline_id=1, metric_id=1, query_ids=[1, 2, 3])

        assert 1 not in missing
        assert 2 in missing
        assert 3 in missing

    def test_prepare_metric_input(self, service):
        execution_result = {
            "retrieved_ids": ["chunk_1", "image_chunk_2"],
            "retrieval_gt": [["chunk_1"], ["image_chunk_3"]],
            "relevance_scores": {"chunk_1": 2, "image_chunk_3": 1},
        }

        metric_input = service._prepare_metric_input(pipeline_id=1, query_id=1, execution_result=execution_result)

        assert isinstance(metric_input, MetricInput)
        assert metric_input.retrieved_ids == ["chunk_1", "image_chunk_2"]
        assert metric_input.retrieval_gt == [["chunk_1"], ["image_chunk_3"]]
        assert metric_input.relevance_scores == {"chunk_1": 2, "image_chunk_3": 1}

    def test_save_evaluation_results(self, service):
        results = [(3, 0.75), (4, 0.80)]
        service._save_evaluation_results(pipeline_id=1, metric_id=1, results=results)

        # Verify saved
        with service._create_uow() as uow:
            saved = uow.evaluation_results.get_by_query_and_pipeline(query_id=3, pipeline_id=1)
            assert any(r.metric_id == 1 and r.metric_result == 0.75 for r in saved)

            # Cleanup
            uow.evaluation_results.delete_by_composite_key(query_id=3, pipeline_id=1, metric_id=1)
            uow.evaluation_results.delete_by_composite_key(query_id=4, pipeline_id=1, metric_id=1)
            uow.commit()

    def test_is_evaluation_complete_false(self, service):
        # Seed: only query 1 has evaluation for pipeline 1, metric 1
        # But query 1 also has ChunkRetrievedResult for pipeline 1
        # Query 2-5 don't have evaluation results
        is_complete = service.is_evaluation_complete(pipeline_id=1, metric_id=1)
        assert is_complete is False

    def test_is_evaluation_complete_true(self, service):
        # Pipeline 1, metric 1: only query 1 has ChunkRetrievedResult and EvaluationResult
        # We need to check that for all queries with execution results, there are evaluation results
        # Since query 1 has both, and only query 1 has ChunkRetrievedResult for pipeline 1
        # But _filter_missing_query_ids checks all query_ids, not just those with execution results
        # So this depends on the implementation

        # Let's create a scenario where it should return True
        # Add evaluation results for all queries for a new metric
        new_metric_id = service.get_or_create_metric("test_complete_metric", "retrieval")

        # Add evaluation results for all 5 queries
        results = [(i, 0.5) for i in range(1, 6)]
        service._save_evaluation_results(pipeline_id=1, metric_id=new_metric_id, results=results)

        is_complete = service.is_evaluation_complete(pipeline_id=1, metric_id=new_metric_id)
        assert is_complete is True

        # Cleanup
        with service._create_uow() as uow:
            for i in range(1, 6):
                uow.evaluation_results.delete_by_composite_key(query_id=i, pipeline_id=1, metric_id=new_metric_id)
            uow.metrics.delete_by_id(new_metric_id)
            uow.commit()

    def test_has_results_for_queries_all_have_results(self, service):
        # Seed: query 1, pipeline 1 has ChunkRetrievedResult
        # Seed: query 3, pipeline 1 has ImageChunkRetrievedResult
        assert service._has_results_for_queries(pipeline_id=1, query_ids=[1]) is True
        assert service._has_results_for_queries(pipeline_id=1, query_ids=[3]) is True
        assert service._has_results_for_queries(pipeline_id=1, query_ids=[1, 3]) is True

    def test_has_results_for_queries_missing_results(self, service):
        # Seed: query 2, 4, 5 have no results for pipeline 1
        assert service._has_results_for_queries(pipeline_id=1, query_ids=[2]) is False
        assert service._has_results_for_queries(pipeline_id=1, query_ids=[1, 2]) is False
        assert service._has_results_for_queries(pipeline_id=1, query_ids=[4, 5]) is False

    def test_verify_pipeline_completion_incomplete(self, service):
        # Seed: pipeline 1 has results only for query 1 (chunk) and query 3 (image_chunk)
        # Missing: queries 2, 4, 5
        assert service.verify_pipeline_completion(pipeline_id=1) is False

    def test_verify_pipeline_completion_complete(self, service):
        # Add retrieval results for all queries to make pipeline complete
        with service._create_uow() as uow:
            schema_classes = service._get_schema_classes()
            chunk_result_cls = schema_classes["ChunkRetrievedResult"]

            # Add results for queries 2, 4, 5 (query 1 and 3 already have results)
            new_results = [
                chunk_result_cls(query_id=2, pipeline_id=1, chunk_id=3, rel_score=0.7),
                chunk_result_cls(query_id=4, pipeline_id=1, chunk_id=4, rel_score=0.6),
                chunk_result_cls(query_id=5, pipeline_id=1, chunk_id=5, rel_score=0.5),
            ]
            for result in new_results:
                uow.chunk_results.add(result)
            uow.commit()

        assert service.verify_pipeline_completion(pipeline_id=1) is True

        # Cleanup
        with service._create_uow() as uow:
            uow.chunk_results.delete_by_composite_key(query_id=2, pipeline_id=1, chunk_id=3)
            uow.chunk_results.delete_by_composite_key(query_id=4, pipeline_id=1, chunk_id=4)
            uow.chunk_results.delete_by_composite_key(query_id=5, pipeline_id=1, chunk_id=5)
            uow.commit()
