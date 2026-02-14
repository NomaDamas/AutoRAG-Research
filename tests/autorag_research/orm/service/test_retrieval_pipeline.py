from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.schema import Chunk
from autorag_research.orm.service.retrieval_pipeline import RetrievalPipelineService


class TestRetrievalPipelineService:
    @pytest.fixture
    def service(self, session_factory):
        return RetrievalPipelineService(session_factory)

    @pytest.fixture
    def mock_retrieval_func(self):
        async def retrieval_func(query_id: int | str, top_k: int) -> list[dict]:
            """Mock async retrieval function that returns 2 results per query ID."""
            return [
                {"doc_id": 1, "score": 0.9},
                {"doc_id": 2, "score": 0.8},
            ][:top_k]

        return retrieval_func

    def test_run_pipeline(self, service, mock_retrieval_func, session_factory):
        # Count actual queries in database
        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        pipeline_id, _ = service.get_or_create_pipeline(
            name="test_pipeline",
            config={"type": "test"},
        )

        result = service.run_pipeline(
            retrieval_func=mock_retrieval_func,
            pipeline_id=pipeline_id,
            top_k=2,
        )

        assert "pipeline_id" in result
        assert "total_queries" in result
        assert "total_results" in result
        assert result["pipeline_id"] == pipeline_id
        assert result["total_queries"] == query_count
        assert result["total_results"] == query_count * 2

        service.delete_pipeline_results(pipeline_id)

    def test_delete_pipeline_results(self, service, mock_retrieval_func, session_factory):
        # Count actual queries in database
        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        pipeline_id, _ = service.get_or_create_pipeline(
            name="test_delete_pipeline",
            config={"type": "test"},
        )

        service.run_pipeline(
            retrieval_func=mock_retrieval_func,
            pipeline_id=pipeline_id,
            top_k=2,
        )

        # Use UoW to verify results before deletion
        with service._create_uow() as uow:
            results_before = uow.chunk_results.get_by_pipeline(pipeline_id)
            assert len(results_before) > 0

        deleted_count = service.delete_pipeline_results(pipeline_id)

        expected_results = query_count * 2
        assert deleted_count == expected_results

        # Use UoW to verify results after deletion
        with service._create_uow() as uow:
            results_after = uow.chunk_results.get_by_pipeline(pipeline_id)
            assert len(results_after) == 0


class TestVectorSearchByEmbedding:
    """Tests for RetrievalPipelineService.vector_search_by_embedding method."""

    @pytest.fixture
    def retrieval_service(self, session_factory: sessionmaker[Session]):
        """Create a RetrievalPipelineService instance for testing."""
        from autorag_research.orm.service.retrieval_pipeline import RetrievalPipelineService

        return RetrievalPipelineService(session_factory=session_factory)

    def test_vector_search_by_embedding(self, retrieval_service):
        """Test vector search using a provided embedding directly."""
        # Mock chunk results
        mock_chunk_results = [
            (Chunk(id=1, contents="Text content 1"), 0.1),  # distance
            (Chunk(id=2, contents="Text content 2"), 0.2),
        ]

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores") as mock_search:
            mock_search.return_value = mock_chunk_results

            test_embedding = [0.1] * 768
            results = retrieval_service.vector_search_by_embedding(embedding=test_embedding, top_k=3)

            assert len(results) == 2

            # Verify score conversion (1 - distance)
            assert results[0]["doc_id"] == 1
            assert results[0]["score"] == pytest.approx(0.9)  # 1 - 0.1
            assert results[0]["content"] == "Text content 1"

            assert results[1]["doc_id"] == 2
            assert results[1]["score"] == pytest.approx(0.8)  # 1 - 0.2

            # Verify vector_search_with_scores was called with correct embedding
            mock_search.assert_called_once()
            call_args = mock_search.call_args
            query_vector = call_args.kwargs.get("query_vector") or call_args.args[0]
            assert query_vector == test_embedding


class TestGetOrCreatePipeline:
    """Tests for RetrievalPipelineService.get_or_create_pipeline method."""

    @pytest.fixture
    def service(self, session_factory):
        return RetrievalPipelineService(session_factory)

    @pytest.fixture
    def unique_name(self):
        """Generate a unique pipeline name to avoid collisions with stale test data."""
        import uuid

        return f"test_ret_gocp_{uuid.uuid4().hex[:8]}"

    def test_get_or_create_pipeline_new(self, service, unique_name):
        """When no pipeline with the name exists, a new one is created."""
        pipeline_id, is_new = service.get_or_create_pipeline(
            name=unique_name,
            config={"type": "bm25", "tokenizer": "bert"},
        )

        assert is_new is True
        assert pipeline_id is not None

        config = service.get_pipeline_config(pipeline_id)
        assert config == {"type": "bm25", "tokenizer": "bert"}

    def test_get_or_create_pipeline_existing(self, service, unique_name):
        """When a pipeline with the name exists, its ID is returned."""
        pipeline_id, is_new = service.get_or_create_pipeline(
            name=unique_name,
            config={"type": "bm25", "tokenizer": "bert"},
        )
        assert is_new is True

        # Second call should find the existing pipeline
        pipeline_id2, is_new2 = service.get_or_create_pipeline(
            name=unique_name,
            config={"type": "bm25", "tokenizer": "bert"},
        )

        assert is_new2 is False
        assert pipeline_id2 == pipeline_id

    def test_get_or_create_pipeline_config_mismatch(self, service, unique_name, caplog):
        """When pipeline exists with different config, logs a warning and reuses it."""
        import logging

        pipeline_id, _ = service.get_or_create_pipeline(
            name=unique_name,
            config={"type": "bm25", "tokenizer": "bert"},
        )

        with caplog.at_level(logging.WARNING, logger="AutoRAG-Research"):
            pipeline_id2, is_new = service.get_or_create_pipeline(
                name=unique_name,
                config={"type": "bm25", "tokenizer": "spacy"},
            )

        assert is_new is False
        assert pipeline_id2 == pipeline_id
        assert "different config" in caplog.text


class TestRetrievalPipelineResume:
    """Tests for query skip logic in retrieval run_pipeline (resume support)."""

    @pytest.fixture
    def service(self, session_factory):
        return RetrievalPipelineService(session_factory)

    @pytest.fixture
    def mock_retrieval_func(self):
        call_log: list[int | str] = []

        async def retrieval_func(query_id: int | str, top_k: int) -> list[dict]:
            call_log.append(query_id)
            return [{"doc_id": 1, "score": 0.9}]

        retrieval_func.call_log = call_log  # type: ignore[attr-defined]
        return retrieval_func

    def test_run_pipeline_skips_completed_queries(self, service, mock_retrieval_func, session_factory):
        """Pre-insert results for some queries, verify retrieval_func not called for them."""
        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        pipeline_id, _ = service.get_or_create_pipeline(
            name="test_skip_completed_retrieval",
            config={"type": "test"},
        )

        # Pre-insert results for the first query (query_id=1)
        with service._create_uow() as uow:
            queries = uow.queries.get_all(limit=1, offset=0)
            first_query_id = queries[0].id
            uow.chunk_results.bulk_insert([
                {
                    "query_id": first_query_id,
                    "pipeline_id": pipeline_id,
                    "chunk_id": 1,
                    "rel_score": 0.95,
                }
            ])
            uow.commit()

        # Run pipeline - should skip the first query
        result = service.run_pipeline(
            retrieval_func=mock_retrieval_func,
            pipeline_id=pipeline_id,
            top_k=1,
        )

        # The pre-inserted query should NOT appear in the call log
        assert first_query_id not in mock_retrieval_func.call_log
        # Should process remaining queries
        assert result["total_queries"] == query_count - 1

        service.delete_pipeline_results(pipeline_id)

    def test_run_pipeline_all_completed(self, service, mock_retrieval_func, session_factory):
        """When all queries have results, processes 0 queries."""
        pipeline_id, _ = service.get_or_create_pipeline(
            name="test_all_completed_retrieval",
            config={"type": "test"},
        )

        # Pre-insert results for ALL queries
        with service._create_uow() as uow:
            queries = uow.queries.get_all(limit=1000, offset=0)
            results_to_insert = [
                {"query_id": q.id, "pipeline_id": pipeline_id, "chunk_id": 1, "rel_score": 0.9} for q in queries
            ]
            uow.chunk_results.bulk_insert(results_to_insert)
            uow.commit()

        # Run pipeline - should skip everything
        result = service.run_pipeline(
            retrieval_func=mock_retrieval_func,
            pipeline_id=pipeline_id,
            top_k=1,
        )

        assert result["total_queries"] == 0
        assert len(mock_retrieval_func.call_log) == 0

        service.delete_pipeline_results(pipeline_id)

    def test_run_pipeline_backward_compat(self, service, session_factory):
        """New pipeline with no pre-existing results processes all queries."""
        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        async def retrieval_func(query_id: int | str, top_k: int) -> list[dict]:
            return [{"doc_id": 1, "score": 0.9}]

        pipeline_id, _ = service.get_or_create_pipeline(
            name="test_backward_compat_retrieval",
            config={"type": "test"},
        )

        result = service.run_pipeline(
            retrieval_func=retrieval_func,
            pipeline_id=pipeline_id,
            top_k=1,
        )

        assert result["total_queries"] == query_count

        service.delete_pipeline_results(pipeline_id)
