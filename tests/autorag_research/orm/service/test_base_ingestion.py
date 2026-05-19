import logging

import pytest

from autorag_research.exceptions import DuplicateRetrievalGTError, LengthMismatchError
from autorag_research.orm.models.retrieval_gt import TextId, text
from autorag_research.orm.schema import Chunk, Query, RetrievalRelation
from autorag_research.orm.service.base_ingestion import BaseIngestionService
from autorag_research.orm.uow.text_uow import TextOnlyUnitOfWork


class ConcreteTestIngestionService(BaseIngestionService):
    def _create_uow(self) -> TextOnlyUnitOfWork:
        return TextOnlyUnitOfWork(self.session_factory, self._schema)

    def _get_schema_classes(self) -> dict[str, type]:
        return {
            "Query": Query,
            "Chunk": Chunk,
            "RetrievalRelation": RetrievalRelation,
        }


class TestBaseIngestionService:
    @pytest.fixture
    def service(self, session_factory):
        return ConcreteTestIngestionService(session_factory)

    def test_add_chunks(self, service):
        chunks = [
            {"contents": "test chunk A"},
            {"contents": "test chunk B"},
        ]
        ids = service.add_chunks(chunks)
        assert len(ids) == 2
        assert all(isinstance(id_, int) for id_ in ids)

        with service._create_uow() as uow:
            for id_ in ids:
                chunk = uow.chunks.get_by_id(id_)
                assert chunk is not None
                uow.chunks.delete(chunk)
            uow.commit()

    def test_add_queries(self, service):
        queries = [
            {"contents": "test query 1", "generation_gt": ["answer1"]},
            {"contents": "test query 2", "generation_gt": None},
        ]
        ids = service.add_queries(queries)
        assert len(ids) == 2
        assert all(isinstance(id_, int) for id_ in ids)

        with service._create_uow() as uow:
            for id_ in ids:
                query = uow.queries.get_by_id(id_)
                assert query is not None
                uow.queries.delete(query)
            uow.commit()

    def test_set_query_embeddings(self, service):
        query_ids = [1, 2]
        embeddings = [[0.1] * 768, [0.2] * 768]
        updated = service.set_query_embeddings(query_ids, embeddings)
        assert updated == 2

        with service._create_uow() as uow:
            for qid in query_ids:
                query = uow.queries.get_by_id(qid)
                query.embedding = None
            uow.commit()

    def test_set_query_embeddings_length_mismatch(self, service):
        with pytest.raises(LengthMismatchError):
            service.set_query_embeddings([1, 2], [[0.1] * 768])

    def test_set_chunk_embeddings(self, service):
        chunk_ids = [1, 2]
        embeddings = [[0.3] * 768, [0.4] * 768]
        updated = service.set_chunk_embeddings(chunk_ids, embeddings)
        assert updated == 2

        with service._create_uow() as uow:
            for cid in chunk_ids:
                chunk = uow.chunks.get_by_id(cid)
                chunk.embedding = None
            uow.commit()

    def test_set_query_multi_embeddings(self, service):
        query_ids = [1, 2]
        # Use 768 dimensions to match schema
        multi_embeddings = [[[0.1] * 768] * 3, [[0.2] * 768] * 3]
        updated = service.set_query_multi_embeddings(query_ids, multi_embeddings)
        assert updated == 2

        with service._create_uow() as uow:
            for qid in query_ids:
                query = uow.queries.get_by_id(qid)
                query.embeddings = None
            uow.commit()

    def test_set_chunk_multi_embeddings(self, service):
        chunk_ids = [1, 2]
        # Use 768 dimensions to match schema
        multi_embeddings = [[[0.3] * 768] * 3, [[0.4] * 768] * 3]
        updated = service.set_chunk_multi_embeddings(chunk_ids, multi_embeddings)
        assert updated == 2

        with service._create_uow() as uow:
            for cid in chunk_ids:
                chunk = uow.chunks.get_by_id(cid)
                chunk.embeddings = None
            uow.commit()

    def test_add_queries_skip_duplicates(self, service):
        """Test add_queries with skip_duplicates=True skips existing queries."""
        # First insert
        queries = [
            {"contents": "skip dup query 1"},
            {"contents": "skip dup query 2"},
        ]
        first_ids = service.add_queries(queries)
        assert len(first_ids) == 2

        # Second insert: all with explicit IDs (duplicates + new)
        new_explicit_id = first_ids[1] + 1000
        overlapping_queries = [
            {"id": first_ids[0], "contents": "duplicate query 1"},
            {"id": first_ids[1], "contents": "duplicate query 2"},
            {"id": new_explicit_id, "contents": "brand new query"},
        ]
        new_ids = service.add_queries(overlapping_queries, skip_duplicates=True)

        # Only the new query should be returned
        assert len(new_ids) == 1
        assert new_ids[0] == new_explicit_id

        # Verify original content is unchanged
        with service._create_uow() as uow:
            q1 = uow.queries.get_by_id(first_ids[0])
            assert q1.contents == "skip dup query 1"

        # Cleanup
        with service._create_uow() as uow:
            for id_ in [*first_ids, new_explicit_id]:
                entity = uow.queries.get_by_id(id_)
                if entity:
                    uow.queries.delete(entity)
            uow.commit()

    def test_add_chunks_skip_duplicates(self, service):
        """Test add_chunks with skip_duplicates=True skips existing chunks."""
        # First insert
        chunks = [{"contents": "skip dup chunk 1"}]
        first_ids = service.add_chunks(chunks)
        assert len(first_ids) == 1

        # Second insert: all with explicit IDs (duplicate + new)
        new_explicit_id = first_ids[0] + 1000
        overlapping_chunks = [
            {"id": first_ids[0], "contents": "duplicate chunk"},
            {"id": new_explicit_id, "contents": "brand new chunk"},
        ]
        new_ids = service.add_chunks(overlapping_chunks, skip_duplicates=True)

        # Only the new chunk should be returned
        assert len(new_ids) == 1
        assert new_ids[0] == new_explicit_id

        # Cleanup
        with service._create_uow() as uow:
            for id_ in [first_ids[0], new_explicit_id]:
                entity = uow.chunks.get_by_id(id_)
                if entity:
                    uow.chunks.delete(entity)
            uow.commit()

    def test_add_retrieval_gt_text_mode(self, service):
        # Test upsert=True overwrites existing relations
        pks = service.add_retrieval_gt(query_id=1, gt=text(1) | text(2), chunk_type="text", upsert=True)
        assert len(pks) == 2
        assert all(pk[0] == 1 for pk in pks)

        # Test upsert=False raises error when relation already exists
        with pytest.raises(DuplicateRetrievalGTError):
            service.add_retrieval_gt(query_id=1, gt=text(1), chunk_type="text", upsert=False)

        with service._create_uow() as uow:
            uow.retrieval_relations.delete_by_query_id(1)
            original_rel = RetrievalRelation(
                query_id=1, group_index=0, group_order=0, chunk_id=1, image_chunk_id=None, score=2
            )
            uow.retrieval_relations.add(original_rel)
            uow.commit()

    def test_add_retrieval_gt_mixed_mode(self, service):
        # Test upsert=True overwrites existing relations
        pks = service.add_retrieval_gt(query_id=2, gt=TextId(3), chunk_type="mixed", upsert=True)
        assert len(pks) == 1
        assert pks[0][0] == 2

        # Test upsert=False raises error when relation already exists
        with pytest.raises(DuplicateRetrievalGTError):
            service.add_retrieval_gt(query_id=2, gt=TextId(3), chunk_type="mixed", upsert=False)

        with service._create_uow() as uow:
            uow.retrieval_relations.delete_by_query_id(2)
            original_rel = RetrievalRelation(
                query_id=2, group_index=0, group_order=0, chunk_id=3, image_chunk_id=None, score=1
            )
            uow.retrieval_relations.add(original_rel)
            uow.commit()

    def test_add_retrieval_gt_batch(self, service):
        items = [
            (3, text(4)),
            (4, text(5)),
        ]
        # Test upsert=True overwrites existing relations
        pks = service.add_retrieval_gt_batch(items, chunk_type="text", upsert=True)
        assert len(pks) == 2

        # Test upsert=False raises error when relation already exists
        with pytest.raises(DuplicateRetrievalGTError):
            service.add_retrieval_gt_batch(items, chunk_type="text", upsert=False)

        with service._create_uow() as uow:
            uow.retrieval_relations.delete_by_query_id(3)
            uow.retrieval_relations.delete_by_query_id(4)
            original_rel_3 = RetrievalRelation(
                query_id=3, group_index=0, group_order=0, chunk_id=None, image_chunk_id=1
            )
            original_rel_4 = RetrievalRelation(
                query_id=4, group_index=0, group_order=0, chunk_id=None, image_chunk_id=2
            )
            uow.retrieval_relations.add(original_rel_3)
            uow.retrieval_relations.add(original_rel_4)
            uow.commit()

    def test_embed_all_queries_skips_failed_items_without_infinite_loop(self, service, caplog):
        """Regression for #239: failing embeddings must not cause an infinite loop.

        The previous implementation re-fetched rows whose embed call returned
        None forever, since the SQL filter is "embedding IS NULL". This test
        proves the loop terminates, the good row gets embedded, the bad row
        stays unembedded, and the summary log surfaces the skip count.
        """
        snapshots = _snapshot_unembedded_queries(service)
        added_ids = service.add_queries([
            {"contents": "embed-fail-test good query", "generation_gt": None},
            {"contents": "embed-fail-test BAD query", "generation_gt": None},
        ])
        good_id, bad_id = added_ids

        try:
            with service._create_uow() as uow:
                uow.queries.get_by_id(good_id).embedding = None
                uow.queries.get_by_id(bad_id).embedding = None
                uow.commit()

            async def flaky_embed(text_value: str) -> list[float]:
                if "BAD" in text_value:
                    msg = "simulated embedding failure"
                    raise RuntimeError(msg)
                return [0.42] * 768

            with caplog.at_level(logging.INFO, logger="AutoRAG-Research"):
                embedded = service.embed_all_queries(
                    flaky_embed,
                    batch_size=10,
                    max_concurrency=2,
                    bm25_tokenizer=None,
                )

            assert embedded >= 1

            with service._create_uow() as uow:
                good = uow.queries.get_by_id(good_id)
                bad = uow.queries.get_by_id(bad_id)
                assert good.embedding is not None, "good query must have been embedded"
                assert bad.embedding is None, "bad query must have been skipped, not embedded"

            summary_messages = [r.message for r in caplog.records if "skipped_failed=" in r.message]
            assert summary_messages, "expected final summary log with skipped_failed counter"
            assert any("skipped_failed=1" in msg for msg in summary_messages), (
                f"expected skipped_failed=1 in summary log, got: {summary_messages}"
            )

            warn_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
            assert any("failed to embed" in msg for msg in warn_messages), (
                f"expected per-batch failure warning, got: {warn_messages}"
            )

        finally:
            with service._create_uow() as uow:
                for qid in added_ids:
                    entity = uow.queries.get_by_id(qid)
                    if entity is not None:
                        uow.queries.delete(entity)
                uow.commit()
            _restore_unembedded_queries(service, snapshots)

    def test_embed_all_queries_terminates_when_every_item_fails(self, service, caplog):
        """All-failure case: loop must terminate (not spin) when nothing can be embedded."""
        snapshots = _snapshot_unembedded_queries(service)
        added_ids = service.add_queries([
            {"contents": "all-fail-test query A", "generation_gt": None},
            {"contents": "all-fail-test query B", "generation_gt": None},
        ])

        try:
            with service._create_uow() as uow:
                for qid in added_ids:
                    uow.queries.get_by_id(qid).embedding = None
                uow.commit()

            async def always_fail(_text: str) -> list[float]:
                msg = "simulated total failure"
                raise RuntimeError(msg)

            with caplog.at_level(logging.INFO, logger="AutoRAG-Research"):
                embedded = service.embed_all_queries(
                    always_fail,
                    batch_size=10,
                    max_concurrency=2,
                    bm25_tokenizer=None,
                )

            assert embedded == 0
            with service._create_uow() as uow:
                for qid in added_ids:
                    assert uow.queries.get_by_id(qid).embedding is None

            summary_messages = [r.message for r in caplog.records if "skipped_failed=" in r.message]
            assert summary_messages, "expected final summary log with skipped_failed counter"

        finally:
            with service._create_uow() as uow:
                for qid in added_ids:
                    entity = uow.queries.get_by_id(qid)
                    if entity is not None:
                        uow.queries.delete(entity)
                uow.commit()
            _restore_unembedded_queries(service, snapshots)


def _snapshot_unembedded_queries(service: BaseIngestionService) -> list[int]:
    """Capture IDs of seed queries whose embedding is None.

    Because `embed_all_queries()` is a global operation against the entire
    database, our tests would otherwise embed seed rows that other tests
    rely on staying un-embedded (see test_retrieval_pipeline_vector.py).
    Snapshot + restore keeps the shared seed state intact.
    """
    with service._create_uow() as uow:
        return [q.id for q in uow.queries.get_without_embeddings(limit=10_000)]


def _restore_unembedded_queries(service: BaseIngestionService, ids: list[int]) -> None:
    """Reset previously-unembedded seed queries back to embedding=None."""
    with service._create_uow() as uow:
        for qid in ids:
            entity = uow.queries.get_by_id(qid)
            if entity is not None:
                entity.embedding = None
        uow.commit()
