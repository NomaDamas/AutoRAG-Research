"""Test cases for BaseVectorRepository.

Tests all vector search methods with real database operations using random embeddings.
Uses ChunkRepository as the concrete implementation of BaseVectorRepository.
"""

import random

import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.schema import Chunk

EMBEDDING_DIM = 768


def random_embedding(dim: int = EMBEDDING_DIM) -> list[float]:
    """Generate a random embedding vector."""
    return [random.uniform(-1.0, 1.0) for _ in range(dim)]


def normalized_embedding(dim: int = EMBEDDING_DIM) -> list[float]:
    """Generate a normalized random embedding vector (unit length)."""
    vec = [random.uniform(-1.0, 1.0) for _ in range(dim)]
    magnitude = sum(x**2 for x in vec) ** 0.5
    return [x / magnitude for x in vec]


@pytest.fixture
def chunk_repository(db_session: Session) -> ChunkRepository:
    """Create a ChunkRepository instance for testing."""
    return ChunkRepository(db_session)


class TestVectorSearch:
    """Tests for vector_search method."""

    def test_vector_search_returns_results(
        self,
        chunk_repository: ChunkRepository,
        db_session: Session,
    ):
        """Test that vector_search returns chunks ordered by similarity."""
        # Create test chunks with embeddings
        chunk_ids = []
        for i in range(3):
            chunk = Chunk(
                contents=f"Vector search test {i}",
                embedding=normalized_embedding(),
            )
            db_session.add(chunk)
            db_session.flush()
            chunk_ids.append(chunk.id)
        db_session.commit()

        query_vector = normalized_embedding()
        results = chunk_repository.vector_search(
            query_vector=query_vector,
            vector_column="embedding",
            limit=3,
        )

        assert len(results) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in results)

        # Cleanup - fetch fresh entities to avoid relationship loading issues
        for chunk_id in chunk_ids:
            chunk = db_session.get(Chunk, chunk_id)
            if chunk:
                db_session.delete(chunk)
        db_session.commit()

    def test_vector_search_respects_limit(
        self,
        chunk_repository: ChunkRepository,
        db_session: Session,
    ):
        """Test that vector_search respects the limit parameter."""
        # Create test chunks
        chunk_ids = []
        for i in range(5):
            chunk = Chunk(
                contents=f"Limit test {i}",
                embedding=normalized_embedding(),
            )
            db_session.add(chunk)
            db_session.flush()
            chunk_ids.append(chunk.id)
        db_session.commit()

        query_vector = normalized_embedding()
        results = chunk_repository.vector_search(
            query_vector=query_vector,
            vector_column="embedding",
            limit=2,
        )

        assert len(results) <= 2

        # Cleanup
        for chunk_id in chunk_ids:
            chunk = db_session.get(Chunk, chunk_id)
            if chunk:
                db_session.delete(chunk)
        db_session.commit()

    def test_vector_search_empty_query_returns_empty(
        self,
        chunk_repository: ChunkRepository,
    ):
        """Test that vector_search with empty query vector returns empty list."""
        results = chunk_repository.vector_search(
            query_vector=[],
            vector_column="embedding",
            limit=10,
        )

        assert results == []

    def test_vector_search_finds_similar_vector(
        self,
        chunk_repository: ChunkRepository,
        db_session: Session,
    ):
        """Test that vector_search finds the most similar chunk."""
        # Create a chunk with a known embedding
        known_embedding = normalized_embedding()
        chunk = Chunk(contents="Known embedding chunk", embedding=known_embedding)
        db_session.add(chunk)
        db_session.flush()
        chunk_id = chunk.id
        db_session.commit()

        # Search with the same embedding - should find it as most similar
        results = chunk_repository.vector_search(
            query_vector=known_embedding,
            vector_column="embedding",
            limit=1,
        )

        assert len(results) == 1
        assert results[0].id == chunk_id

        # Cleanup
        chunk = db_session.get(Chunk, chunk_id)
        if chunk:
            db_session.delete(chunk)
        db_session.commit()


class TestVectorSearchWithScores:
    """Tests for vector_search_with_scores method."""

    def test_vector_search_with_scores_returns_tuples(
        self,
        chunk_repository: ChunkRepository,
        db_session: Session,
    ):
        """Test that vector_search_with_scores returns (entity, score) tuples."""
        # Create test chunks
        chunk_ids = []
        for i in range(3):
            chunk = Chunk(
                contents=f"Score test {i}",
                embedding=normalized_embedding(),
            )
            db_session.add(chunk)
            db_session.flush()
            chunk_ids.append(chunk.id)
        db_session.commit()

        query_vector = normalized_embedding()
        results = chunk_repository.vector_search_with_scores(
            query_vector=query_vector,
            vector_column="embedding",
            limit=3,
        )

        assert len(results) >= 1
        for chunk, score in results:
            assert isinstance(chunk, Chunk)
            assert isinstance(score, float)

        # Cleanup
        for chunk_id in chunk_ids:
            chunk = db_session.get(Chunk, chunk_id)
            if chunk:
                db_session.delete(chunk)
        db_session.commit()

    def test_vector_search_with_scores_ordered_by_distance(
        self,
        chunk_repository: ChunkRepository,
        db_session: Session,
    ):
        """Test that results are ordered by distance (ascending)."""
        # Create test chunks
        chunk_ids = []
        for i in range(5):
            chunk = Chunk(
                contents=f"Order test {i}",
                embedding=normalized_embedding(),
            )
            db_session.add(chunk)
            db_session.flush()
            chunk_ids.append(chunk.id)
        db_session.commit()

        query_vector = normalized_embedding()
        results = chunk_repository.vector_search_with_scores(
            query_vector=query_vector,
            vector_column="embedding",
            limit=5,
        )

        scores = [score for _, score in results]
        assert scores == sorted(scores), "Results should be ordered by distance ascending"

        # Cleanup
        for chunk_id in chunk_ids:
            chunk = db_session.get(Chunk, chunk_id)
            if chunk:
                db_session.delete(chunk)
        db_session.commit()

    def test_vector_search_with_scores_empty_query_returns_empty(
        self,
        chunk_repository: ChunkRepository,
    ):
        """Test that empty query vector returns empty list."""
        results = chunk_repository.vector_search_with_scores(
            query_vector=[],
            vector_column="embedding",
            limit=10,
        )

        assert results == []

    def test_vector_search_with_scores_same_vector_has_zero_distance(
        self,
        chunk_repository: ChunkRepository,
        db_session: Session,
    ):
        """Test that searching with the same vector returns distance close to 0."""
        known_embedding = normalized_embedding()
        chunk = Chunk(contents="Zero distance chunk", embedding=known_embedding)
        db_session.add(chunk)
        db_session.flush()
        chunk_id = chunk.id
        db_session.commit()

        results = chunk_repository.vector_search_with_scores(
            query_vector=known_embedding,
            vector_column="embedding",
            limit=1,
        )

        assert len(results) == 1
        found_chunk, distance = results[0]
        assert found_chunk.id == chunk_id
        # Cosine distance of same vector should be very close to 0
        assert distance < 0.01, f"Expected distance ~0, got {distance}"

        # Cleanup
        chunk = db_session.get(Chunk, chunk_id)
        if chunk:
            db_session.delete(chunk)
        db_session.commit()

    def test_vector_search_with_scores_respects_limit(
        self,
        chunk_repository: ChunkRepository,
        db_session: Session,
    ):
        """Test that limit parameter is respected."""
        # Create test chunks
        chunk_ids = []
        for i in range(5):
            chunk = Chunk(
                contents=f"Limit score test {i}",
                embedding=normalized_embedding(),
            )
            db_session.add(chunk)
            db_session.flush()
            chunk_ids.append(chunk.id)
        db_session.commit()

        query_vector = normalized_embedding()
        results = chunk_repository.vector_search_with_scores(
            query_vector=query_vector,
            vector_column="embedding",
            limit=2,
        )

        assert len(results) <= 2

        # Cleanup
        for chunk_id in chunk_ids:
            chunk = db_session.get(Chunk, chunk_id)
            if chunk:
                db_session.delete(chunk)
        db_session.commit()


class TestSetMultiVectorEmbedding:
    """Tests for set_multi_vector_embedding method."""

    def test_set_multi_vector_embedding_success(
        self,
        chunk_repository: ChunkRepository,
        db_session: Session,
    ):
        """Test setting multi-vector embedding on a chunk."""
        chunk = Chunk(contents="Multi-embedding test chunk")
        db_session.add(chunk)
        db_session.flush()
        chunk_id = chunk.id
        db_session.commit()

        # Set multi-vector embeddings (3 vectors)
        multi_embeddings = [normalized_embedding() for _ in range(3)]

        success = chunk_repository.set_multi_vector_embedding(
            entity_id=chunk_id,
            embeddings=multi_embeddings,
            vector_column="embeddings",
        )
        db_session.commit()

        assert success is True

        # Verify embeddings were set
        refreshed_chunk = db_session.get(Chunk, chunk_id)
        db_session.refresh(refreshed_chunk)
        assert refreshed_chunk.embeddings is not None
        assert len(refreshed_chunk.embeddings) == 3

        # Cleanup
        chunk = db_session.get(Chunk, chunk_id)
        if chunk:
            db_session.delete(chunk)
        db_session.commit()

    def test_set_multi_vector_embedding_empty_returns_false(
        self,
        chunk_repository: ChunkRepository,
        db_session: Session,
    ):
        """Test that empty embeddings list returns False."""
        chunk = Chunk(contents="Empty embedding test chunk")
        db_session.add(chunk)
        db_session.flush()
        chunk_id = chunk.id
        db_session.commit()

        success = chunk_repository.set_multi_vector_embedding(
            entity_id=chunk_id,
            embeddings=[],
            vector_column="embeddings",
        )

        assert success is False

        # Cleanup
        chunk = db_session.get(Chunk, chunk_id)
        if chunk:
            db_session.delete(chunk)
        db_session.commit()

    def test_set_multi_vector_embedding_nonexistent_id_returns_false(
        self,
        chunk_repository: ChunkRepository,
    ):
        """Test that non-existent entity ID returns False."""
        multi_embeddings = [normalized_embedding() for _ in range(3)]

        success = chunk_repository.set_multi_vector_embedding(
            entity_id=999999,  # Non-existent ID
            embeddings=multi_embeddings,
            vector_column="embeddings",
        )

        assert success is False


class TestSetMultiVectorEmbeddingsBatch:
    """Tests for set_multi_vector_embeddings_batch method."""

    def test_set_multi_vector_embeddings_batch_success(
        self,
        chunk_repository: ChunkRepository,
        db_session: Session,
    ):
        """Test batch setting multi-vector embeddings."""
        # Create test chunks
        chunk_ids = []
        for i in range(3):
            chunk = Chunk(contents=f"Batch test chunk {i}")
            db_session.add(chunk)
            db_session.flush()
            chunk_ids.append(chunk.id)
        db_session.commit()

        embeddings_list = [[normalized_embedding() for _ in range(2)] for _ in range(3)]

        updated = chunk_repository.set_multi_vector_embeddings_batch(
            entity_ids=chunk_ids,
            embeddings_list=embeddings_list,
            vector_column="embeddings",
        )
        db_session.commit()

        assert updated == 3

        # Verify all chunks have embeddings
        for chunk_id in chunk_ids:
            refreshed = db_session.get(Chunk, chunk_id)
            db_session.refresh(refreshed)
            assert refreshed.embeddings is not None
            assert len(refreshed.embeddings) == 2

        # Cleanup
        for chunk_id in chunk_ids:
            chunk = db_session.get(Chunk, chunk_id)
            if chunk:
                db_session.delete(chunk)
        db_session.commit()

    def test_set_multi_vector_embeddings_batch_length_mismatch_raises(
        self,
        chunk_repository: ChunkRepository,
    ):
        """Test that mismatched list lengths raise error."""
        from autorag_research.exceptions import LengthMismatchError

        with pytest.raises(LengthMismatchError):
            chunk_repository.set_multi_vector_embeddings_batch(
                entity_ids=[1, 2, 3],
                embeddings_list=[[normalized_embedding()]],  # Only 1 item
                vector_column="embeddings",
            )


class TestMaxSimSearch:
    """Tests for maxsim_search method."""

    def test_maxsim_search_returns_results(
        self,
        chunk_repository: ChunkRepository,
        db_session: Session,
    ):
        """Test that maxsim_search returns (entity, score) tuples."""
        # Create chunks with multi-vector embeddings
        chunk_ids = []
        for i in range(3):
            chunk = Chunk(contents=f"MaxSim test chunk {i}")
            db_session.add(chunk)
            db_session.flush()
            chunk_ids.append(chunk.id)

            # Set multi-vector embeddings
            multi_embeddings = [normalized_embedding() for _ in range(2)]
            chunk_repository.set_multi_vector_embedding(
                entity_id=chunk.id,
                embeddings=multi_embeddings,
                vector_column="embeddings",
            )
        db_session.commit()

        # Search with multi-vector query
        query_vectors = [normalized_embedding() for _ in range(2)]

        results = chunk_repository.maxsim_search(
            query_vectors=query_vectors,
            vector_column="embeddings",
            limit=3,
        )

        assert len(results) >= 1
        for chunk, score in results:
            assert isinstance(chunk, Chunk)
            assert isinstance(score, float)

        # Cleanup
        for chunk_id in chunk_ids:
            chunk = db_session.get(Chunk, chunk_id)
            if chunk:
                db_session.delete(chunk)
        db_session.commit()

    def test_maxsim_search_empty_query_returns_empty(
        self,
        chunk_repository: ChunkRepository,
    ):
        """Test that empty query vectors returns empty list."""
        results = chunk_repository.maxsim_search(
            query_vectors=[],
            vector_column="embeddings",
            limit=10,
        )

        assert results == []

    def test_maxsim_search_ordered_by_distance(
        self,
        chunk_repository: ChunkRepository,
        db_session: Session,
    ):
        """Test that results are ordered by MaxSim distance."""
        # Create chunks with multi-vector embeddings
        chunk_ids = []
        for i in range(5):
            chunk = Chunk(contents=f"MaxSim order test chunk {i}")
            db_session.add(chunk)
            db_session.flush()
            chunk_ids.append(chunk.id)

            multi_embeddings = [normalized_embedding() for _ in range(2)]
            chunk_repository.set_multi_vector_embedding(
                entity_id=chunk.id,
                embeddings=multi_embeddings,
                vector_column="embeddings",
            )
        db_session.commit()

        query_vectors = [normalized_embedding() for _ in range(2)]

        results = chunk_repository.maxsim_search(
            query_vectors=query_vectors,
            vector_column="embeddings",
            limit=5,
        )

        scores = [score for _, score in results]
        assert scores == sorted(scores), "Results should be ordered by distance ascending"

        # Cleanup
        for chunk_id in chunk_ids:
            chunk = db_session.get(Chunk, chunk_id)
            if chunk:
                db_session.delete(chunk)
        db_session.commit()


class TestMaxSimSearchWithIds:
    """Tests for maxsim_search_with_ids method."""

    def test_maxsim_search_with_ids_returns_id_score_tuples(
        self,
        chunk_repository: ChunkRepository,
        db_session: Session,
    ):
        """Test that maxsim_search_with_ids returns (id, score) tuples."""
        # Create chunks with multi-vector embeddings
        chunk_ids = []
        for i in range(3):
            chunk = Chunk(contents=f"MaxSim ID test chunk {i}")
            db_session.add(chunk)
            db_session.flush()
            chunk_ids.append(chunk.id)

            multi_embeddings = [normalized_embedding() for _ in range(2)]
            chunk_repository.set_multi_vector_embedding(
                entity_id=chunk.id,
                embeddings=multi_embeddings,
                vector_column="embeddings",
            )
        db_session.commit()

        query_vectors = [normalized_embedding() for _ in range(2)]

        results = chunk_repository.maxsim_search_with_ids(
            query_vectors=query_vectors,
            vector_column="embeddings",
            limit=3,
        )

        assert len(results) >= 1
        for entity_id, score in results:
            assert isinstance(entity_id, int)
            assert isinstance(score, float)

        # Cleanup
        for chunk_id in chunk_ids:
            chunk = db_session.get(Chunk, chunk_id)
            if chunk:
                db_session.delete(chunk)
        db_session.commit()

    def test_maxsim_search_with_ids_empty_query_returns_empty(
        self,
        chunk_repository: ChunkRepository,
    ):
        """Test that empty query vectors returns empty list."""
        results = chunk_repository.maxsim_search_with_ids(
            query_vectors=[],
            vector_column="embeddings",
            limit=10,
        )

        assert results == []

    def test_maxsim_search_with_ids_respects_limit(
        self,
        chunk_repository: ChunkRepository,
        db_session: Session,
    ):
        """Test that limit parameter is respected."""
        # Create chunks with multi-vector embeddings
        chunk_ids = []
        for i in range(5):
            chunk = Chunk(contents=f"MaxSim limit test chunk {i}")
            db_session.add(chunk)
            db_session.flush()
            chunk_ids.append(chunk.id)

            multi_embeddings = [normalized_embedding() for _ in range(2)]
            chunk_repository.set_multi_vector_embedding(
                entity_id=chunk.id,
                embeddings=multi_embeddings,
                vector_column="embeddings",
            )
        db_session.commit()

        query_vectors = [normalized_embedding() for _ in range(2)]

        results = chunk_repository.maxsim_search_with_ids(
            query_vectors=query_vectors,
            vector_column="embeddings",
            limit=2,
        )

        assert len(results) <= 2

        # Cleanup
        for chunk_id in chunk_ids:
            chunk = db_session.get(Chunk, chunk_id)
            if chunk:
                db_session.delete(chunk)
        db_session.commit()
