"""Multi-Modal Data Ingestion Service for AutoRAG-Research.

Provides service layer for ingesting multi-modal data including files, documents,
pages, chunks, image chunks, queries, and retrieval ground truth relations.
"""

import logging

from autorag_research.orm.service.base_ingestion import (
    BaseIngestionService,
    ImageEmbeddingFunc,
    ImageMultiVectorEmbeddingFunc,
)
from autorag_research.orm.uow.multi_modal_uow import MultiModalUnitOfWork

logger = logging.getLogger("AutoRAG-Research")


class MultiModalIngestionService(BaseIngestionService):
    """Service for multi-modal data ingestion operations.

    This service provides batch-only methods for ingesting multi-modal RAG datasets.
    Users can access repositories directly via UoW for basic CRUD operations.

    Design Principles:
    - Batch-only methods (no single-add methods)
    - No simple wrappers around repository functions
    - Value-added operations with transaction management and validation
    - Mixed multi-hop support for retrieval ground truth

    Example:
        ```python
        from autorag_research.orm.connection import DBConnection
        from autorag_research.orm.service import MultiModalIngestionService

        # Setup database connection
        db = DBConnection.from_config()  # or DBConnection.from_env()
        session_factory = db.get_session_factory()

        # Initialize service
        service = MultiModalIngestionService(session_factory)

        # Read image file as bytes
        with open("/path/to/image1.jpg", "rb") as f:
            image_bytes = f.read()

        # Batch add files
        file_ids = service.add_files([
            {"path": "/path/to/image1.jpg", "file_type": "image"},
            {"path": "/path/to/document1.pdf", "file_type": "raw"},
        ])
        ```
    """

    def _create_uow(self) -> MultiModalUnitOfWork:
        """Create a new MultiModalUnitOfWork instance.

        Returns:
            New MultiModalUnitOfWork instance.
        """
        return MultiModalUnitOfWork(self.session_factory, self._schema)

    def _get_schema_classes(self) -> dict[str, type]:
        """Get all model classes from schema.

        Returns:
            Dictionary mapping class names to model classes.
        """
        if self._schema is not None:
            return {
                "File": self._schema.File,
                "Document": self._schema.Document,
                "Page": self._schema.Page,
                "Chunk": self._schema.Chunk,
                "ImageChunk": self._schema.ImageChunk,
                "PageChunkRelation": self._schema.PageChunkRelation,
                "Query": self._schema.Query,
                "RetrievalRelation": self._schema.RetrievalRelation,
            }
        from autorag_research.orm.schema import (
            Chunk,
            Document,
            File,
            ImageChunk,
            Page,
            PageChunkRelation,
            Query,
            RetrievalRelation,
        )

        return {
            "File": File,
            "Document": Document,
            "Page": Page,
            "Chunk": Chunk,
            "ImageChunk": ImageChunk,
            "PageChunkRelation": PageChunkRelation,
            "Query": Query,
            "RetrievalRelation": RetrievalRelation,
        }

    # ==================== Batch Add Operations ====================

    def add_files(self, files: list[dict[str, str]]) -> list[int]:
        """Batch add files to the database.

        Args:
            files: List of dictionary (path, file_type).
                   file_type can be: "raw", "image", "audio", "video".

        Returns:
            List of created File IDs.
        """
        return self._add(files, table_name="File", repository_property="files")

    def add_documents(self, documents: list[dict]) -> list[int]:
        """Batch add documents to the database.

        Args:
            documents: List of dicts with keys:
                      - filename (str | None)
                      - title (str | None)
                      - author (str | None)
                      - filepath_id (int | None) - FK to File
                      - metadata (dict | None) - JSONB metadata

        Returns:
            List of created Document IDs.
        """
        return self._add(documents, table_name="Document", repository_property="documents")

    def add_pages(self, pages: list[dict]) -> list[int]:
        """Batch add pages to the database.

        Args:
            pages: List of dicts with keys:
                  - document_id (int) - FK to Document (required)
                  - page_num (int) - Page number (required)
                  - image_content (bytes | None) - Image binary data
                  - mimetype (str | None) - Image MIME type (e.g., "image/png")
                  - metadata (dict | None) - JSONB metadata

        Returns:
            List of created Page IDs.
        """
        return self._add(pages, table_name="Page", repository_property="pages")

    def add_image_chunks(self, image_chunks: list[dict[str, bytes | str | int | None]]) -> list[int | str]:
        """Batch add image chunks to the database.

        Uses memory-efficient bulk insert (SQLAlchemy Core) instead of ORM objects.
        This reduces memory usage by ~3-5x for large batches.

        Args:
            image_chunks: List of dictionary (content, mimetype, parent_page_id).
                         content: Image binary data (required)
                         mimetype: Image MIME type e.g., "image/png" (required)
                         parent_page_id: FK to Page (optional)

        Returns:
            List of created ImageChunk IDs.
        """
        return self._add_bulk(image_chunks, repository_property="image_chunks")

    # ==================== Embedding Operations ====================

    def set_image_chunk_embeddings(
        self,
        image_chunk_ids: list[int],
        embeddings: list[list[float]],
    ) -> int:
        """Batch set embeddings for image chunks.

        Args:
            image_chunk_ids: List of image chunk IDs.
            embeddings: List of embedding vectors (must match image_chunk_ids length).

        Returns:
            Total number of image chunks successfully updated.

        Raises:
            LengthMismatchError: If image_chunk_ids and embeddings have different lengths.
        """
        return self._set_embeddings(image_chunk_ids, embeddings, "image_chunks", is_multi_vector=False)

    def set_image_chunk_multi_embeddings(
        self,
        image_chunk_ids: list[int],
        embeddings: list[list[list[float]]],
    ) -> int:
        """Batch set multi-vector embeddings for image chunks.

        Args:
            image_chunk_ids: List of image chunk IDs.
            embeddings: List of multi-vector embeddings (list of list of floats per image chunk).

        Returns:
            Total number of image chunks successfully updated.

        Raises:
            LengthMismatchError: If image_chunk_ids and embeddings have different lengths.
        """
        return self._set_embeddings(image_chunk_ids, embeddings, "image_chunks", is_multi_vector=True)

    # ==================== Async Batch Embedding Operations ====================

    def embed_all_image_chunks(
        self,
        embed_func: ImageEmbeddingFunc,
        batch_size: int = 100,
        max_concurrency: int = 10,
    ) -> int:
        """Embed all image chunks that don't have embeddings.

        Args:
            embed_func: Async function that takes image bytes and returns embedding vector.
            batch_size: Number of image chunks to process per batch.
            max_concurrency: Maximum concurrent embedding calls.

        Returns:
            Total number of image chunks successfully embedded.
        """
        return self._embed_entities("image_chunk", "single", embed_func, batch_size, max_concurrency)

    def embed_all_image_chunks_multi_vector(
        self,
        embed_func: ImageMultiVectorEmbeddingFunc,
        batch_size: int = 100,
        max_concurrency: int = 10,
    ) -> int:
        """Embed all image chunks that don't have multi-vector embeddings.

        Args:
            embed_func: Async function that takes image bytes and returns multi-vector embedding.
            batch_size: Number of image chunks to process per batch.
            max_concurrency: Maximum concurrent embedding calls.

        Returns:
            Total number of image chunks successfully embedded.
        """
        return self._embed_entities("image_chunk", "multi_vector", embed_func, batch_size, max_concurrency)

    # ==================== Statistics ====================

    def get_statistics(self) -> dict:
        """Get statistics about the ingested data.

        Returns:
            Dictionary with counts for all entity types and embedding status.
        """
        with self._create_uow() as uow:
            total_files = uow.files.count()
            total_documents = uow.documents.count()
            total_pages = uow.pages.count()
            total_chunks = uow.chunks.count()
            total_image_chunks = uow.image_chunks.count()
            total_queries = uow.queries.count()
            total_retrieval_relations = uow.retrieval_relations.count()

            # Embedding status
            chunks_with_emb = len(uow.chunks.get_with_embeddings())
            chunks_without_emb = len(uow.chunks.get_without_embeddings())
            image_chunks_with_emb = len(uow.image_chunks.get_with_embeddings())
            image_chunks_without_emb = len(uow.image_chunks.get_without_embeddings())

            return {
                "files": total_files,
                "documents": total_documents,
                "pages": total_pages,
                "chunks": {
                    "total": total_chunks,
                    "with_embeddings": chunks_with_emb,
                    "without_embeddings": chunks_without_emb,
                },
                "image_chunks": {
                    "total": total_image_chunks,
                    "with_embeddings": image_chunks_with_emb,
                    "without_embeddings": image_chunks_without_emb,
                },
                "queries": total_queries,
                "retrieval_relations": total_retrieval_relations,
            }
