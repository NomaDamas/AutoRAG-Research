"""Multi-Modal Data Ingestion Service for AutoRAG-Research.

Provides service layer for ingesting multi-modal data including files, documents,
pages, captions, chunks, image chunks, queries, and retrieval ground truth relations.
"""

import logging

from autorag_research.exceptions import SessionNotSetError
from autorag_research.orm.repository.multi_modal_uow import MultiModalUnitOfWork
from autorag_research.orm.service.base_ingestion import (
    BaseIngestionService,
    ImageEmbeddingFunc,
    ImageMultiVectorEmbeddingFunc,
    TextMultiVectorEmbeddingFunc,
)

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
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from autorag_research.orm.service import MultiModalIngestionService

        # Setup database connection
        engine = create_engine("postgresql://user:pass@localhost/dbname")
        session_factory = sessionmaker(bind=engine)

        # Initialize service
        service = MultiModalIngestionService(session_factory)

        # Read image file as bytes
        with open("/path/to/image1.jpg", "rb") as f:
            image_bytes = f.read()

        # Ingest image pages (convenience method)
        # Images are stored directly in the database as BYTEA
        results = service.ingest_image_pages(
            document_id=1,
            pages_data=[
                (1, image_bytes, "image/jpeg", "Caption for page 1"),
                (2, image_bytes, "image/jpeg", "Caption for page 2"),
            ]
        )

        # Add mixed multi-hop retrieval ground truth
        service.add_retrieval_gt_multihop_mixed(
            query_id=1,
            groups=[
                [("chunk", 1), ("chunk", 2)],       # First hop: text chunks
                [("image_chunk", 1)],               # Second hop: image chunk
            ]
        )
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
                "Caption": self._schema.Caption,
                "Chunk": self._schema.Chunk,
                "ImageChunk": self._schema.ImageChunk,
                "Query": self._schema.Query,
                "RetrievalRelation": self._schema.RetrievalRelation,
            }
        from autorag_research.orm.schema import (
            Caption,
            Chunk,
            Document,
            File,
            ImageChunk,
            Page,
            Query,
            RetrievalRelation,
        )

        return {
            "File": File,
            "Document": Document,
            "Page": Page,
            "Caption": Caption,
            "Chunk": Chunk,
            "ImageChunk": ImageChunk,
            "Query": Query,
            "RetrievalRelation": RetrievalRelation,
        }

    # ==================== Batch Add Operations ====================

    def add_files(self, files: list[tuple[str, str]]) -> list[int]:
        """Batch add files to the database.

        Args:
            files: List of tuples (path, file_type).
                   file_type can be: "raw", "image", "audio", "video".

        Returns:
            List of created File IDs.
        """
        classes = self._get_schema_classes()
        File = classes["File"]

        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            file_entities = [File(path=path, type=file_type) for path, file_type in files]
            uow.files.add_all(file_entities)
            uow.flush()
            file_ids = [f.id for f in file_entities]
            uow.commit()
            return file_ids

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
        classes = self._get_schema_classes()
        Document = classes["Document"]

        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            doc_entities = [
                Document(
                    filename=doc.get("filename"),
                    title=doc.get("title"),
                    author=doc.get("author"),
                    filepath=doc.get("filepath_id"),
                    doc_metadata=doc.get("metadata"),
                )
                for doc in documents
            ]
            uow.documents.add_all(doc_entities)
            uow.flush()
            doc_ids = [d.id for d in doc_entities]
            uow.commit()
            return doc_ids

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
        classes = self._get_schema_classes()
        Page = classes["Page"]

        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            page_entities = [
                Page(
                    document_id=page["document_id"],
                    page_num=page["page_num"],
                    image_content=page.get("image_content"),
                    mimetype=page.get("mimetype"),
                    page_metadata=page.get("metadata"),
                )
                for page in pages
            ]
            uow.pages.add_all(page_entities)
            uow.flush()
            page_ids = [p.id for p in page_entities]
            uow.commit()
            return page_ids

    def add_captions(self, captions: list[tuple[int, str]]) -> list[int]:
        """Batch add captions to the database.

        Args:
            captions: List of tuples (page_id, contents).

        Returns:
            List of created Caption IDs.
        """
        classes = self._get_schema_classes()
        Caption = classes["Caption"]

        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            caption_entities = [Caption(page_id=page_id, contents=contents) for page_id, contents in captions]
            uow.captions.add_all(caption_entities)
            uow.flush()
            caption_ids = [c.id for c in caption_entities]
            uow.commit()
            return caption_ids

    def add_chunks(self, chunks: list[tuple[str, int | None]]) -> list[int]:
        """Batch add text chunks to the database.

        Args:
            chunks: List of tuples (contents, parent_caption_id).
                   parent_caption_id can be None for standalone chunks.

        Returns:
            List of created Chunk IDs.
        """
        classes = self._get_schema_classes()
        Chunk = classes["Chunk"]

        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            chunk_entities = [
                Chunk(contents=contents, parent_caption=parent_caption_id) for contents, parent_caption_id in chunks
            ]
            uow.chunks.add_all(chunk_entities)
            uow.flush()
            chunk_ids = [c.id for c in chunk_entities]
            uow.commit()
            return chunk_ids

    def add_image_chunks(self, image_chunks: list[tuple[bytes, str, int | None]]) -> list[int]:
        """Batch add image chunks to the database.

        Args:
            image_chunks: List of tuples (content, mimetype, parent_page_id).
                         content: Image binary data (required)
                         mimetype: Image MIME type e.g., "image/png" (required)
                         parent_page_id: FK to Page (optional)

        Returns:
            List of created ImageChunk IDs.
        """
        classes = self._get_schema_classes()
        ImageChunk = classes["ImageChunk"]

        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            image_chunk_entities = [
                ImageChunk(content=content, mimetype=mimetype, parent_page=parent_page_id)
                for content, mimetype, parent_page_id in image_chunks
            ]
            uow.image_chunks.add_all(image_chunk_entities)
            uow.flush()
            image_chunk_ids = [ic.id for ic in image_chunk_entities]
            uow.commit()
            return image_chunk_ids

    def add_queries(self, queries: list[tuple[str, list[str] | None]]) -> list[int]:
        """Batch add queries to the database.

        Args:
            queries: List of tuples (query_text, generation_gt).
                    generation_gt can be None.

        Returns:
            List of created Query IDs.
        """
        classes = self._get_schema_classes()
        Query = classes["Query"]

        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            query_entities = [Query(query=query_text, generation_gt=gen_gt) for query_text, gen_gt in queries]
            uow.queries.add_all(query_entities)
            uow.flush()
            query_ids = [q.id for q in query_entities]
            uow.commit()
            return query_ids

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

    def embed_all_queries_multi_vector(
        self,
        embed_func: TextMultiVectorEmbeddingFunc,
        batch_size: int = 100,
        max_concurrency: int = 10,
    ) -> int:
        """Embed all queries that don't have multi-vector embeddings.

        Args:
            embed_func: Async function that takes query text and returns multi-vector embedding.
            batch_size: Number of queries to process per batch.
            max_concurrency: Maximum concurrent embedding calls.

        Returns:
            Total number of queries successfully embedded.
        """
        return self._embed_entities("query", "multi_vector", embed_func, batch_size, max_concurrency)

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

    # ==================== Convenience Methods ====================

    def ingest_image_pages(
        self,
        document_id: int,
        pages_data: list[tuple[int, bytes, str, str]],
    ) -> list[tuple[int, int, int]]:
        """Batch ingest image pages with captions and image chunks.

        Creates Page, Caption, and ImageChunk entities in a single transaction
        for each page. Images are stored directly in the database as BYTEA.
        This is a convenience method for common multi-modal ingestion.

        Args:
            document_id: The document ID to add pages to.
            pages_data: List of tuples (page_num, image_content, mimetype, caption_text).
                       - page_num: Page number
                       - image_content: Image binary data
                       - mimetype: Image MIME type (e.g., "image/png")
                       - caption_text: Caption text for the page

        Returns:
            List of tuples (page_id, caption_id, image_chunk_id) for each page.
        """
        classes = self._get_schema_classes()
        Page = classes["Page"]
        Caption = classes["Caption"]
        ImageChunk = classes["ImageChunk"]

        results: list[tuple[int, int, int]] = []

        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError

            for page_num, image_content, mimetype, caption_text in pages_data:
                # Create page with image content stored directly
                page = Page(
                    document_id=document_id,
                    page_num=page_num,
                    image_content=image_content,
                    mimetype=mimetype,
                )
                uow.pages.add(page)
                uow.flush()

                # Create caption for the page
                caption = Caption(page_id=page.id, contents=caption_text)
                uow.captions.add(caption)
                uow.flush()

                # Create image chunk with image content stored directly
                image_chunk = ImageChunk(
                    content=image_content,
                    mimetype=mimetype,
                    parent_page=page.id,
                )
                uow.image_chunks.add(image_chunk)
                uow.flush()

                results.append((page.id, caption.id, image_chunk.id))

            uow.commit()

        return results

    def get_or_create_files(
        self,
        files: list[tuple[str, str]],
    ) -> list[tuple[int, bool]]:
        """Get existing or create new files.

        Args:
            files: List of tuples (path, file_type).

        Returns:
            List of tuples (file_id, was_created) where was_created is True if
            the file was created, False if it already existed.
        """
        classes = self._get_schema_classes()
        File = classes["File"]

        results: list[tuple[int, bool]] = []

        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError

            for path, file_type in files:
                existing = uow.files.get_by_path(path)
                if existing:
                    results.append((existing.id, False))
                else:
                    new_file = File(path=path, type=file_type)
                    uow.files.add(new_file)
                    uow.flush()
                    results.append((new_file.id, True))

            uow.commit()

        return results

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
            total_captions = uow.captions.count()
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
                "captions": total_captions,
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
