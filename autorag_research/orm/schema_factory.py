"""Dynamic Schema Factory for AutoRAG-Research.

Provides a factory function to create ORM schema classes with configurable
embedding dimensions. Supports multiple dimensions in a single process.
"""

from functools import lru_cache
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


@lru_cache(maxsize=16)
def create_schema(embedding_dim: int = 768):
    """Create ORM schema classes with specified embedding dimension.

    This factory function generates a complete set of ORM classes with
    the given embedding dimension for Vector columns. Results are cached
    by dimension for reuse within the same process.

    Args:
        embedding_dim: The dimension for embedding vectors (default: 768).

    Returns:
        A Schema namespace object containing all ORM classes:
        - Base: The declarative base class
        - File, Document, Page, Caption, Chunk, ImageChunk, etc.
        - embedding_dim: The dimension used for this schema

    Example:
        >>> schema = create_schema(1024)
        >>> engine = create_engine("postgresql://...")
        >>> schema.Base.metadata.create_all(engine)
        >>> chunk = schema.Chunk(contents="...", embedding=[...])
    """

    class Base(DeclarativeBase):
        pass

    class File(Base):
        """File storage table for various file types"""

        __tablename__ = "file"

        id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
        type: Mapped[str] = mapped_column(String(255), nullable=False)
        path: Mapped[str] = mapped_column(String(255), nullable=False)

        # Relationships
        documents: Mapped[list["Document"]] = relationship(foreign_keys="Document.filepath", back_populates="file")
        pages: Mapped[list["Page"]] = relationship(foreign_keys="Page.image_path", back_populates="image_file")
        image_chunks: Mapped[list["ImageChunk"]] = relationship(
            foreign_keys="ImageChunk.image_path", back_populates="image_file"
        )

    class Document(Base):
        """Document metadata table"""

        __tablename__ = "document"

        id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
        filepath: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("file.id", ondelete="CASCADE"), unique=True)
        filename: Mapped[str | None] = mapped_column(Text)
        author: Mapped[str | None] = mapped_column(Text)
        title: Mapped[str | None] = mapped_column(Text)
        doc_metadata: Mapped[dict | None] = mapped_column(JSONB)

        # Relationships
        file: Mapped[Optional["File"]] = relationship(foreign_keys=[filepath], back_populates="documents")
        pages: Mapped[list["Page"]] = relationship(back_populates="document", cascade="all, delete-orphan")

    class Page(Base):
        """Page table for document pages"""

        __tablename__ = "page"

        id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
        page_num: Mapped[int] = mapped_column(Integer, nullable=False)
        document_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("document.id", ondelete="CASCADE"), nullable=False
        )
        image_path: Mapped[int | None] = mapped_column(
            BigInteger, ForeignKey("file.id", ondelete="SET NULL"), unique=True
        )
        page_metadata: Mapped[dict | None] = mapped_column(JSONB)

        __table_args__ = (UniqueConstraint("document_id", "page_num", name="uq_document_page"),)

        # Relationships
        document: Mapped["Document"] = relationship(back_populates="pages")
        image_file: Mapped[Optional["File"]] = relationship(foreign_keys=[image_path], back_populates="pages")
        captions: Mapped[list["Caption"]] = relationship(back_populates="page", cascade="all, delete-orphan")
        image_chunks: Mapped[list["ImageChunk"]] = relationship(back_populates="page", cascade="all, delete-orphan")

    class Caption(Base):
        """Caption table for page captions"""

        __tablename__ = "caption"

        id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
        page_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("page.id", ondelete="CASCADE"), nullable=False)
        contents: Mapped[str] = mapped_column(Text, nullable=False)

        # Relationships
        page: Mapped["Page"] = relationship(back_populates="captions")
        chunks: Mapped[list["Chunk"]] = relationship(back_populates="parent_caption_obj", cascade="all, delete-orphan")
        caption_chunk_relations: Mapped[list["CaptionChunkRelation"]] = relationship(
            back_populates="caption", cascade="all, delete-orphan"
        )

    class Chunk(Base):
        """Text chunk table with embeddings"""

        __tablename__ = "chunk"

        id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
        parent_caption: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("caption.id", ondelete="CASCADE"))
        contents: Mapped[str] = mapped_column(Text, nullable=False)
        embedding: Mapped[Vector | None] = mapped_column(Vector(embedding_dim))
        embeddings: Mapped[list[Vector] | None] = mapped_column(ARRAY(Vector(embedding_dim)))

        # Relationships
        parent_caption_obj: Mapped[Optional["Caption"]] = relationship(back_populates="chunks")
        caption_chunk_relations: Mapped[list["CaptionChunkRelation"]] = relationship(
            back_populates="chunk", cascade="all, delete-orphan"
        )
        retrieval_relations: Mapped[list["RetrievalRelation"]] = relationship(
            back_populates="chunk", cascade="all, delete-orphan"
        )
        chunk_retrieved_results: Mapped[list["ChunkRetrievedResult"]] = relationship(
            back_populates="chunk", cascade="all, delete-orphan"
        )

    class ImageChunk(Base):
        """Image chunk table with embeddings"""

        __tablename__ = "image_chunk"

        id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
        parent_page: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("page.id", ondelete="CASCADE"))
        image_path: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("file.id", ondelete="CASCADE"), nullable=False, unique=True
        )
        embedding: Mapped[Vector | None] = mapped_column(Vector(embedding_dim))
        embeddings: Mapped[list[Vector] | None] = mapped_column(ARRAY(Vector(embedding_dim)))

        # Relationships
        page: Mapped[Optional["Page"]] = relationship(back_populates="image_chunks")
        image_file: Mapped["File"] = relationship(foreign_keys=[image_path], back_populates="image_chunks")
        retrieval_relations: Mapped[list["RetrievalRelation"]] = relationship(
            back_populates="image_chunk", cascade="all, delete-orphan"
        )
        image_chunk_retrieved_results: Mapped[list["ImageChunkRetrievedResult"]] = relationship(
            back_populates="image_chunk", cascade="all, delete-orphan"
        )

    class CaptionChunkRelation(Base):
        """Many-to-many relation between Caption and Chunk"""

        __tablename__ = "caption_chunk_relation"

        caption_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("caption.id", ondelete="CASCADE"), primary_key=True
        )
        chunk_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("chunk.id", ondelete="CASCADE"), primary_key=True)

        # Relationships
        caption: Mapped["Caption"] = relationship(back_populates="caption_chunk_relations")
        chunk: Mapped["Chunk"] = relationship(back_populates="caption_chunk_relations")

    class Query(Base):
        """Query table for retrieval and generation evaluation"""

        __tablename__ = "query"

        id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
        query: Mapped[str] = mapped_column(Text, nullable=False)
        generation_gt: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=True)
        embedding: Mapped[Vector | None] = mapped_column(Vector(embedding_dim))
        embeddings: Mapped[list[Vector] | None] = mapped_column(ARRAY(Vector(embedding_dim)))

        # Relationships
        retrieval_relations: Mapped[list["RetrievalRelation"]] = relationship(
            back_populates="query_obj", cascade="all, delete-orphan"
        )
        executor_results: Mapped[list["ExecutorResult"]] = relationship(
            back_populates="query_obj", cascade="all, delete-orphan"
        )
        evaluation_results: Mapped[list["EvaluationResult"]] = relationship(
            back_populates="query_obj", cascade="all, delete-orphan"
        )
        image_chunk_retrieved_results: Mapped[list["ImageChunkRetrievedResult"]] = relationship(
            back_populates="query_obj", cascade="all, delete-orphan"
        )
        chunk_retrieved_results: Mapped[list["ChunkRetrievedResult"]] = relationship(
            back_populates="query_obj", cascade="all, delete-orphan"
        )

    class RetrievalRelation(Base):
        """Retrieval ground truth relation table"""

        __tablename__ = "retrieval_relation"

        query_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("query.id", ondelete="CASCADE"), nullable=False, primary_key=True
        )
        group_index: Mapped[int] = mapped_column(Integer, nullable=False, primary_key=True)
        group_order: Mapped[int] = mapped_column(Integer, nullable=False, primary_key=True)
        chunk_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("chunk.id", ondelete="CASCADE"))
        image_chunk_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("image_chunk.id", ondelete="CASCADE"))

        __table_args__ = (
            CheckConstraint(
                "(chunk_id IS NOT NULL AND image_chunk_id IS NULL) OR (chunk_id IS NULL AND image_chunk_id IS NOT NULL)",
                name="check_one_chunk_type",
            ),
        )

        # Relationships
        query_obj: Mapped["Query"] = relationship(back_populates="retrieval_relations")
        chunk: Mapped[Optional["Chunk"]] = relationship(back_populates="retrieval_relations")
        image_chunk: Mapped[Optional["ImageChunk"]] = relationship(back_populates="retrieval_relations")

    class Pipeline(Base):
        """Pipeline configuration table"""

        __tablename__ = "pipeline"

        id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
        name: Mapped[str] = mapped_column(String(255), nullable=False)
        config: Mapped[dict] = mapped_column(JSONB, nullable=False)

        # Relationships
        executor_results: Mapped[list["ExecutorResult"]] = relationship(
            back_populates="pipeline", cascade="all, delete-orphan"
        )
        evaluation_results: Mapped[list["EvaluationResult"]] = relationship(
            back_populates="pipeline", cascade="all, delete-orphan"
        )
        image_chunk_retrieved_results: Mapped[list["ImageChunkRetrievedResult"]] = relationship(
            back_populates="pipeline", cascade="all, delete-orphan"
        )
        chunk_retrieved_results: Mapped[list["ChunkRetrievedResult"]] = relationship(
            back_populates="pipeline", cascade="all, delete-orphan"
        )
        summaries: Mapped[list["Summary"]] = relationship(back_populates="pipeline", cascade="all, delete-orphan")

    class Metric(Base):
        """Metric definition table"""

        __tablename__ = "metric"

        id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
        name: Mapped[str] = mapped_column(String(255), nullable=False)
        type: Mapped[str] = mapped_column(String(255), nullable=False)

        # Relationships
        evaluation_results: Mapped[list["EvaluationResult"]] = relationship(
            back_populates="metric", cascade="all, delete-orphan"
        )
        image_chunk_retrieved_results: Mapped[list["ImageChunkRetrievedResult"]] = relationship(
            back_populates="metric", cascade="all, delete-orphan"
        )
        chunk_retrieved_results: Mapped[list["ChunkRetrievedResult"]] = relationship(
            back_populates="metric", cascade="all, delete-orphan"
        )
        summaries: Mapped[list["Summary"]] = relationship(back_populates="metric", cascade="all, delete-orphan")

    class ExecutorResult(Base):
        """Executor result table for query execution details"""

        __tablename__ = "executor_result"

        query_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("query.id", ondelete="CASCADE"), nullable=False, primary_key=True
        )
        pipeline_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("pipeline.id", ondelete="CASCADE"), nullable=False, primary_key=True
        )
        generation_result: Mapped[str | None] = mapped_column(Text)
        token_usage: Mapped[int | None] = mapped_column(Integer)
        execution_time: Mapped[int | None] = mapped_column(Integer)
        result_metadata: Mapped[dict | None] = mapped_column(JSONB)

        # Relationships
        query_obj: Mapped["Query"] = relationship(back_populates="executor_results")
        pipeline: Mapped["Pipeline"] = relationship(back_populates="executor_results")

    class EvaluationResult(Base):
        """Evaluation result table for query-level metrics"""

        __tablename__ = "evaluation_result"

        query_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("query.id", ondelete="CASCADE"), nullable=False, primary_key=True
        )
        pipeline_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("pipeline.id", ondelete="CASCADE"), nullable=False, primary_key=True
        )
        metric_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("metric.id", ondelete="CASCADE"), nullable=False, primary_key=True
        )
        metric_result: Mapped[float | None] = mapped_column(Float)

        # Relationships
        query_obj: Mapped["Query"] = relationship(back_populates="evaluation_results")
        pipeline: Mapped["Pipeline"] = relationship(back_populates="evaluation_results")
        metric: Mapped["Metric"] = relationship(back_populates="evaluation_results")

    class ImageChunkRetrievedResult(Base):
        """Image chunk retrieval result table"""

        __tablename__ = "image_chunk_retrieved_result"

        query_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("query.id", ondelete="CASCADE"), nullable=False, primary_key=True
        )
        pipeline_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("pipeline.id", ondelete="CASCADE"), nullable=False, primary_key=True
        )
        metric_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("metric.id", ondelete="CASCADE"), nullable=False, primary_key=True
        )
        image_chunk_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("image_chunk.id", ondelete="CASCADE"), nullable=False, primary_key=True
        )
        rel_score: Mapped[float | None] = mapped_column(Float)

        # Relationships
        query_obj: Mapped["Query"] = relationship(back_populates="image_chunk_retrieved_results")
        pipeline: Mapped["Pipeline"] = relationship(back_populates="image_chunk_retrieved_results")
        metric: Mapped["Metric"] = relationship(back_populates="image_chunk_retrieved_results")
        image_chunk: Mapped["ImageChunk"] = relationship(back_populates="image_chunk_retrieved_results")

    class ChunkRetrievedResult(Base):
        """Text chunk retrieval result table"""

        __tablename__ = "chunk_retrieved_result"

        query_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("query.id", ondelete="CASCADE"), nullable=False, primary_key=True
        )
        pipeline_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("pipeline.id", ondelete="CASCADE"), nullable=False, primary_key=True
        )
        metric_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("metric.id", ondelete="CASCADE"), nullable=False, primary_key=True
        )
        chunk_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("chunk.id", ondelete="CASCADE"), nullable=False, primary_key=True
        )
        rel_score: Mapped[float | None] = mapped_column(Float)

        # Relationships
        query_obj: Mapped["Query"] = relationship(back_populates="chunk_retrieved_results")
        pipeline: Mapped["Pipeline"] = relationship(back_populates="chunk_retrieved_results")
        metric: Mapped["Metric"] = relationship(back_populates="chunk_retrieved_results")
        chunk: Mapped["Chunk"] = relationship(back_populates="chunk_retrieved_results")

    class Summary(Base):
        """Summary table for aggregated pipeline metrics"""

        __tablename__ = "summary"

        pipeline_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("pipeline.id", ondelete="CASCADE"), nullable=False, primary_key=True
        )
        metric_id: Mapped[int] = mapped_column(
            BigInteger, ForeignKey("metric.id", ondelete="CASCADE"), nullable=False, primary_key=True
        )
        metric_result: Mapped[float] = mapped_column(Float, nullable=False)
        token_usage: Mapped[int | None] = mapped_column(Integer)
        execution_time: Mapped[int | None] = mapped_column(Integer)
        result_metadata: Mapped[dict | None] = mapped_column(JSONB)

        # Relationships
        pipeline: Mapped["Pipeline"] = relationship(back_populates="summaries")
        metric: Mapped["Metric"] = relationship(back_populates="summaries")

    class Schema:
        """Namespace containing all ORM classes for a specific embedding dimension."""

        pass

    # Attach all classes to Schema namespace
    Schema.Base = Base  # ty: ignore
    Schema.File = File  # ty: ignore
    Schema.Document = Document  # ty: ignore
    Schema.Page = Page  # ty: ignore
    Schema.Caption = Caption  # ty: ignore
    Schema.Chunk = Chunk  # ty: ignore
    Schema.ImageChunk = ImageChunk  # ty: ignore
    Schema.CaptionChunkRelation = CaptionChunkRelation  # ty: ignore
    Schema.Query = Query  # ty: ignore
    Schema.RetrievalRelation = RetrievalRelation  # ty: ignore
    Schema.Pipeline = Pipeline  # ty: ignore
    Schema.Metric = Metric  # ty: ignore
    Schema.ExecutorResult = ExecutorResult  # ty: ignore
    Schema.EvaluationResult = EvaluationResult  # ty: ignore
    Schema.ImageChunkRetrievedResult = ImageChunkRetrievedResult  # ty: ignore
    Schema.ChunkRetrievedResult = ChunkRetrievedResult  # ty: ignore
    Schema.Summary = Summary  # ty: ignore
    Schema.embedding_dim = embedding_dim  # ty: ignore

    return Schema
