"""Dynamic Schema Factory for AutoRAG-Research.

Provides a factory function to create ORM schema classes with configurable
embedding dimensions and primary key types. Supports multiple dimensions 
and key types in a single process.
"""

from functools import lru_cache
from typing import Optional, Literal, Union

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from autorag_research.orm.types import VectorArray


@lru_cache(maxsize=16)
def create_schema(
    embedding_dim: int = 768,
    primary_key_type: Literal["bigint", "string"] = "bigint"
):
    """Create ORM schema classes with specified embedding dimension and primary key type.

    This factory function generates a complete set of ORM classes with
    the given embedding dimension for Vector columns and configurable
    primary key types. Results are cached for reuse within the same process.

    Args:
        embedding_dim: The dimension for embedding vectors (default: 768).
        primary_key_type: Type of primary keys - "bigint" (auto-increment) or "string" (user-provided)

    Returns:
        A Schema namespace object containing all ORM classes:
        - Base: The declarative base class
        - File, Document, Page, Caption, Chunk, ImageChunk, etc.
        - embedding_dim: The dimension used for this schema
        - primary_key_type: The primary key type used for this schema

    Example:
        >>> # BigInt primary keys (default)
        >>> schema = create_schema(1024)
        >>> 
        >>> # String primary keys (user-provided IDs)
        >>> schema = create_schema(768, primary_key_type="string")
        >>> chunk = schema.Chunk(id="my-chunk-001", contents="...")
    """

    class Base(DeclarativeBase):
        pass

    # Helper functions for primary and foreign keys
    if primary_key_type == "string":
        def make_pk_column():
            """Create a string primary key column (user must provide unique values)"""
            return mapped_column(String(255), primary_key=True, nullable=False)
        
        def make_fk_column(ref_table: str, ref_column: str = "id", nullable: bool = False, primary_key: bool = False, **kwargs):
            """Create a string foreign key column"""
            return mapped_column(
                String(255),
                ForeignKey(f"{ref_table}.{ref_column}", ondelete="CASCADE"),
                nullable=nullable,
                primary_key=primary_key,
                **kwargs
            )
    else:
        def make_pk_column():
            """Create a bigint primary key column with auto-increment"""
            return mapped_column(BigInteger, primary_key=True, autoincrement=True)
        
        def make_fk_column(ref_table: str, ref_column: str = "id", nullable: bool = False, primary_key: bool = False, **kwargs):
            """Create a bigint foreign key column"""
            return mapped_column(
                BigInteger,
                ForeignKey(f"{ref_table}.{ref_column}", ondelete="CASCADE"),
                nullable=nullable,
                primary_key=primary_key,
                **kwargs
            )

    class File(Base):
        """File storage table for various file types"""

        __tablename__ = "file"

        id: Mapped[Union[int, str]] = make_pk_column()
        type: Mapped[str] = mapped_column(String(255), nullable=False)
        path: Mapped[str] = mapped_column(String(255), nullable=False)

        # Relationships
        documents: Mapped[list["Document"]] = relationship(foreign_keys="Document.path", back_populates="file")

    class Document(Base):
        """Document metadata table"""

        __tablename__ = "document"

        id: Mapped[Union[int, str]] = make_pk_column()
        path: Mapped[Union[int, str, None]] = make_fk_column("file", nullable=True, unique=True)
        filename: Mapped[str | None] = mapped_column(Text)
        author: Mapped[str | None] = mapped_column(Text)
        title: Mapped[str | None] = mapped_column(Text)
        doc_metadata: Mapped[dict | None] = mapped_column(JSONB)

        # Relationships
        file: Mapped[Optional["File"]] = relationship(foreign_keys=[path], back_populates="documents")
        pages: Mapped[list["Page"]] = relationship(back_populates="document", cascade="all, delete-orphan")

    class Page(Base):
        """Page table for document pages"""

        __tablename__ = "page"

        id: Mapped[Union[int, str]] = make_pk_column()
        page_num: Mapped[int] = mapped_column(Integer, nullable=False)
        document_id: Mapped[Union[int, str]] = make_fk_column("document")
        image_contents: Mapped[bytes | None] = mapped_column(LargeBinary)
        mimetype: Mapped[str | None] = mapped_column(String(255))
        page_metadata: Mapped[dict | None] = mapped_column(JSONB)

        __table_args__ = (UniqueConstraint("document_id", "page_num", name="uq_document_page"),)

        # Relationships
        document: Mapped["Document"] = relationship(back_populates="pages")
        captions: Mapped[list["Caption"]] = relationship(back_populates="page", cascade="all, delete-orphan")
        image_chunks: Mapped[list["ImageChunk"]] = relationship(back_populates="page", cascade="all, delete-orphan")

    class Caption(Base):
        """Caption table for page captions"""

        __tablename__ = "caption"

        id: Mapped[Union[int, str]] = make_pk_column()
        page_id: Mapped[Union[int, str]] = make_fk_column("page")
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

        id: Mapped[Union[int, str]] = make_pk_column()
        parent_caption: Mapped[Union[int, str, None]] = make_fk_column("caption", nullable=True)
        contents: Mapped[str] = mapped_column(Text, nullable=False)
        embedding: Mapped[Vector | None] = mapped_column(Vector(embedding_dim))
        chunk_metadata: Mapped[dict | None] = mapped_column(JSONB)

        # Relationships
        parent_caption_obj: Mapped[Optional["Caption"]] = relationship(foreign_keys=[parent_caption], back_populates="chunks")
        caption_chunk_relations: Mapped[list["CaptionChunkRelation"]] = relationship(
            foreign_keys="CaptionChunkRelation.chunk_id", back_populates="chunk", cascade="all, delete-orphan"
        )
        retrieval_relations: Mapped[list["RetrievalRelation"]] = relationship(
            foreign_keys="RetrievalRelation.chunk_id", back_populates="chunk", cascade="all, delete-orphan"
        )
        chunk_retrieved_results: Mapped[list["ChunkRetrievedResult"]] = relationship(
            back_populates="chunk", cascade="all, delete-orphan"
        )

    class ImageChunk(Base):
        """Image chunk table with image embeddings"""

        __tablename__ = "image_chunk"

        id: Mapped[Union[int, str]] = make_pk_column()
        page_id: Mapped[Union[int, str]] = make_fk_column("page")
        image_embedding: Mapped[VectorArray | None] = mapped_column(VectorArray(embedding_dim))
        image_chunk_metadata: Mapped[dict | None] = mapped_column(JSONB)

        # Relationships
        page: Mapped["Page"] = relationship(back_populates="image_chunks")
        caption_chunk_relations: Mapped[list["CaptionChunkRelation"]] = relationship(
            back_populates="image_chunk", cascade="all, delete-orphan"
        )
        image_chunk_retrieved_results: Mapped[list["ImageChunkRetrievedResult"]] = relationship(
            back_populates="image_chunk", cascade="all, delete-orphan"
        )

    class CaptionChunkRelation(Base):
        """Relation between captions, chunks and image chunks"""

        __tablename__ = "caption_chunk_relation"

        caption_id: Mapped[Union[int, str]] = make_fk_column("caption", primary_key=True)
        chunk_id: Mapped[Union[int, str, None]] = make_fk_column("chunk", nullable=True)
        image_chunk_id: Mapped[Union[int, str, None]] = make_fk_column("image_chunk", nullable=True)

        __table_args__ = (
            CheckConstraint(
                "(chunk_id IS NULL) != (image_chunk_id IS NULL)",
                name="check_chunk_or_image_chunk",
            ),
        )

        # Relationships
        caption: Mapped["Caption"] = relationship(back_populates="caption_chunk_relations")
        chunk: Mapped[Optional["Chunk"]] = relationship(
            foreign_keys=[chunk_id], back_populates="caption_chunk_relations"
        )
        image_chunk: Mapped[Optional["ImageChunk"]] = relationship(
            foreign_keys=[image_chunk_id], back_populates="caption_chunk_relations"
        )

    class Query(Base):
        """Query table for retrieval queries"""

        __tablename__ = "query"

        id: Mapped[Union[int, str]] = make_pk_column()
        contents: Mapped[str] = mapped_column(Text, nullable=False)
        embedding: Mapped[Vector | None] = mapped_column(Vector(embedding_dim))
        query_metadata: Mapped[dict | None] = mapped_column(JSONB)

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
        """Relation between queries and chunks with scores"""

        __tablename__ = "retrieval_relation"

        query_id: Mapped[Union[int, str]] = make_fk_column("query", primary_key=True)
        chunk_id: Mapped[Union[int, str]] = make_fk_column("chunk", primary_key=True)
        chunk_scores: Mapped[list[float] | None] = mapped_column(ARRAY(Float))
        retrieval_metadata: Mapped[dict | None] = mapped_column(JSONB)

        # Relationships
        query_obj: Mapped["Query"] = relationship(back_populates="retrieval_relations")
        chunk: Mapped["Chunk"] = relationship(foreign_keys=[chunk_id], back_populates="retrieval_relations")

    class Pipeline(Base):
        """Pipeline table for processing pipelines"""

        __tablename__ = "pipeline"

        id: Mapped[Union[int, str]] = make_pk_column()
        name: Mapped[str | None] = mapped_column(Text)
        config: Mapped[dict | None] = mapped_column(JSONB)
        trial_dir: Mapped[str | None] = mapped_column(Text)
        pipeline_metadata: Mapped[dict | None] = mapped_column(JSONB)

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
        """Metric table for evaluation metrics"""

        __tablename__ = "metric"

        id: Mapped[Union[int, str]] = make_pk_column()
        name: Mapped[str] = mapped_column(String(255), nullable=False)
        metric_type: Mapped[str] = mapped_column(String(255), nullable=False)
        metric_metadata: Mapped[dict | None] = mapped_column(JSONB)

        # Relationships
        executor_results: Mapped[list["ExecutorResult"]] = relationship(
            back_populates="metric", cascade="all, delete-orphan"
        )
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
        """Executor result table for pipeline execution results"""

        __tablename__ = "executor_result"

        query_id: Mapped[Union[int, str]] = make_fk_column("query", primary_key=True)
        pipeline_id: Mapped[Union[int, str]] = make_fk_column("pipeline", primary_key=True)
        metric_id: Mapped[Union[int, str]] = make_fk_column("metric", primary_key=True)
        generation: Mapped[str | None] = mapped_column(Text)
        latency: Mapped[float | None] = mapped_column(Float)
        token_usage: Mapped[int | None] = mapped_column(Integer)
        result_metadata: Mapped[dict | None] = mapped_column(JSONB)

        # Relationships
        query_obj: Mapped["Query"] = relationship(back_populates="executor_results")
        pipeline: Mapped["Pipeline"] = relationship(back_populates="executor_results")
        metric: Mapped["Metric"] = relationship(back_populates="executor_results")

    class EvaluationResult(Base):
        """Evaluation result table for pipeline evaluation metrics"""

        __tablename__ = "evaluation_result"

        query_id: Mapped[Union[int, str]] = make_fk_column("query", primary_key=True)
        pipeline_id: Mapped[Union[int, str]] = make_fk_column("pipeline", primary_key=True)
        metric_id: Mapped[Union[int, str]] = make_fk_column("metric", primary_key=True)
        metric_result: Mapped[float | None] = mapped_column(Float)

        # Relationships
        query_obj: Mapped["Query"] = relationship(back_populates="evaluation_results")
        pipeline: Mapped["Pipeline"] = relationship(back_populates="evaluation_results")
        metric: Mapped["Metric"] = relationship(back_populates="evaluation_results")

    class ImageChunkRetrievedResult(Base):
        """Image chunk retrieval result table"""

        __tablename__ = "image_chunk_retrieved_result"

        query_id: Mapped[Union[int, str]] = make_fk_column("query", primary_key=True)
        pipeline_id: Mapped[Union[int, str]] = make_fk_column("pipeline", primary_key=True)
        metric_id: Mapped[Union[int, str]] = make_fk_column("metric", primary_key=True)
        image_chunk_id: Mapped[Union[int, str]] = make_fk_column("image_chunk", primary_key=True)
        rel_score: Mapped[float | None] = mapped_column(Float)

        # Relationships
        query_obj: Mapped["Query"] = relationship(back_populates="image_chunk_retrieved_results")
        pipeline: Mapped["Pipeline"] = relationship(back_populates="image_chunk_retrieved_results")
        metric: Mapped["Metric"] = relationship(back_populates="image_chunk_retrieved_results")
        image_chunk: Mapped["ImageChunk"] = relationship(back_populates="image_chunk_retrieved_results")

    class ChunkRetrievedResult(Base):
        """Text chunk retrieval result table"""

        __tablename__ = "chunk_retrieved_result"

        query_id: Mapped[Union[int, str]] = make_fk_column("query", primary_key=True)
        pipeline_id: Mapped[Union[int, str]] = make_fk_column("pipeline", primary_key=True)
        metric_id: Mapped[Union[int, str]] = make_fk_column("metric", primary_key=True)
        chunk_id: Mapped[Union[int, str]] = make_fk_column("chunk", primary_key=True)
        rel_score: Mapped[float | None] = mapped_column(Float)

        # Relationships
        query_obj: Mapped["Query"] = relationship(back_populates="chunk_retrieved_results")
        pipeline: Mapped["Pipeline"] = relationship(back_populates="chunk_retrieved_results")
        metric: Mapped["Metric"] = relationship(back_populates="chunk_retrieved_results")
        chunk: Mapped["Chunk"] = relationship(back_populates="chunk_retrieved_results")

    class Summary(Base):
        """Summary table for aggregated pipeline metrics"""

        __tablename__ = "summary"

        pipeline_id: Mapped[Union[int, str]] = make_fk_column("pipeline", primary_key=True)
        metric_id: Mapped[Union[int, str]] = make_fk_column("metric", primary_key=True)
        metric_result: Mapped[float] = mapped_column(Float, nullable=False)
        token_usage: Mapped[int | None] = mapped_column(Integer)
        execution_time: Mapped[int | None] = mapped_column(Integer)
        result_metadata: Mapped[dict | None] = mapped_column(JSONB)

        # Relationships
        pipeline: Mapped["Pipeline"] = relationship(back_populates="summaries")
        metric: Mapped["Metric"] = relationship(back_populates="summaries")

    class Schema:
        """Namespace containing all ORM classes for a specific embedding dimension and PK type."""
        pass

    # Attach all classes to Schema namespace
    Schema.Base = Base  # type: ignore
    Schema.File = File  # type: ignore
    Schema.Document = Document  # type: ignore
    Schema.Page = Page  # type: ignore
    Schema.Caption = Caption  # type: ignore
    Schema.Chunk = Chunk  # type: ignore
    Schema.ImageChunk = ImageChunk  # type: ignore
    Schema.CaptionChunkRelation = CaptionChunkRelation  # type: ignore
    Schema.Query = Query  # type: ignore
    Schema.RetrievalRelation = RetrievalRelation  # type: ignore
    Schema.Pipeline = Pipeline  # type: ignore
    Schema.Metric = Metric  # type: ignore
    Schema.ExecutorResult = ExecutorResult  # type: ignore
    Schema.EvaluationResult = EvaluationResult  # type: ignore
    Schema.ImageChunkRetrievedResult = ImageChunkRetrievedResult  # type: ignore
    Schema.ChunkRetrievedResult = ChunkRetrievedResult  # type: ignore
    Schema.Summary = Summary  # type: ignore
    Schema.embedding_dim = embedding_dim  # type: ignore
    Schema.primary_key_type = primary_key_type  # type: ignore

    return Schema
