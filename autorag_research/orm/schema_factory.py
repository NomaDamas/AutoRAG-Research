"""Dynamic Schema Factory for AutoRAG-Research.

Provides a factory function to create ORM schema classes with configurable
embedding dimensions and primary key types. Supports multiple dimensions 
and key types in a single process.
"""

from functools import lru_cache
from typing import Literal

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
    def make_pk_column():
        """Create a primary key column based on primary_key_type."""
        return mapped_column(BigInteger if primary_key_type == "bigint" else String(255), primary_key=True, autoincrement=True)

    def make_fk_column(ref_table: str, ref_column: str = "id", nullable: bool = False, primary_key: bool = False, **kwargs):
        """Create a foreign key column based on primary_key_type."""
        return mapped_column(
            BigInteger if primary_key_type == "bigint" else String(255),
            ForeignKey(f"{ref_table}.{ref_column}", ondelete="CASCADE"),
            nullable=nullable,
            primary_key=primary_key,
            **kwargs
        )

    class File(Base):
        """File storage table for various file types"""

        __tablename__ = "file"

        id: Mapped[int | str] = make_pk_column()
        type: Mapped[str] = mapped_column(String(255), nullable=False)
        path: Mapped[str] = mapped_column(String(255), nullable=False)

        # Relationships
        documents: Mapped[list["Document"]] = relationship(foreign_keys="Document.path", back_populates="file")

    class Document(Base):
        """Document metadata table"""

        __tablename__ = "document"

        id: Mapped[int | str] = make_pk_column()
        path: Mapped[int | str | None] = make_fk_column("file", nullable=True, unique=True)
        filename: Mapped[str | None] = mapped_column(Text)
        author: Mapped[str | None] = mapped_column(Text)
        title: Mapped[str | None] = mapped_column(Text)
        doc_metadata: Mapped[dict | None] = mapped_column(JSONB)

        # Relationships
        file: Mapped["File | None"] = relationship(foreign_keys=[path], back_populates="documents")
        pages: Mapped[list["Page"]] = relationship(back_populates="document", cascade="all, delete-orphan")

    class Page(Base):
        """Page table for document pages"""

        __tablename__ = "page"

        id: Mapped[int | str] = make_pk_column()
        page_num: Mapped[int] = mapped_column(Integer, nullable=False)
        document_id: Mapped[int | str] = make_fk_column("document")
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

        id: Mapped[int | str] = make_pk_column()
        page_id: Mapped[int | str] = make_fk_column("page")
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


        id: Mapped[int | str] = make_pk_column()
        parent_caption: Mapped[int | str | None] = make_fk_column("caption", nullable=True)
        contents: Mapped[str] = mapped_column(Text, nullable=False)
        embedding: Mapped[Vector | None] = mapped_column(Vector(embedding_dim))
        embeddings: Mapped[list[list[float]] | None] = mapped_column(VectorArray(embedding_dim))


        # Relationships
        parent_caption_obj: Mapped["Caption | None"] = relationship(foreign_keys=[parent_caption], back_populates="chunks")
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


        id: Mapped[int | str] = make_pk_column()
        parent_page: Mapped[int | str | None] = make_fk_column("page", nullable=True)
        contents: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
        mimetype: Mapped[str] = mapped_column(String(255), nullable=False)
        embedding: Mapped[Vector | None] = mapped_column(Vector(embedding_dim))
        embeddings: Mapped[list[list[float]] | None] = mapped_column(VectorArray(embedding_dim))


        # Relationships
        page: Mapped["Page"] = relationship(back_populates="image_chunks")
        retrieval_relations: Mapped[list["RetrievalRelation"]] = relationship(
            back_populates="image_chunk", cascade="all, delete-orphan"
        )
        image_chunk_retrieved_results: Mapped[list["ImageChunkRetrievedResult"]] = relationship(
            back_populates="image_chunk", cascade="all, delete-orphan"
        )

    class CaptionChunkRelation(Base):
        """Relation between captions and chunks"""

        __tablename__ = "caption_chunk_relation"

        caption_id: Mapped[int | str] = make_fk_column("caption", primary_key=True)
        chunk_id: Mapped[int | str] = make_fk_column("chunk", primary_key=True)

        # Relationships
        caption: Mapped["Caption"] = relationship(back_populates="caption_chunk_relations")
        chunk: Mapped["Chunk"] = relationship(
            foreign_keys=[chunk_id], back_populates="caption_chunk_relations"
        )

    class Query(Base):
        """Query table for retrieval and generation evaluation"""

        __tablename__ = "query"


        id: Mapped[int | str] = make_pk_column()
        contents: Mapped[str] = mapped_column(Text, nullable=False)
        generation_gt: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
        embedding: Mapped[Vector | None] = mapped_column(Vector(embedding_dim))
        embeddings: Mapped[list[list[float]] | None] = mapped_column(VectorArray(embedding_dim))


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


        query_id: Mapped[int | str] = make_fk_column("query", nullable=False, primary_key=True)
        group_index: Mapped[int] = mapped_column(Integer, nullable=False, primary_key=True)
        group_order: Mapped[int] = mapped_column(Integer, nullable=False, primary_key=True)
        chunk_id: Mapped[int | str | None] = make_fk_column("chunk", nullable=True)
        image_chunk_id: Mapped[int | str | None] = make_fk_column("image_chunk", nullable=True)

        __table_args__ = (
            CheckConstraint(
                "(chunk_id IS NOT NULL AND image_chunk_id IS NULL) OR (chunk_id IS NULL AND image_chunk_id IS NOT NULL)",
                name="check_one_chunk_type",
            ),
        )

        # Relationships
        query_obj: Mapped["Query"] = relationship(back_populates="retrieval_relations")
        chunk: Mapped["Chunk | None"] = relationship(foreign_keys=[chunk_id], back_populates="retrieval_relations")
        image_chunk: Mapped["ImageChunk | None"] = relationship(back_populates="retrieval_relations")

    class Pipeline(Base):
        """Pipeline configuration table"""

        __tablename__ = "pipeline"


        id: Mapped[int | str] = make_pk_column()
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
        """Metric table for evaluation metrics"""

        __tablename__ = "metric"


        id: Mapped[int | str] = make_pk_column()
        name: Mapped[str] = mapped_column(String(255), nullable=False)
        type: Mapped[str] = mapped_column(String(255), nullable=False)


        # Relationships
        evaluation_results: Mapped[list["EvaluationResult"]] = relationship(
            back_populates="metric", cascade="all, delete-orphan"
        )
        summaries: Mapped[list["Summary"]] = relationship(back_populates="metric", cascade="all, delete-orphan")

    class ExecutorResult(Base):
        """Executor result table for pipeline execution results"""

        __tablename__ = "executor_result"


        query_id: Mapped[int | str] = make_fk_column("query", primary_key=True)
        pipeline_id: Mapped[int | str] = make_fk_column("pipeline", primary_key=True)
        generation_result: Mapped[str | None] = mapped_column(Text)
        token_usage: Mapped[int | None] = mapped_column(Integer)
        execution_time: Mapped[int | None] = mapped_column(Integer)
        result_metadata: Mapped[dict | None] = mapped_column(JSONB)


        # Relationships
        query_obj: Mapped["Query"] = relationship(back_populates="executor_results")
        pipeline: Mapped["Pipeline"] = relationship(back_populates="executor_results")

    class EvaluationResult(Base):
        """Evaluation result table for pipeline evaluation metrics"""

        __tablename__ = "evaluation_result"


        query_id: Mapped[int | str] = make_fk_column("query", primary_key=True)
        pipeline_id: Mapped[int | str] = make_fk_column("pipeline", primary_key=True)
        metric_id: Mapped[int | str] = make_fk_column("metric", primary_key=True)
        metric_result: Mapped[float] = mapped_column(Float, nullable=False)


        # Relationships
        query_obj: Mapped["Query"] = relationship(back_populates="evaluation_results")
        pipeline: Mapped["Pipeline"] = relationship(back_populates="evaluation_results")
        metric: Mapped["Metric"] = relationship(back_populates="evaluation_results")

    class ImageChunkRetrievedResult(Base):
        """Image chunk retrieval result table"""

        __tablename__ = "image_chunk_retrieved_result"


        query_id: Mapped[int | str] = make_fk_column("query", primary_key=True)
        pipeline_id: Mapped[int | str] = make_fk_column("pipeline", primary_key=True)
        image_chunk_id: Mapped[int | str] = make_fk_column("image_chunk", primary_key=True)
        rel_score: Mapped[float | None] = mapped_column(Float)


        # Relationships
        query_obj: Mapped["Query"] = relationship(back_populates="image_chunk_retrieved_results")
        pipeline: Mapped["Pipeline"] = relationship(back_populates="image_chunk_retrieved_results")
        image_chunk: Mapped["ImageChunk"] = relationship(back_populates="image_chunk_retrieved_results")

    class ChunkRetrievedResult(Base):
        """Text chunk retrieval result table"""

        __tablename__ = "chunk_retrieved_result"


        query_id: Mapped[int | str] = make_fk_column("query", primary_key=True)
        pipeline_id: Mapped[int | str] = make_fk_column("pipeline", primary_key=True)
        chunk_id: Mapped[int | str] = make_fk_column("chunk", primary_key=True)
        rel_score: Mapped[float | None] = mapped_column(Float)


        # Relationships
        query_obj: Mapped["Query"] = relationship(back_populates="chunk_retrieved_results")
        pipeline: Mapped["Pipeline"] = relationship(back_populates="chunk_retrieved_results")
        chunk: Mapped["Chunk"] = relationship(back_populates="chunk_retrieved_results")

    class Summary(Base):
        """Summary table for aggregated pipeline metrics"""

        __tablename__ = "summary"

        pipeline_id: Mapped[int | str] = make_fk_column("pipeline", primary_key=True)
        metric_id: Mapped[int | str] = make_fk_column("metric", primary_key=True)
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
    Schema.primary_key_type = primary_key_type  # ty: ignore

    return Schema
