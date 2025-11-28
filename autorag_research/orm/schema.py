"""ORM Schema definitions for AutoRAG-Research.

This module provides ORM classes with a default embedding dimension of 768.
For custom embedding dimensions, use `create_schema(dim)` from schema_factory.

Example:
    # Using default 768 dimension
    from autorag_research.orm.schema import Base, Chunk, Query

    # Using custom dimension
    from autorag_research.orm.schema import create_schema
    schema = create_schema(1024)
    # Use schema.Base, schema.Chunk, schema.Query, etc.
"""

from autorag_research.orm.schema_factory import create_schema

# Create default schema with 768 embedding dimension
_default_schema = create_schema(768)

# Re-export all classes from default schema for backward compatibility
Base = _default_schema.Base
File = _default_schema.File
Document = _default_schema.Document
Page = _default_schema.Page
Caption = _default_schema.Caption
Chunk = _default_schema.Chunk
ImageChunk = _default_schema.ImageChunk
CaptionChunkRelation = _default_schema.CaptionChunkRelation
Query = _default_schema.Query
RetrievalRelation = _default_schema.RetrievalRelation
Pipeline = _default_schema.Pipeline
Metric = _default_schema.Metric
ExecutorResult = _default_schema.ExecutorResult
EvaluationResult = _default_schema.EvaluationResult
ImageChunkRetrievedResult = _default_schema.ImageChunkRetrievedResult
ChunkRetrievedResult = _default_schema.ChunkRetrievedResult
Summary = _default_schema.Summary

__all__ = [
    "Base",
    "Caption",
    "CaptionChunkRelation",
    "Chunk",
    "ChunkRetrievedResult",
    "Document",
    "EvaluationResult",
    "ExecutorResult",
    "File",
    "ImageChunk",
    "ImageChunkRetrievedResult",
    "Metric",
    "Page",
    "Pipeline",
    "Query",
    "RetrievalRelation",
    "Summary",
    "create_schema",
]
