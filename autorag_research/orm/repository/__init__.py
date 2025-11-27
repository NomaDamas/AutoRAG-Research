"""Repository module for AutoRAG-Research ORM.

This module provides repository classes for data access layer operations.
"""

from autorag_research.orm.repository.base import (
    BaseVectorRepository,
    GenericRepository,
    UnitOfWork,
)
from autorag_research.orm.repository.caption import CaptionRepository
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.document import DocumentRepository
from autorag_research.orm.repository.executor_result import ExecutorResultRepository
from autorag_research.orm.repository.file import FileRepository
from autorag_research.orm.repository.image_chunk import ImageChunkRepository
from autorag_research.orm.repository.metric import MetricRepository
from autorag_research.orm.repository.page import PageRepository
from autorag_research.orm.repository.pipeline import PipelineRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.repository.summary import SummaryRepository
from autorag_research.orm.repository.text_uow import TextOnlyUnitOfWork

__all__ = [
    "BaseVectorRepository",
    "CaptionRepository",
    "ChunkRepository",
    "DocumentRepository",
    "ExecutorResultRepository",
    "FileRepository",
    "GenericRepository",
    "ImageChunkRepository",
    "MetricRepository",
    "PageRepository",
    "PipelineRepository",
    "QueryRepository",
    "SummaryRepository",
    "TextOnlyUnitOfWork",
    "UnitOfWork",
]
