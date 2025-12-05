"""Unit of Work (UoW) pattern implementations for AutoRAG-Research.

Provides transaction management and repository coordination:
- BaseUnitOfWork: Abstract base class with common patterns
- TextOnlyUnitOfWork: For text-only data ingestion
- MultiModalUnitOfWork: For multi-modal data ingestion
- RetrievalUnitOfWork: For retrieval pipeline execution
"""

from autorag_research.orm.uow.base import BaseUnitOfWork
from autorag_research.orm.uow.multi_modal_uow import MultiModalUnitOfWork
from autorag_research.orm.uow.retrieval_uow import RetrievalUnitOfWork
from autorag_research.orm.uow.text_uow import TextOnlyUnitOfWork

__all__ = [
    "BaseUnitOfWork",
    "MultiModalUnitOfWork",
    "RetrievalUnitOfWork",
    "TextOnlyUnitOfWork",
]
