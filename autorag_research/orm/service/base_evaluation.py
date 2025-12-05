"""Base Evaluation Service for AutoRAG-Research.

Provides abstract base class for evaluation services.
"""

from abc import ABC

from autorag_research.orm.service.base import BaseService


class BaseEvaluationService(BaseService, ABC):
    """Abstract base class for evaluation services."""

    pass
