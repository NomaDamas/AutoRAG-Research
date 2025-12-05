"""Retrieval Evaluation Service for AutoRAG-Research.

Provides service layer for evaluating retrieval pipelines:
1. Fetch queries and ground truth from database
2. Fetch retrieval results
3. Compute evaluation metrics
4. Store evaluation results
"""

import logging

from autorag_research.orm.service.base_evaluation import BaseEvaluationService
from autorag_research.orm.uow.evaluation_uow import RetrievalEvaluationUnitOfWork

__all__ = ["RetrievalEvaluationService"]

logger = logging.getLogger("AutoRAG-Research")


class RetrievalEvaluationService(BaseEvaluationService):
    """Service for evaluating retrieval pipelines.

    This service handles the evaluation workflow for retrieval pipelines:
    1. Fetch queries and ground truth (RetrievalRelation)
    2. Fetch retrieval results (ChunkRetrievedResult)
    3. Compute evaluation metrics (e.g., Recall@K, Precision@K, MRR)
    4. Store results in EvaluationResult table

    Example:
        ```python
        from autorag_research.orm.service import RetrievalEvaluationService

        # Create service
        service = RetrievalEvaluationService(session_factory, schema)

        # Evaluate a pipeline (to be implemented)
        # results = service.evaluate_pipeline(
        #     pipeline_id=pipeline_id,
        #     metric_names=["recall@10", "precision@10"],
        # )
        ```
    """

    def _create_uow(self) -> RetrievalEvaluationUnitOfWork:
        """Create a new RetrievalEvaluationUnitOfWork instance.

        Returns:
            RetrievalEvaluationUnitOfWork for managing evaluation transactions.
        """
        return RetrievalEvaluationUnitOfWork(self.session_factory, self._schema)
