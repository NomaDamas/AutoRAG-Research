"""ImageChunkRetrievedResult repository for AutoRAG-Research.

Implements CRUD operations for storing and querying image chunk retrieval results
from various retrieval pipelines like CLIP, ColPali, etc.
"""

from sqlalchemy.orm import Session

from autorag_research.orm.repository.chunk_retrieved_result import BaseRetrievedResultRepository


class ImageChunkRetrievedResultRepository(BaseRetrievedResultRepository):
    """Repository for ImageChunkRetrievedResult entity."""

    def __init__(self, session: Session, model_cls: type | None = None):
        """Initialize image chunk retrieved result repository.

        Args:
            session: SQLAlchemy session for database operations.
            model_cls: The ImageChunkRetrievedResult model class to use. If None, uses default schema.
        """
        if model_cls is None:
            from autorag_research.orm.schema import ImageChunkRetrievedResult

            model_cls = ImageChunkRetrievedResult
        super().__init__(session, model_cls)
