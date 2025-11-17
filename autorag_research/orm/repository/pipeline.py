"""Pipeline repository for AutoRAG-Research.

Implements pipeline-specific CRUD operations and relationship queries
for managing RAG pipelines and their experiment results.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository.base import GenericRepository
from autorag_research.orm.schema import Pipeline


class PipelineRepository(GenericRepository[Pipeline]):
    """Repository for Pipeline entity with relationship loading capabilities."""

    def __init__(self, session: Session):
        """Initialize pipeline repository.

        Args:
            session: SQLAlchemy session for database operations.
        """
        super().__init__(session, Pipeline)

    def get_by_name(self, name: str) -> Pipeline | None:
        """Retrieve a pipeline by its name.

        Args:
            name: The pipeline name to search for.

        Returns:
            The pipeline if found, None otherwise.
        """
        stmt = select(Pipeline).where(Pipeline.name == name)
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_experiment_results(self, pipeline_id: int) -> Pipeline | None:
        """Retrieve a pipeline with its experiment results eagerly loaded.

        Args:
            pipeline_id: The pipeline ID.

        Returns:
            The pipeline with experiment results loaded, None if not found.
        """
        stmt = select(Pipeline).where(Pipeline.id == pipeline_id).options(joinedload(Pipeline.experiment_results))
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_summaries(self, pipeline_id: int) -> Pipeline | None:
        """Retrieve a pipeline with its summaries eagerly loaded.

        Args:
            pipeline_id: The pipeline ID.

        Returns:
            The pipeline with summaries loaded, None if not found.
        """
        stmt = select(Pipeline).where(Pipeline.id == pipeline_id).options(joinedload(Pipeline.summaries))
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_retrieved_results(self, pipeline_id: int) -> Pipeline | None:
        """Retrieve a pipeline with its retrieved results eagerly loaded.

        Args:
            pipeline_id: The pipeline ID.

        Returns:
            The pipeline with chunk and image chunk retrieved results loaded, None if not found.
        """
        stmt = (
            select(Pipeline)
            .where(Pipeline.id == pipeline_id)
            .options(
                joinedload(Pipeline.chunk_retrieved_results),
                joinedload(Pipeline.image_chunk_retrieved_results),
            )
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_all_relations(self, pipeline_id: int) -> Pipeline | None:
        """Retrieve a pipeline with all relations eagerly loaded.

        Args:
            pipeline_id: The pipeline ID.

        Returns:
            The pipeline with all relations loaded, None if not found.
        """
        stmt = (
            select(Pipeline)
            .where(Pipeline.id == pipeline_id)
            .options(
                joinedload(Pipeline.experiment_results),
                joinedload(Pipeline.summaries),
                joinedload(Pipeline.chunk_retrieved_results),
                joinedload(Pipeline.image_chunk_retrieved_results),
            )
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def search_by_name(self, search_text: str, limit: int = 10) -> list[Pipeline]:
        """Search pipelines containing the specified text in their name.

        Args:
            search_text: Text to search for in pipeline names.
            limit: Maximum number of results to return.

        Returns:
            List of pipelines containing the search text.
        """
        stmt = select(Pipeline).where(Pipeline.name.ilike(f"%{search_text}%")).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_all_ordered_by_name(self, limit: int | None = None, offset: int | None = None) -> list[Pipeline]:
        """Retrieve all pipelines ordered by name.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of all pipelines ordered by name.
        """
        stmt = select(Pipeline).order_by(Pipeline.name)
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_by_config_key(self, key: str, value: str | int | float | bool) -> list[Pipeline]:
        """Retrieve pipelines with a specific config key-value pair.

        Args:
            key: The config key to search for.
            value: The config value to match.

        Returns:
            List of pipelines with matching config.

        Note:
            Uses JSONB containment operator (@>) for efficient config searching.
        """
        stmt = select(Pipeline).where(Pipeline.config[key].as_string() == str(value))
        return list(self.session.execute(stmt).scalars().all())

    def exists_by_name(self, name: str) -> bool:
        """Check if a pipeline exists with the given name.

        Args:
            name: The pipeline name to check.

        Returns:
            True if pipeline exists, False otherwise.
        """
        return self.get_by_name(name) is not None
