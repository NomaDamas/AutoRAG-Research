"""Base Pipeline Service for AutoRAG-Research.

Provides shared pipeline management logic (get_or_create, config retrieval)
for both retrieval and generation pipeline services.
"""

import logging
from abc import ABC
from typing import Any

from autorag_research.orm.service.base import BaseService

logger = logging.getLogger("AutoRAG-Research")


class BasePipelineService(BaseService, ABC):
    """Abstract base for pipeline services with shared pipeline management.

    Provides:
    - get_or_create_pipeline(): Idempotent pipeline creation with resume support
    - get_pipeline_config(): Pipeline config retrieval by ID

    Subclasses must implement _create_uow() and _get_schema_classes() as required by BaseService.
    The UoW returned by _create_uow() must expose a `pipelines` property (PipelineRepository).
    """

    def get_or_create_pipeline(self, name: str, config: dict) -> tuple[int | str, bool]:
        """Get existing pipeline by name or create a new one.

        If a pipeline with the given name already exists, returns its ID.
        If the existing pipeline has a different config, logs a warning.
        If no pipeline exists, creates a new one.

        Args:
            name: Name for this pipeline (used as experiment identifier).
            config: Configuration dictionary for the pipeline.

        Returns:
            Tuple of (pipeline_id, is_new) where is_new is True if a new pipeline was created.
        """
        with self._create_uow() as uow:
            existing = uow.pipelines.get_by_name(name)
            if existing is not None:
                if existing.config != config:
                    logger.warning(
                        f"Pipeline '{name}' exists with different config. "
                        f"Existing: {existing.config}, New: {config}. Reusing existing pipeline."
                    )
                else:
                    logger.info(f"Resuming pipeline '{name}' (pipeline_id={existing.id})")
                return existing.id, False

            pipeline = self._get_schema_classes()["Pipeline"](name=name, config=config)
            uow.pipelines.add(pipeline)
            uow.flush()
            pipeline_id = pipeline.id
            uow.commit()
            logger.info(f"Created new pipeline '{name}' (pipeline_id={pipeline_id})")
            return pipeline_id, True

    def get_pipeline_config(self, pipeline_id: int | str) -> dict[Any, Any] | None:
        """Get pipeline configuration by ID.

        Args:
            pipeline_id: ID of the pipeline.

        Returns:
            Pipeline config dict if found, None otherwise.
        """
        with self._create_uow() as uow:
            pipeline = uow.pipelines.get_by_id(pipeline_id)
            return pipeline.config if pipeline else None
