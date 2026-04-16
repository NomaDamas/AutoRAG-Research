"""Shared loader for retrieval pipelines and nested retrieval dependencies."""

import inspect
import logging
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.cli.config_resolver import ConfigResolver
from autorag_research.cli.utils import get_config_dir
from autorag_research.config import BasePipelineConfig, BaseRetrievalPipelineConfig
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline

logger = logging.getLogger("AutoRAG-Research")

PIPELINE_TYPES = ["pipelines", "retrieval"]


class RetrievalPipelineLoader:
    """Load retrieval pipelines and inject nested retrieval dependencies."""

    _DEPENDENCY_FIELDS: tuple[tuple[str, str], ...] = (
        ("retrieval_pipeline_name", "_retrieval_pipeline"),
        ("inner_retrieval_pipeline_name", "_inner_retrieval_pipeline"),
    )

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        schema: Any | None = None,
        config_dir: Path | None = None,
        config_resolver: ConfigResolver | None = None,
        dependency_pipelines: dict[str, BaseRetrievalPipeline] | None = None,
    ) -> None:
        """Initialize the retrieval pipeline loader."""
        self.session_factory = session_factory
        self.schema = schema
        self.config_dir = config_dir or get_config_dir()
        self._config_resolver = config_resolver or ConfigResolver(self.config_dir)
        self._dependency_pipelines = dependency_pipelines if dependency_pipelines is not None else {}

    def resolve_dependencies(
        self,
        config: BasePipelineConfig,
        resolving_stack: tuple[str, ...] = (),
        resolution_key: str | None = None,
    ) -> None:
        """Resolve retrieval-pipeline dependencies for configs that declare them."""
        inject_retrieval_pipeline = getattr(config, "inject_retrieval_pipeline", None)

        if not callable(inject_retrieval_pipeline):
            return

        dependency_name = ""
        existing_pipeline = None
        for dependency_attr, existing_attr in self._DEPENDENCY_FIELDS:
            candidate_name = getattr(config, dependency_attr, "")
            if not candidate_name:
                continue
            dependency_name = candidate_name
            existing_pipeline = getattr(config, existing_attr, None)
            break

        if not dependency_name:
            return

        if existing_pipeline is not None:
            logger.debug(f"Retrieval pipeline already injected for {config.name}")
            return

        current_stack = self._extend_resolution_stack(config, resolving_stack, resolution_key)
        if dependency_name in current_stack:
            cycle_path = " -> ".join([*current_stack, dependency_name])
            msg = f"Cyclic retrieval pipeline dependency detected: {cycle_path}"
            raise ValueError(msg)

        inject_retrieval_pipeline(self.load_pipeline(dependency_name, current_stack))

    def load_pipeline(
        self,
        name: str,
        resolving_stack: tuple[str, ...] = (),
    ) -> BaseRetrievalPipeline:
        """Load a retrieval pipeline by config name and resolve nested wrappers."""
        if name in self._dependency_pipelines:
            logger.debug(f"Using cached retrieval pipeline: {name}")
            return self._dependency_pipelines[name]

        logger.info(f"Resolving retrieval pipeline dependency: {name}")
        pipeline_cfg = self._config_resolver.resolve_config(PIPELINE_TYPES, name)
        pipeline_config: BaseRetrievalPipelineConfig = instantiate(pipeline_cfg)
        resolved_name = getattr(pipeline_config, "name", "")
        if resolved_name and resolved_name in resolving_stack:
            cycle_path = " -> ".join([*resolving_stack, name])
            msg = f"Cyclic retrieval pipeline dependency detected: {cycle_path}"
            raise ValueError(msg)

        self.resolve_dependencies(pipeline_config, resolving_stack, name)
        pipeline = self._instantiate_pipeline(pipeline_config)
        self._dependency_pipelines[name] = pipeline
        return pipeline

    @staticmethod
    def _extend_resolution_stack(
        config: BasePipelineConfig,
        resolving_stack: tuple[str, ...],
        resolution_key: str | None,
    ) -> tuple[str, ...]:
        current_identifiers: list[str] = []
        for identifier in (resolution_key, getattr(config, "name", "")):
            if identifier and identifier not in current_identifiers:
                current_identifiers.append(identifier)

        return resolving_stack + tuple(
            identifier for identifier in current_identifiers if identifier not in resolving_stack
        )

    def _instantiate_pipeline(self, pipeline_config: BaseRetrievalPipelineConfig) -> BaseRetrievalPipeline:
        pipeline_class = pipeline_config.get_pipeline_class()
        pipeline_kwargs = pipeline_config.get_pipeline_kwargs()

        if self.config_dir is not None and "config_dir" in inspect.signature(pipeline_class.__init__).parameters:
            pipeline_kwargs.setdefault("config_dir", self.config_dir)

        return pipeline_class(
            session_factory=self.session_factory,
            name=pipeline_config.name,
            schema=self.schema,
            **pipeline_kwargs,
        )
