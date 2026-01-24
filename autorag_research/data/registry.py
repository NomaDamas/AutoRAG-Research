"""Ingestor registry with decorator-based registration and automatic parameter detection.

This module provides a decorator-based registration system for data ingestors.
CLI parameters are automatically extracted from __init__ signatures using type hints.

Example:
    from typing import Literal
    from autorag_research.data.registry import register_ingestor

    DATASETS = Literal["dataset_a", "dataset_b"]

    @register_ingestor(name="my_ingestor", description="My ingestor description")
    class MyIngestor(TextEmbeddingDataIngestor):
        def __init__(
            self,
            embedding_model: BaseEmbedding,  # Skipped (known dependency)
            dataset_name: DATASETS,           # -> --dataset-name, choices=[...], required
            batch_size: int = 100,            # -> --batch-size, default=100
        ):
            ...
"""

import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from importlib.metadata import entry_points
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

logger = logging.getLogger("AutoRAG-Research")

# Known parameters to skip (injected by CLI, not user-provided)
SKIP_PARAMS = {"self", "embedding_model", "late_interaction_embedding_model"}


@dataclass
class ParamMeta:
    """CLI parameter metadata extracted from __init__ signature."""

    name: str  # Python param name (e.g., "dataset_name")
    cli_option: str  # CLI option name (e.g., "dataset-name")
    param_type: type  # Python type (str, int, bool, etc.)
    choices: list[str] | None = None  # From Literal[...] or Enum
    required: bool = True
    default: Any = None
    help: str = ""
    is_list: bool = False  # For list[str] parameters


@dataclass
class IngestorMeta:
    """Ingestor metadata registered via decorator."""

    name: str
    ingestor_class: type
    description: str
    params: list[ParamMeta] = field(default_factory=list)


# Global registry (populated by @register_ingestor decorators at import time)
_INGESTOR_REGISTRY: dict[str, IngestorMeta] = {}

# Modules in autorag_research.data that are NOT ingestors (skip during auto-discovery)
_NON_INGESTOR_MODULES = {"base", "registry", "restore", "util"}


def register_ingestor(
    name: str,
    description: str = "",
):
    """Decorator to register an ingestor class.

    CLI parameters are automatically extracted from __init__ signature.
    Use Literal type hints to define choices.

    Args:
        name: CLI command name (e.g., "beir")
        description: Help text for CLI

    Example:
        @register_ingestor(name="beir", description="BEIR benchmark")
        class BEIRIngestor(TextEmbeddingDataIngestor):
            def __init__(
                self,
                embedding_model: BaseEmbedding,
                dataset_name: Literal["msmarco", "scifact"],
                subset: Literal["train", "dev", "test"] = "test",
            ):
                ...
    """

    def decorator(cls):
        params = _extract_params_from_init(cls)
        _INGESTOR_REGISTRY[name] = IngestorMeta(
            name=name,
            ingestor_class=cls,
            description=description,
            params=params,
        )
        return cls

    return decorator


def _extract_params_from_init(cls) -> list[ParamMeta]:
    """Extract CLI parameters from __init__ signature."""
    try:
        hints = get_type_hints(cls.__init__)
    except Exception:
        hints = {}

    sig = inspect.signature(cls.__init__)
    params = []

    for param_name, param in sig.parameters.items():
        if param_name in SKIP_PARAMS:
            continue

        hint = hints.get(param_name, str)
        choices = _extract_choices(hint)
        param_type = _get_base_type(hint)
        is_list = _is_list_type(hint)
        has_default = param.default is not inspect.Parameter.empty

        params.append(
            ParamMeta(
                name=param_name,
                cli_option=param_name.replace("_", "-"),
                param_type=param_type,
                choices=choices,
                required=not has_default,
                default=param.default if has_default else None,
                is_list=is_list,
            )
        )

    return params


def _extract_choices(hint) -> list[str] | None:
    """Extract choices from Literal type hint or Enum."""
    origin = get_origin(hint)

    # Handle Literal[...]
    if origin is Literal:
        return [str(arg) for arg in get_args(hint)]

    # Handle Enum subclasses
    if isinstance(hint, type) and issubclass(hint, Enum):
        return [member.value for member in hint]

    return None


def _get_base_type(hint) -> type:
    """Get base type from type hint (unwrap Literal, Optional, list, etc.)."""
    origin = get_origin(hint)

    # Handle Literal[...]
    if origin is Literal:
        args = get_args(hint)
        if args:
            return type(args[0])
        return str

    # Handle list[...]
    if origin is list:
        return str  # For CLI, list params are comma-separated strings

    # Handle Union (e.g., str | None)
    if origin is Union:
        args = get_args(hint)
        # Return first non-None type
        for arg in args:
            if arg is not type(None):
                return _get_base_type(arg)
        return str

    # Handle Enum
    if isinstance(hint, type) and issubclass(hint, Enum):
        return str

    # Return the type directly if it's a basic type
    if isinstance(hint, type):
        return hint

    return str


def _is_list_type(hint) -> bool:
    """Check if type hint is a list type."""
    origin = get_origin(hint)

    if origin is list:
        return True

    # Handle Union (e.g., list[str] | None)
    if origin is Union:
        args = get_args(hint)
        for arg in args:
            if get_origin(arg) is list:
                return True

    return False


def get_ingestor(name: str) -> IngestorMeta | None:
    """Get ingestor metadata by name."""
    ingestor_registry = discover_ingestors()
    return ingestor_registry.get(name)


@lru_cache(maxsize=1)
def discover_ingestors() -> dict[str, IngestorMeta]:
    """Discover all registered ingestors (internal + plugins).

    Internal ingestors are auto-discovered by scanning autorag_research.data package.
    External plugins are loaded via entry_points.

    Results are cached after first call via @lru_cache.
    """
    # Auto-import all modules in autorag_research.data
    _auto_import_data_modules()

    # Load external plugins
    _load_plugin_ingestors()

    return _INGESTOR_REGISTRY


def _auto_import_data_modules() -> None:
    """Auto-discover and import all modules in autorag_research.data package."""
    import contextlib
    import importlib
    import pkgutil

    import autorag_research.data as data_package

    for _finder, modname, ispkg in pkgutil.iter_modules(data_package.__path__):
        if ispkg or modname.startswith("_") or modname in _NON_INGESTOR_MODULES:
            continue
        with contextlib.suppress(ImportError):
            importlib.import_module(f"autorag_research.data.{modname}")


def _load_plugin_ingestors() -> None:
    """Load external ingestor plugins via entry_points."""
    try:
        eps = entry_points(group="autorag_research.ingestors")
        for ep in eps:
            try:
                ep.load()  # Import module -> triggers @register_ingestor
                logger.info(f"Loaded ingestor plugin: {ep.name}")
            except Exception as e:
                logger.warning(f"Failed to load ingestor plugin {ep.name}: {e}")
    except Exception as e:
        logger.warning(f"Failed to query ingestor entry_points: {e}")


def get_ingestor_help() -> str:
    """Generate help text showing available ingestors and their parameters."""
    ingestors = discover_ingestors()
    lines = ["Available ingestors:"]

    for name, meta in sorted(ingestors.items()):
        lines.append(f"\n  {name}: {meta.description}")
        if meta.params:
            for param in meta.params:
                default_str = f" (default: {param.default})" if not param.required else " (required)"
                choices_str = f" choices: {param.choices}" if param.choices else ""
                lines.append(f"    --{param.cli_option}{default_str}{choices_str}")

    return "\n".join(lines)
