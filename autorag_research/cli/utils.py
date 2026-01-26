"""CLI utility functions."""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding

logger = logging.getLogger("AutoRAG-Research")


# =============================================================================
# Config Discovery Functions
# =============================================================================


def discover_configs(config_dir: Path) -> dict[str, str]:
    """Scan YAML configs and return {name: description} dict.

    Args:
        config_dir: Directory containing YAML config files (searches recursively).

    Returns:
        Dictionary mapping config name (filename without .yaml) to description.
        When files with the same stem exist in different subdirectories,
        the parent directory is used as prefix (e.g., "retrieval/recall").
    """
    result = {}
    if not config_dir.exists():
        raise FileNotFoundError

    # First pass: collect all files and detect collisions
    files_by_stem: dict[str, list[Path]] = {}
    for yaml_file in sorted(config_dir.glob("**/*.yaml")):
        stem = yaml_file.stem
        files_by_stem.setdefault(stem, []).append(yaml_file)

    # Second pass: build result with prefixes for collisions
    for stem, files in files_by_stem.items():
        use_prefix = len(files) > 1
        for yaml_file in files:
            try:
                with open(yaml_file) as f:
                    cfg = yaml.safe_load(f)
                description = cfg.get("description", cfg.get("_target_", "No description"))
            except Exception as e:
                logger.warning(f"Failed to parse {yaml_file}: {e}")
                description = "Error loading config"

            name = f"{yaml_file.parent.name}/{stem}" if use_prefix else stem
            result[name] = description

    return result


def discover_pipelines(pipeline_type: str) -> dict[str, str]:
    """Discover available pipelines from configs/pipelines/{pipeline_type}/.

    Args:
        pipeline_type: Type of pipeline to discover ("retrieval" or "generation").

    Returns:
        Dict: {name: description}
        Example: {"dense": "Dense retrieval pipeline...", "sparse": "..."}
    """
    return discover_configs(get_config_dir() / "pipelines" / pipeline_type)


def discover_metrics(pipeline_type: str) -> dict[str, str]:
    """Discover available metrics from configs/metrics/.

    Returns:
        Nested dict: {subdir: {name: description}}
        Example: {"retrieval": {"recall": "Recall@k..."}, "generation": {...}}
    """
    return discover_configs(get_config_dir() / "metrics" / pipeline_type)


# =============================================================================
# Path and Config Utilities
# =============================================================================


def get_config_dir() -> Path:
    """Get the configs directory.

    Returns CONFIG_PATH if set by CLI, otherwise falls back to CWD/configs.
    """
    import autorag_research.cli as cli

    return cli.CONFIG_PATH or Path.cwd() / "configs"


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# =============================================================================
# Embedding Model Utilities
# =============================================================================


def load_embedding_model(config_name: str) -> "BaseEmbedding":
    """Load LlamaIndex embedding model directly from YAML via Hydra instantiate.

    Args:
        config_name: Name of the embedding config file (without .yaml extension).
                    e.g., "openai-small", "openai-large", "openai-like"

    Returns:
        LlamaIndex BaseEmbedding instance.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    from hydra.utils import instantiate
    from llama_index.core.base.embeddings.base import BaseEmbedding

    yaml_path = get_config_dir() / "embedding" / f"{config_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError

    cfg = OmegaConf.load(yaml_path)
    model = instantiate(cfg)

    if not isinstance(model, BaseEmbedding):
        raise TypeError(f"Expected BaseEmbedding, got {type(model)}")  # noqa: TRY003

    return model


def health_check_embedding(model: "BaseEmbedding") -> int:
    """Health check embedding model and return embedding dimension.

    Args:
        model: LlamaIndex BaseEmbedding instance.

    Returns:
        Embedding dimension (length of embedding vector).

    Raises:
        EmbeddingNotSetError: If health check fails.
    """
    from autorag_research.exceptions import EmbeddingNotSetError

    try:
        embedding = model.get_text_embedding("health check")
        return len(embedding)
    except Exception as e:
        raise EmbeddingNotSetError from e


def discover_embedding_configs() -> dict[str, str]:
    """Discover available embedding configs from configs/embedding/.

    Returns:
        Dictionary mapping config name to _target_ class path.
    """
    return discover_configs(get_config_dir() / "embedding")
