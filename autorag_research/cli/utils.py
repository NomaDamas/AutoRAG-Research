"""CLI utility functions."""

import logging
import sys
from pathlib import Path

import yaml

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


def discover_embedding_configs() -> dict[str, str]:
    """Discover available embedding configs from configs/embedding/.

    Returns:
        Dictionary mapping config name to _target_ class path.
    """
    return discover_configs(get_config_dir() / "embedding")


def discover_llm_configs() -> dict[str, str]:
    """Discover available LLM configs from configs/llm/.

    Returns:
        Dictionary mapping config name to _target_ class path.
    """
    return discover_configs(get_config_dir() / "llm")


def discover_reranker_configs() -> dict[str, str]:
    """Discover available reranker configs from configs/reranker/.

    Returns:
        Dictionary mapping config name to _target_ class path.
    """
    return discover_configs(get_config_dir() / "reranker")
