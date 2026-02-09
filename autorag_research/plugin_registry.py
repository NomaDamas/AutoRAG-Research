"""Plugin registry for discovering and syncing external plugin configs.

This module provides entry_points-based discovery for external AutoRAG-Research plugins.
Plugins register via ``[project.entry-points]`` groups:

- ``autorag_research.pipelines`` - pipeline YAML configs
- ``autorag_research.metrics`` - metric YAML configs

After ``pip install <plugin>``, run ``autorag-research plugin sync`` to copy
plugin YAML configs into the local ``configs/`` directory.

Example plugin ``pyproject.toml``::

    [project.entry-points."autorag_research.pipelines"]
    elasticsearch = "autorag_research_elasticsearch.pipelines"
"""

import logging
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import entry_points
from importlib.resources import files
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from importlib.resources.abc import Traversable
    else:
        from importlib.abc import Traversable

logger = logging.getLogger("AutoRAG-Research")


@dataclass(frozen=True)
class PluginConfigInfo:
    """Metadata for a single YAML config discovered from a plugin package."""

    plugin_name: str
    config_name: str
    description: str
    source_path: Path
    subcategory: str | None
    category: str


@dataclass(frozen=True)
class SyncResult:
    """Result of syncing a single plugin config file."""

    config_name: str
    plugin_name: str
    destination: Path
    copied: bool
    reason: str


@lru_cache(maxsize=4)
def discover_plugin_configs(group: str, category: str) -> list[PluginConfigInfo]:
    """Discover YAML configs from installed plugin packages.

    Scans entry_points for the given *group*, loads each entry point module,
    and finds YAML files inside the module's package using
    ``importlib.resources.files()``.

    Supports two layouts inside a plugin package:

    - **Flat**: ``module/es_search.yaml``
    - **Nested**: ``module/retrieval/es_search.yaml``

    Args:
        group: Entry-point group name (e.g. ``"autorag_research.pipelines"``).
        category: Config category label (``"pipelines"`` or ``"metrics"``).

    Returns:
        List of :class:`PluginConfigInfo` for every YAML found.
    """
    results: list[PluginConfigInfo] = []
    try:
        eps = entry_points(group=group)
    except Exception as e:
        logger.warning(f"Failed to query entry_points for group {group}: {e}")
        return results

    for ep in eps:
        try:
            module = ep.load()
            if isinstance(module, ModuleType):
                results.extend(_scan_module_yamls(module, ep.name, category))
            else:
                logger.warning(f"Plugin {ep.name} entry point did not resolve to a module")
        except Exception as e:
            logger.warning(f"Failed to load plugin {ep.name} from group {group}: {e}")

    return results


def _scan_module_yamls(module: ModuleType, plugin_name: str, category: str) -> list[PluginConfigInfo]:
    """Scan a module's package directory for YAML config files.

    Args:
        module: The loaded module object.
        plugin_name: Name of the entry-point (used as plugin_name).
        category: ``"pipelines"`` or ``"metrics"``.

    Returns:
        List of discovered :class:`PluginConfigInfo`.
    """
    results: list[PluginConfigInfo] = []
    try:
        pkg_files = files(module)
    except Exception as e:
        logger.warning(f"Could not access package files for plugin {plugin_name}: {e}")
        return results

    if pkg_files is None:
        return results

    for resource in pkg_files.iterdir():
        name = str(resource.name)
        if name.endswith(".yaml"):
            info = _parse_yaml_resource(resource, plugin_name, category, subcategory=None)
            if info is not None:
                results.append(info)
        elif resource.is_dir():
            results.extend(_scan_subdir_yamls(resource, plugin_name, category))

    return results


def _scan_subdir_yamls(subdir: "Traversable", plugin_name: str, category: str) -> list[PluginConfigInfo]:
    """Scan a subdirectory for YAML config files.

    Args:
        subdir: A traversable subdirectory resource.
        plugin_name: Name of the entry-point.
        category: ``"pipelines"`` or ``"metrics"``.

    Returns:
        List of discovered :class:`PluginConfigInfo`.
    """
    results: list[PluginConfigInfo] = []
    subdir_name = str(subdir.name)
    try:
        for sub_resource in subdir.iterdir():
            if str(sub_resource.name).endswith(".yaml"):
                info = _parse_yaml_resource(sub_resource, plugin_name, category, subcategory=subdir_name)
                if info is not None:
                    results.append(info)
    except Exception as e:
        logger.warning(f"Failed to scan subdirectory {subdir_name} for plugin {plugin_name}: {e}")
    return results


def _parse_yaml_resource(
    resource: "Traversable",
    plugin_name: str,
    category: str,
    subcategory: str | None,
) -> PluginConfigInfo | None:
    """Parse a YAML resource into a :class:`PluginConfigInfo`.

    Args:
        resource: An ``importlib.resources`` traversable object.
        plugin_name: Plugin entry-point name.
        category: ``"pipelines"`` or ``"metrics"``.
        subcategory: Subdirectory name (e.g. ``"retrieval"``) or ``None``.

    Returns:
        A :class:`PluginConfigInfo` or ``None`` if parsing fails.
    """
    try:
        text = resource.read_text(encoding="utf-8")
        cfg = yaml.safe_load(text) or {}
        config_name = str(resource.name).removesuffix(".yaml")
        description = cfg.get("description", "")
        source_path = Path(str(resource))

        if subcategory is None:
            subcategory = _infer_subcategory(cfg)

        return PluginConfigInfo(
            plugin_name=plugin_name,
            config_name=config_name,
            description=description,
            source_path=source_path,
            subcategory=subcategory,
            category=category,
        )
    except Exception as e:
        logger.warning(f"Failed to parse YAML from plugin {plugin_name}: {e}")
        return None


def _infer_subcategory(cfg: dict) -> str | None:
    """Infer subcategory from the ``_target_`` field in a YAML config.

    When a plugin uses flat layout (no subdirectory), we inspect the ``_target_``
    string to guess whether it's a retrieval or generation config.

    Args:
        cfg: Parsed YAML config dictionary.

    Returns:
        ``"retrieval"``, ``"generation"``, or ``None`` if inference fails.
    """
    target = cfg.get("_target_", "")
    target_lower = target.lower()
    if "retrieval" in target_lower:
        return "retrieval"
    if "generation" in target_lower:
        return "generation"
    return None


def sync_plugin_configs(config_dir: Path) -> list[SyncResult]:
    """Copy plugin YAML configs into the local configs directory.

    Discovers all installed plugin configs (pipelines and metrics) and copies
    them into ``config_dir/pipelines/{subcategory}/`` or
    ``config_dir/metrics/{subcategory}/``. Existing files are never overwritten.

    Args:
        config_dir: Root config directory (e.g. ``./configs``).

    Returns:
        List of :class:`SyncResult` describing what was copied or skipped.
    """
    results: list[SyncResult] = []

    for group, category in [
        ("autorag_research.pipelines", "pipelines"),
        ("autorag_research.metrics", "metrics"),
    ]:
        configs = _discover_plugin_configs_uncached(group, category)
        for info in configs:
            dest_dir = config_dir / category
            if info.subcategory:
                dest_dir = dest_dir / info.subcategory
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest_file = dest_dir / f"{info.config_name}.yaml"
            if dest_file.exists():
                results.append(
                    SyncResult(
                        config_name=info.config_name,
                        plugin_name=info.plugin_name,
                        destination=dest_file,
                        copied=False,
                        reason="already exists",
                    )
                )
            else:
                shutil.copy2(info.source_path, dest_file)
                results.append(
                    SyncResult(
                        config_name=info.config_name,
                        plugin_name=info.plugin_name,
                        destination=dest_file,
                        copied=True,
                        reason="copied",
                    )
                )

    return results


def _discover_plugin_configs_uncached(group: str, category: str) -> list[PluginConfigInfo]:
    """Non-cached version of :func:`discover_plugin_configs` for sync.

    Sync needs fresh results every call (user may install plugins between
    syncs), so it bypasses the ``lru_cache``.
    """
    results: list[PluginConfigInfo] = []
    try:
        eps = entry_points(group=group)
    except Exception as e:
        logger.warning(f"Failed to query entry_points for group {group}: {e}")
        return results

    for ep in eps:
        try:
            module = ep.load()
            if isinstance(module, ModuleType):
                results.extend(_scan_module_yamls(module, ep.name, category))
            else:
                logger.warning(f"Plugin {ep.name} entry point did not resolve to a module")
        except Exception as e:
            logger.warning(f"Failed to load plugin {ep.name} from group {group}: {e}")

    return results
