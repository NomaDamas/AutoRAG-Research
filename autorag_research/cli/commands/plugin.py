"""plugin command - Manage AutoRAG-Research plugins.

Commands:
    autorag-research plugin sync     Copy plugin YAML configs into configs/
    autorag-research plugin create   Scaffold a new plugin project
"""

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

logger = logging.getLogger("AutoRAG-Research")
console = Console()

plugin_app = typer.Typer(
    name="plugin",
    help="Manage AutoRAG-Research plugins.",
    no_args_is_help=True,
)


@plugin_app.command(name="sync")
def sync() -> None:
    """Discover installed plugins and copy their YAML configs into configs/.

    Scans all installed packages that register ``autorag_research.pipelines``
    or ``autorag_research.metrics`` entry points. YAML files are copied into
    the local ``configs/`` directory. Existing files are never overwritten.

    Example::

        pip install autorag-research-elasticsearch
        autorag-research plugin sync
    """
    from autorag_research.cli.utils import get_config_dir
    from autorag_research.plugin_registry import sync_plugin_configs

    config_dir = get_config_dir()
    results = sync_plugin_configs(config_dir)

    if not results:
        console.print("[yellow]No plugins found.[/yellow] Install a plugin package and try again.")
        return

    copied = [r for r in results if r.copied]
    skipped = [r for r in results if not r.copied]

    if copied:
        console.print(f"\n[green]Copied {len(copied)} config(s):[/green]")
        for r in copied:
            console.print(f"  [green]+[/green] {r.destination.relative_to(config_dir)}  (from plugin: {r.plugin_name})")

    if skipped:
        console.print(f"\n[yellow]Skipped {len(skipped)} config(s) (already exist):[/yellow]")
        for r in skipped:
            console.print(
                f"  [yellow]=[/yellow] {r.destination.relative_to(config_dir)}  (from plugin: {r.plugin_name})"
            )

    console.print(f"\nTotal: {len(copied)} copied, {len(skipped)} skipped")


@plugin_app.command(name="create")
def create(
    name: Annotated[str, typer.Argument(help="Plugin name (e.g., 'my_custom_retrieval')")],
    plugin_type: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Plugin type: retrieval, generation, metric_retrieval, metric_generation, ingestor",
        ),
    ],
) -> None:
    """Scaffold a new plugin project in the current directory.

    Creates a plugin directory structure with:
    - pyproject.toml with entry_points configured
    - Pipeline/metric skeleton code
    - YAML config file
    - Basic test file

    Example::

        autorag-research plugin create my_search --type=retrieval
        cd my_search_plugin
        pip install -e .
        autorag-research plugin sync
    """
    from autorag_research.util import validate_plugin_name

    if not validate_plugin_name(name):
        console.print(
            "[red]Error:[/red] Plugin name must start with a lowercase letter"
            " and contain only lowercase letters, digits, and underscores."
        )
        raise typer.Exit(1)

    valid_types = {"retrieval", "generation", "metric_retrieval", "metric_generation", "ingestor"}
    if plugin_type not in valid_types:
        console.print(f"[red]Error:[/red] --type must be one of: {', '.join(sorted(valid_types))}")
        raise typer.Exit(1)

    plugin_dir_name = f"{name}_plugin"
    plugin_dir = Path.cwd() / plugin_dir_name
    package_name = f"{name}_plugin"

    if plugin_dir.exists():
        console.print(f"[red]Error:[/red] Directory '{plugin_dir_name}' already exists.")
        raise typer.Exit(1)

    _scaffold_plugin(plugin_dir, package_name, name, plugin_type)

    console.print(f"\n[green]Created plugin scaffold:[/green] {plugin_dir_name}/")
    console.print("\nNext steps:")
    console.print(f"  1. cd {plugin_dir_name}")
    if plugin_type == "ingestor":
        console.print(f"  2. Edit src/{package_name}/ingestor.py  (implement your logic)")
        console.print("  3. pip install -e .")
        console.print(f"     (uv users: uv add --editable ./{plugin_dir_name})")
        console.print(f"  4. autorag-research ingest --name={name}")
    else:
        console.print(f"  2. Edit src/{package_name}/pipeline.py  (implement your logic)")
        console.print("  3. pip install -e .")
        console.print(f"     (uv users: uv add --editable ./{plugin_dir_name})")
        console.print("  4. Go back to root directory and run:")
        console.print("  autorag-research plugin sync")


def _scaffold_plugin(plugin_dir: Path, package_name: str, name: str, plugin_type: str) -> None:
    """Create the plugin directory structure with skeleton files.

    Args:
        plugin_dir: Root directory for the plugin project.
        package_name: Python package name (e.g. ``my_search_plugin``).
        name: Base plugin name (e.g. ``my_search``).
        plugin_type: One of ``retrieval``, ``generation``, ``metric_retrieval``, ``metric_generation``.
    """
    src_dir = plugin_dir / "src" / package_name
    test_dir = plugin_dir / "tests"

    src_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    is_ingestor = plugin_type == "ingestor"
    is_metric = plugin_type.startswith("metric_")
    base_type = plugin_type.removeprefix("metric_") if is_metric else plugin_type

    # Generate files
    _write_pyproject_toml(plugin_dir, package_name, name, plugin_type, is_metric, base_type)
    _write_init_py(src_dir)
    if is_ingestor:
        _write_ingestor_py(src_dir, package_name, name)
        _write_ingestor_test(test_dir, package_name, name)
    elif is_metric:
        _write_metric_py(src_dir, package_name, name, base_type)
        _write_metric_yaml(src_dir, package_name, name, base_type)
        _write_metric_test(test_dir, package_name, name, base_type)
    else:
        _write_pipeline_py(src_dir, package_name, name, base_type)
        _write_pipeline_yaml(src_dir, package_name, name, base_type)
        _write_pipeline_test(test_dir, package_name, name, base_type)


def _write_pyproject_toml(
    plugin_dir: Path,
    package_name: str,
    name: str,
    plugin_type: str,
    is_metric: bool,
    base_type: str,
) -> None:
    """Write pyproject.toml with entry_points."""
    if plugin_type == "ingestor":
        entry_group = "autorag_research.ingestors"
        entry_value = f"{package_name}"
    elif is_metric:
        entry_group = "autorag_research.metrics"
        entry_value = f"{package_name}"
    else:
        entry_group = "autorag_research.pipelines"
        entry_value = f"{package_name}"

    content = f'''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{name}-plugin"
version = "0.1.0"
description = "AutoRAG-Research plugin: {name}"
requires-python = ">=3.10"
dependencies = [
    "autorag-research",
]

[project.entry-points."{entry_group}"]
{name} = "{entry_value}"

[tool.hatch.build.targets.wheel]
packages = ["src/{package_name}"]
'''
    (plugin_dir / "pyproject.toml").write_text(content)


def _write_init_py(src_dir: Path) -> None:
    """Write __init__.py."""
    (src_dir / "__init__.py").write_text("")


def _write_pipeline_py(src_dir: Path, package_name: str, name: str, base_type: str) -> None:
    """Write pipeline.py skeleton for retrieval or generation plugin."""
    class_name = _to_class_name(name)
    if base_type == "retrieval":
        content = f'''"""Pipeline implementation for {name} plugin."""

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig, PipelineType
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline


@dataclass(kw_only=True)
class {class_name}PipelineConfig(BaseRetrievalPipelineConfig):
    """{class_name} pipeline configuration."""

    pipeline_type: PipelineType = field(default=PipelineType.RETRIEVAL, init=False)

    def get_pipeline_class(self) -> type["{class_name}Pipeline"]:
        return {class_name}Pipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        return {{}}


class {class_name}Pipeline(BaseRetrievalPipeline):
    """{class_name} retrieval pipeline."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        schema: Any | None = None,
        **kwargs: Any,
    ):
        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        return {{"type": "{name}"}}

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        raise NotImplementedError("Implement retrieval by query ID")

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        raise NotImplementedError("Implement retrieval by query text")
'''
    else:
        content = f'''"""Pipeline implementation for {name} plugin."""

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseGenerationPipelineConfig, PipelineType
from autorag_research.orm.service.generation_pipeline import GenerationResult
from autorag_research.pipelines.generation.base import BaseGenerationPipeline
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline


@dataclass(kw_only=True)
class {class_name}PipelineConfig(BaseGenerationPipelineConfig):
    """{class_name} pipeline configuration."""

    pipeline_type: PipelineType = field(default=PipelineType.GENERATION, init=False)

    def get_pipeline_class(self) -> type["{class_name}Pipeline"]:
        return {class_name}Pipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        return {{}}


class {class_name}Pipeline(BaseGenerationPipeline):
    """{class_name} generation pipeline."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        schema: Any | None = None,
        **kwargs: Any,
    ):
        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        return {{"type": "{name}"}}

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        raise NotImplementedError("Implement generation logic")
'''
    (src_dir / "pipeline.py").write_text(content)


def _write_pipeline_yaml(src_dir: Path, package_name: str, name: str, base_type: str) -> None:
    """Write pipeline YAML config."""
    class_name = _to_class_name(name)
    content = f"""_target_: {package_name}.pipeline.{class_name}PipelineConfig
description: "{class_name} {base_type} pipeline"
name: {name}
top_k: 10
batch_size: 128
max_concurrency: 16
max_retries: 3
retry_delay: 1.0
"""
    if base_type == "generation":
        content += """retrieval_pipeline_name: bm25
llm: gpt-4o-mini
"""
    yaml_dir = src_dir / base_type
    yaml_dir.mkdir(parents=True, exist_ok=True)
    (yaml_dir / f"{name}.yaml").write_text(content)


def _write_metric_py(src_dir: Path, package_name: str, name: str, base_type: str) -> None:
    """Write metric.py skeleton."""
    class_name = _to_class_name(name)
    if base_type == "retrieval":
        base_config = "BaseRetrievalMetricConfig"
        base_import = "from autorag_research.config import BaseRetrievalMetricConfig"
    else:
        base_config = "BaseGenerationMetricConfig"
        base_import = "from autorag_research.config import BaseGenerationMetricConfig"

    content = f'''"""Metric implementation for {name} plugin."""

from collections.abc import Callable
from dataclasses import dataclass

{base_import}


def {name}_metric(**kwargs) -> float:
    """Compute the {name} metric.

    Implement your metric logic here.
    """
    raise NotImplementedError("Implement metric computation")


@dataclass
class {class_name}MetricConfig({base_config}):
    """{class_name} metric configuration."""

    def get_metric_func(self) -> Callable:
        return {name}_metric
'''
    (src_dir / "metric.py").write_text(content)


def _write_metric_yaml(src_dir: Path, package_name: str, name: str, base_type: str) -> None:
    """Write metric YAML config."""
    class_name = _to_class_name(name)
    content = f"""_target_: {package_name}.metric.{class_name}MetricConfig
description: "{class_name} {base_type} metric"
"""
    yaml_dir = src_dir / base_type
    yaml_dir.mkdir(parents=True, exist_ok=True)
    (yaml_dir / f"{name}.yaml").write_text(content)


def _write_metric_test(test_dir: Path, package_name: str, name: str, base_type: str) -> None:
    """Write basic test skeleton for metric plugin."""
    class_name = _to_class_name(name)
    content = f'''"""Tests for {name} metric plugin."""

from {package_name}.metric import {class_name}MetricConfig


def test_{name}_metric_config():
    """Test that the metric config can be instantiated."""
    config = {class_name}MetricConfig()
    assert config.get_metric_func() is not None
'''
    (test_dir / f"test_{name}.py").write_text(content)


def _write_pipeline_test(test_dir: Path, package_name: str, name: str, base_type: str) -> None:
    """Write basic test skeleton for pipeline plugin."""
    class_name = _to_class_name(name)
    content = f'''"""Tests for {name} pipeline plugin."""

from {package_name}.pipeline import {class_name}PipelineConfig


def test_{name}_pipeline_config():
    """Test that the pipeline config can be instantiated."""
    config = {class_name}PipelineConfig(name="{name}")
    assert config.name == "{name}"
'''
    if base_type == "generation":
        content = f'''"""Tests for {name} pipeline plugin."""

from unittest.mock import MagicMock

from {package_name}.pipeline import {class_name}PipelineConfig


def test_{name}_pipeline_config():
    """Test that the pipeline config can be instantiated."""
    config = {class_name}PipelineConfig(
        name="{name}",
        llm=MagicMock(),
        retrieval_pipeline_name="bm25",
    )
    assert config.name == "{name}"
'''
    (test_dir / f"test_{name}.py").write_text(content)


def _write_ingestor_py(src_dir: Path, package_name: str, name: str) -> None:
    """Write ingestor.py skeleton for ingestor plugin."""
    class_name = _to_class_name(name)
    content = f'''"""Ingestor implementation for {name} plugin."""

from typing import Literal

from langchain_core.embeddings import Embeddings

from autorag_research.data.base import TextEmbeddingDataIngestor
from autorag_research.data.registry import register_ingestor

DATASETS = Literal["dataset_a", "dataset_b"]


@register_ingestor(name="{name}", description="{class_name} data ingestor")
class {class_name}Ingestor(TextEmbeddingDataIngestor):
    """{class_name} data ingestor."""

    def __init__(
        self,
        embedding_model: Embeddings,
        dataset_name: DATASETS,
    ):
        super().__init__(embedding_model)
        self.dataset_name = dataset_name

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
    ) -> None:
        raise NotImplementedError("Implement data ingestion logic")

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        raise NotImplementedError("Return \\"bigint\\" or \\"string\\"")
'''
    (src_dir / "ingestor.py").write_text(content)


def _write_ingestor_test(test_dir: Path, package_name: str, name: str) -> None:
    """Write basic test skeleton for ingestor plugin."""
    class_name = _to_class_name(name)
    content = f'''"""Tests for {name} ingestor plugin."""

from langchain_core.embeddings import FakeEmbeddings

from {package_name}.ingestor import {class_name}Ingestor


def test_{name}_ingestor_instantiation():
    """Test that the ingestor can be instantiated."""
    embeddings = FakeEmbeddings(size=128)
    ingestor = {class_name}Ingestor(
        embedding_model=embeddings,
        dataset_name="dataset_a",
    )
    assert ingestor.dataset_name == "dataset_a"
    assert ingestor.embedding_model is embeddings
'''
    (test_dir / f"test_{name}.py").write_text(content)


def _to_class_name(name: str) -> str:
    """Convert snake_case name to PascalCase class name.

    Args:
        name: A snake_case plugin name (e.g. ``my_search``).

    Returns:
        PascalCase string (e.g. ``MySearch``).
    """
    return "".join(part.capitalize() for part in name.split("_"))
