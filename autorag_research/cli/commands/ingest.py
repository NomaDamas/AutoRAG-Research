"""ingest command - Ingest datasets into PostgreSQL using Typer CLI.

This module provides a simplified CLI interface for data ingestion.
Instead of dynamic subcommands, it uses a single `ingest` command with:
- --name to specify the ingestor (beir, mteb, ragbench, etc.)
- --extra for ingestor-specific parameters as key=value pairs

This design enables true lazy loading - ingestor modules are only imported
when the command is actually executed, not at CLI startup.

Examples:
    autorag-research ingest --name=beir --extra dataset-name=scifact
    autorag-research ingest --name=mteb --extra task-name=NFCorpus
    autorag-research ingest -n ragbench -e configs=covidqa,msmarco
"""

import logging
from typing import TYPE_CHECKING, Annotated, Literal

import typer
from langchain_core.embeddings import Embeddings
from rich.console import Console

from autorag_research.cli.utils import discover_embedding_configs
from autorag_research.embeddings.base import MultiVectorMultiModalEmbedding
from autorag_research.injection import health_check_embedding, load_embedding_model
from autorag_research.orm.connection import DBConnection

if TYPE_CHECKING:
    from autorag_research.data.registry import IngestorMeta

logger = logging.getLogger("AutoRAG-Research")
console = Console()

# Create ingest sub-app (no lazy group needed - single command via callback)
ingest_app = typer.Typer(
    name="ingest",
    help="Ingest datasets into PostgreSQL.",
)


def _parse_extra_params(extra: list[str]) -> dict[str, str]:
    """Parse --extra key=value pairs into dict.

    Args:
        extra: List of "key=value" strings from CLI

    Returns:
        Dict mapping parameter names (with underscores) to values

    Raises:
        typer.BadParameter: If format is invalid
    """
    params = {}
    for item in extra:
        if "=" not in item:
            raise typer.BadParameter(f"Invalid format: '{item}'. Expected key=value")  # noqa: TRY003
        key, value = item.split("=", 1)
        # Convert CLI-style kebab-case to Python snake_case
        params[key.replace("-", "_")] = value
    return params


def _validate_required_params(meta: "IngestorMeta", extra_params: dict) -> None:
    """Validate that required ingestor params are provided.

    Args:
        meta: Ingestor metadata containing parameter definitions
        extra_params: Parsed extra parameters from CLI

    Raises:
        typer.Exit: If required parameter is missing
    """
    for param in meta.params:
        if param.required and param.name not in extra_params:
            typer.echo(f"Error: --extra {param.cli_option}=<value> is required for '{meta.name}'", err=True)
            if param.choices:
                typer.echo(f"  Available values: {', '.join(param.choices)}", err=True)
            raise typer.Exit(1)


def _convert_param_value(value: str, param_meta) -> str | int | bool | list[str]:
    """Convert string value to appropriate type based on parameter metadata.

    Args:
        value: String value from CLI
        param_meta: Parameter metadata with type info

    Returns:
        Converted value
    """
    # Handle list types (comma-separated)
    if param_meta.is_list:
        return [v.strip() for v in value.split(",")]

    # Handle basic types
    if param_meta.param_type is int:
        return int(value)
    elif param_meta.param_type is bool:
        return value.lower() in ("true", "1", "yes")

    return value


def generate_db_name(ingestor_name: str, params: dict, subset: str, embedding_model: str) -> str:
    """Generate database schema name from ingestor parameters.

    Examples:
        beir + dataset_name=scifact + test + bge-small -> beir_scifact_test_bge_small
        ragbench + configs=[covidqa, msmarco] + test + openai -> ragbench_covidqa_msmarco_test_openai
    """
    parts = [ingestor_name]

    # Include all parameter values
    for _param_name, param_value in params.items():
        if param_value is None:
            continue
        if isinstance(param_value, list):
            parts.extend([v.lower().replace("-", "_") for v in param_value])
        elif isinstance(param_value, str):
            parts.append(param_value.lower().replace("-", "_"))

    parts.append(subset)
    parts.append(embedding_model.lower().replace("-", "_"))
    return "_".join(parts)


@ingest_app.callback(invoke_without_command=True)
def ingest(  # noqa: C901
    ctx: typer.Context,
    # Required: ingestor name
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            "-n",
            help="Ingestor name (beir, mteb, ragbench, etc.). Use 'autorag-research show ingestors' to see all.",
        ),
    ] = None,
    # Dynamic parameters: --extra key=value
    extra: Annotated[
        list[str] | None,
        typer.Option(
            "--extra",
            "-e",
            help="Ingestor-specific params as key=value (e.g., --extra dataset-name=scifact)",
        ),
    ] = None,
    # Common options
    subset: Annotated[
        Literal["train", "dev", "test"], typer.Option("--subset", help="Dataset split: train, dev, or test")
    ] = "test",
    query_limit: Annotated[
        int | None, typer.Option("--query-limit", help="Maximum number of queries to ingest")
    ] = None,
    min_corpus_cnt: Annotated[
        int | None, typer.Option("--min-corpus-cnt", help="Minimum number of corpus documents to ingest")
    ] = None,
    db_name: Annotated[
        str | None, typer.Option("--db-name", help="Custom database name (auto-generated if not specified)")
    ] = None,
    # Embedding options
    embedding_model: Annotated[
        str, typer.Option("--embedding-model", help="Embedding model config name from configs/embedding/")
    ] = "openai-small",
    embed_batch_size: Annotated[int, typer.Option("--embed-batch-size", help="Batch size for embedding")] = 128,
    embed_concurrency: Annotated[int, typer.Option("--embed-concurrency", help="Max concurrent embedding calls")] = 16,
    skip_embedding: Annotated[
        bool, typer.Option("--skip-embedding", help="Skip embedding step (ingest data only)")
    ] = False,
) -> None:
    """Ingest a dataset into PostgreSQL.

    Specify the ingestor with --name and pass ingestor-specific parameters using --extra.
    Use 'autorag-research show ingestors' to see available ingestors and their parameters.

    Examples:
      autorag-research ingest --name=beir --extra dataset-name=scifact
      autorag-research ingest --name=mteb --extra task-name=NFCorpus --extra score-threshold=1
      autorag-research ingest -n ragbench -e configs=covidqa,msmarco
      autorag-research ingest --name=beir --extra dataset-name=scifact --skip-embedding
    """
    # If no --name provided, show help
    if name is None:
        typer.echo(ctx.get_help())
        typer.echo("\nUse 'autorag-research show ingestors' to see available ingestors and their parameters.")
        # Show available embedding configs
        typer.echo("\nAvailable embedding models (--embedding-model):")
        for emb_name, target in discover_embedding_configs().items():
            typer.echo(f"  {emb_name}: {target}")
        return

    # 1. Parse --extra parameters
    extra_params = _parse_extra_params(extra or [])

    # 2. Load ingestor metadata (lazy import)
    from autorag_research.data.registry import get_ingestor

    meta = get_ingestor(name)
    if meta is None:
        typer.echo(f"Error: Unknown ingestor '{name}'", err=True)
        typer.echo("Use 'autorag-research show ingestors' to see available ingestors.", err=True)
        raise typer.Exit(1)

    # 3. Validate required parameters
    _validate_required_params(meta, extra_params)

    # 4. Convert parameters to correct types and build init kwargs
    init_kwargs = {}
    for param in meta.params:
        if param.name in extra_params:
            init_kwargs[param.name] = _convert_param_value(extra_params[param.name], param)
        elif param.default is not None:
            init_kwargs[param.name] = param.default

    # Generate db_name from all parameters
    final_db_name = db_name or generate_db_name(name, init_kwargs, subset, embedding_model)

    # Load DB config from YAML first, then override with CLI options
    db_conn = DBConnection.from_config()
    db_conn.database = final_db_name

    typer.echo(f"\nIngesting dataset: {name}")
    for param in meta.params:
        param_value = init_kwargs.get(param.name)
        if param_value is not None:
            typer.echo(f"  {param.cli_option}: {param_value}")
    typer.echo(f"  subset: {subset}")
    typer.echo(f"  database: {db_conn.host}:{db_conn.port}/{final_db_name}")
    typer.echo(f"  embedding model: {embedding_model}")
    typer.echo("=" * 60)

    # 5. Load embedding model from YAML config
    try:
        with console.status(f"[bold blue]Loading embedding model: {embedding_model}..."):
            embed_model = load_embedding_model(embedding_model)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        typer.echo("\nAvailable embedding configs:", err=True)
        for emb_name in discover_embedding_configs():
            typer.echo(f"  {emb_name}", err=True)
        raise typer.Exit(1) from None
    console.print(f"[green]✓[/green] Loaded embedding model: {embedding_model}")

    # 6. Health check embedding model and get dimension
    try:
        with console.status("[bold blue]Checking embedding model health..."):
            embedding_dim = health_check_embedding(embed_model)
    except Exception as e:
        typer.echo(f"Error: Embedding health check failed: {e}", err=True)
        raise typer.Exit(1) from None
    console.print(f"[green]✓[/green] Embedding model healthy. Dimension: {embedding_dim}")

    # 7. Create ingestor with embedding model
    from autorag_research.data.base import MultiModalEmbeddingDataIngestor, TextEmbeddingDataIngestor
    from autorag_research.embeddings.base import SingleVectorMultiModalEmbedding

    ingestor_class = meta.ingestor_class
    if issubclass(ingestor_class, TextEmbeddingDataIngestor):
        if not isinstance(embed_model, Embeddings):
            raise TypeError("Text ingestor requires an Embeddings model")  # noqa: TRY003
        ingestor = ingestor_class(embed_model, **init_kwargs)
    elif issubclass(ingestor_class, MultiModalEmbeddingDataIngestor):
        if isinstance(embed_model, SingleVectorMultiModalEmbedding):
            ingestor = ingestor_class(embedding_model=embed_model, **init_kwargs)
        elif isinstance(embed_model, MultiVectorMultiModalEmbedding):
            ingestor = ingestor_class(late_interaction_embedding_model=embed_model, **init_kwargs)
        else:
            typer.echo(
                "Error: Multi-modal ingestor requires a SingleVectorMultiModalEmbedding or MultiVectorMultiModalEmbedding",
                err=True,
            )
            raise typer.Exit(1)
    else:
        typer.echo(f"Error: Unknown ingestor type: {ingestor_class}", err=True)
        raise typer.Exit(1)

    # 8. Detect primary key type from dataset
    detected_pkey_type: Literal["bigint", "string"] = ingestor.detect_primary_key_type()
    console.print(f"[green]✓[/green] Detected primary key type: {detected_pkey_type}")

    # 9. Create database and schema
    db_conn.create_database()
    console.print(f"[green]✓[/green] Created database schema: {final_db_name}")

    # 10. Create session factory and service
    session_factory = db_conn.get_session_factory()
    schema = db_conn.create_schema(embedding_dim, detected_pkey_type)

    if issubclass(ingestor_class, TextEmbeddingDataIngestor):
        from autorag_research.orm.service.text_ingestion import TextDataIngestionService

        text_service = TextDataIngestionService(session_factory, schema)
        ingestor.set_service(text_service)
    else:
        from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService

        mm_service = MultiModalIngestionService(session_factory, schema)
        ingestor.set_service(mm_service)  # ty: ignore[invalid-argument-type]

    # 11. Ingest data
    subset_literal: Literal["train", "dev", "test"] = subset
    with console.status(f"[bold blue]Ingesting {name} dataset..."):
        ingestor.ingest(
            subset=subset_literal,
            query_limit=query_limit,
            min_corpus_cnt=min_corpus_cnt,
        )
    console.print("[green]✓[/green] Ingestion complete")

    # 12. Embed data (unless skipped)
    if not skip_embedding:
        console.print(
            f"[bold blue]Embedding all data (batch_size={embed_batch_size}, concurrency={embed_concurrency})...[/bold blue]"
        )
        if isinstance(embed_model, MultiVectorMultiModalEmbedding):
            ingestor.embed_all_late_interaction(max_concurrency=embed_concurrency, batch_size=embed_batch_size)  # ty: ignore[possibly-missing-attribute]
        else:
            ingestor.embed_all(max_concurrency=embed_concurrency, batch_size=embed_batch_size)
        console.print("[green]✓[/green] Embedding complete")
    else:
        console.print("[yellow]⊘[/yellow] Skipping embedding step (--skip-embedding)")

    # 13. Print summary
    typer.echo("\n" + "=" * 60)
    typer.echo("Ingestion Summary:")
    typer.echo(f"  Schema: {final_db_name}")
    typer.echo(f"  Embedding dimension: {embedding_dim}")
    typer.echo(f"  Primary key type: {detected_pkey_type}")
    typer.echo("\nNext steps:")
    typer.echo(f"  autorag-research run --db-name={final_db_name}")
