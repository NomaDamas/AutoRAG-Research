"""ingest command - Ingest datasets into PostgreSQL using Typer CLI."""

import logging
from typing import TYPE_CHECKING, Annotated, Literal

import typer
import typer.main
import typer.models
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typer.core import TyperGroup, click

from autorag_research.cli.configs.db import DatabaseConfig
from autorag_research.cli.utils import (
    discover_embedding_configs,
    health_check_embedding,
    load_db_config_from_yaml,
    load_embedding_model,
)

if TYPE_CHECKING:
    from autorag_research.data.registry import IngestorMeta

logger = logging.getLogger("AutoRAG-Research")


class LazyIngestorGroup(TyperGroup):
    """TyperGroup that lazily registers ingestor commands on first access.

    This avoids importing heavy dependencies (mteb, beir, datasets) at CLI startup.
    Commands are only registered when the 'ingest' subcommand is actually invoked.
    """

    _commands_registered: bool = False

    def _ensure_commands_registered(self) -> None:
        """Register ingestor commands if not already done."""
        if LazyIngestorGroup._commands_registered:
            return
        LazyIngestorGroup._commands_registered = True

        from autorag_research.data.registry import discover_ingestors

        ingestors = discover_ingestors()
        for name, meta in ingestors.items():
            command_func = create_ingest_command(name, meta)
            # Convert to Click command and add directly to this group
            click_command = typer.main.get_command_from_info(
                typer.models.CommandInfo(
                    name=name,
                    callback=command_func,
                    help=meta.description,
                ),
                pretty_exceptions_short=ingest_app.pretty_exceptions_short,
                rich_markup_mode=ingest_app.rich_markup_mode,
            )
            self.add_command(click_command, name)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Get a command by name, ensuring commands are registered first."""
        self._ensure_commands_registered()
        return super().get_command(ctx, cmd_name)

    def list_commands(self, ctx: click.Context) -> list[str]:
        """List all commands, ensuring commands are registered first."""
        self._ensure_commands_registered()
        return super().list_commands(ctx)


# Create ingest sub-app with lazy command registration
ingest_app = typer.Typer(
    name="ingest",
    help="Ingest datasets into PostgreSQL.",
    no_args_is_help=True,
    cls=LazyIngestorGroup,
)


def generate_db_name(ingestor_name: str, params: dict, subset: str) -> str:
    """Generate database schema name from ingestor parameters.

    Examples:
        beir + dataset_name=scifact + test -> beir_scifact_test
        ragbench + configs=[covidqa, msmarco] + test -> ragbench_covidqa_msmarco_test
    """
    parts = [ingestor_name]

    # Get the first parameter value (the main dataset identifier)
    for _param_name, param_value in params.items():
        if param_value is None:
            continue
        if isinstance(param_value, list):
            parts.extend([v.lower().replace("-", "_") for v in param_value])
        elif isinstance(param_value, str):
            parts.append(param_value.lower().replace("-", "_"))
        break  # Only use the first non-None param for db name

    parts.append(subset)
    return "_".join(parts)


def create_ingest_command(ingestor_name: str, meta: "IngestorMeta"):  # noqa: C901
    """Create a Typer command function for an ingestor dynamically."""

    # Get the first param (main dataset identifier) for CLI option and available values
    main_param = meta.params[0] if meta.params else None
    cli_option = main_param.cli_option if main_param else "dataset"
    available_values = main_param.choices if main_param and main_param.choices else []

    def ingest_command(  # noqa: C901
        dataset_value: Annotated[
            str | None,
            typer.Option(f"--{cli_option}", help="Dataset to ingest. Use --list for available values."),
        ] = None,
        show_list: Annotated[bool, typer.Option("--list", help="Show all available dataset values and exit")] = False,
        # Common options
        subset: Annotated[str, typer.Option("--subset", help="Dataset split: train, dev, or test")] = "test",
        query_limit: Annotated[
            int | None, typer.Option("--query-limit", help="Maximum number of queries to ingest")
        ] = None,
        min_corpus_cnt: Annotated[
            int | None, typer.Option("--min-corpus-cnt", help="Minimum number of corpus documents to ingest")
        ] = None,
        db_name: Annotated[
            str | None, typer.Option("--db-name", help="Custom database schema name (auto-generated if not specified)")
        ] = None,
        # Embedding options
        embedding_model: Annotated[
            str, typer.Option("--embedding-model", help="Embedding model config name from configs/embedding/")
        ] = "openai-small",
        embed_batch_size: Annotated[int, typer.Option("--embed-batch-size", help="Batch size for embedding")] = 128,
        embed_concurrency: Annotated[
            int, typer.Option("--embed-concurrency", help="Max concurrent embedding calls")
        ] = 16,
        skip_embedding: Annotated[
            bool, typer.Option("--skip-embedding", help="Skip embedding step (ingest data only)")
        ] = False,
        # Database connection options
        db_host: Annotated[
            str | None, typer.Option("--db-host", help="Database host (default: from configs/db/default.yaml)")
        ] = None,
        db_port: Annotated[
            int | None, typer.Option("--db-port", help="Database port (default: from configs/db/default.yaml)")
        ] = None,
        db_user: Annotated[
            str | None, typer.Option("--db-user", help="Database user (default: from configs/db/default.yaml)")
        ] = None,
        db_password: Annotated[
            str | None, typer.Option("--db-password", help="Database password (or set PGPASSWORD)")
        ] = None,
        db_database: Annotated[
            str | None, typer.Option("--db-database", help="Database name (default: from configs/db/default.yaml)")
        ] = None,
        # Extra options for specific ingestors (from __init__ signature)
        batch_size: Annotated[
            int | None, typer.Option("--batch-size", help="Batch size for streaming ingestion")
        ] = None,
        score_threshold: Annotated[
            int | None, typer.Option("--score-threshold", help="Minimum relevance score threshold (0-2)")
        ] = None,
        include_instruction: Annotated[
            bool | None,
            typer.Option(
                "--include-instruction/--no-include-instruction",
                help="Include instruction prefix for InstructionRetrieval tasks",
            ),
        ] = None,
        document_mode: Annotated[
            str | None, typer.Option("--document-mode", help="Document mode: 'short' or 'long'")
        ] = None,
    ) -> None:
        # Handle --list flag
        if show_list:
            typer.echo(f"\nAvailable values for --{cli_option}:")
            for val in available_values:
                typer.echo(f"  {val}")
            return

        # Validate dataset value
        if not dataset_value:
            typer.echo(f"Error: --{cli_option} is required", err=True)
            typer.echo("Use --list to see available values", err=True)
            raise typer.Exit(1)

        # Handle list parameters (comma-separated) for ingestors with list[str] params
        processed_value: str | list[str] = dataset_value
        if main_param and main_param.is_list and dataset_value:
            processed_value = [v.strip() for v in str(dataset_value).split(",")]

        # Generate db_name
        params_for_name = {main_param.name: processed_value} if main_param else {}
        final_db_name = db_name or generate_db_name(ingestor_name, params_for_name, subset)

        # Load DB config from YAML first, then override with CLI options
        db_config = load_db_config_from_yaml()

        # Override with CLI options if provided
        if db_host is not None:
            db_config.host = db_host
        if db_port is not None:
            db_config.port = db_port
        if db_user is not None:
            db_config.user = db_user
        if db_password is not None:
            db_config.password = db_password
        if db_database is not None:
            db_config.database = db_database

        typer.echo(f"\nIngesting dataset: {ingestor_name}")
        typer.echo(f"  {cli_option}: {processed_value}")
        typer.echo(f"  subset: {subset}")
        typer.echo(f"  target schema: {final_db_name}")
        typer.echo(f"  database: {db_config.host}:{db_config.port}/{db_config.database}")
        typer.echo(f"  embedding model: {embedding_model}")
        typer.echo("=" * 60)

        # 1. Load embedding model from YAML config
        typer.echo(f"\nLoading embedding model: {embedding_model}")
        try:
            embed_model = load_embedding_model(embedding_model)
        except FileNotFoundError as e:
            typer.echo(f"Error: {e}", err=True)
            typer.echo("\nAvailable embedding configs:", err=True)
            for name in discover_embedding_configs():
                typer.echo(f"  {name}", err=True)
            raise typer.Exit(1) from None

        # 2. Health check embedding model and get dimension
        typer.echo("Checking embedding model health...")
        try:
            embedding_dim = health_check_embedding(embed_model)
            typer.echo(f"Embedding model healthy. Dimension: {embedding_dim}")
        except Exception as e:
            typer.echo(f"Error: Embedding health check failed: {e}", err=True)
            raise typer.Exit(1) from None

        # 3. Build ingestor constructor kwargs from CLI params
        init_kwargs = {main_param.name: processed_value} if main_param else {}
        if batch_size is not None:
            init_kwargs["batch_size"] = batch_size
        if score_threshold is not None:
            init_kwargs["score_threshold"] = score_threshold
        if include_instruction is not None:
            init_kwargs["include_instruction"] = include_instruction
        if document_mode is not None:
            init_kwargs["document_mode"] = document_mode

        # 4. Create ingestor with embedding model
        from autorag_research.data.base import MultiModalEmbeddingDataIngestor, TextEmbeddingDataIngestor

        ingestor_class = meta.ingestor_class
        if issubclass(ingestor_class, TextEmbeddingDataIngestor):
            ingestor = ingestor_class(embed_model, **init_kwargs)
        elif issubclass(ingestor_class, MultiModalEmbeddingDataIngestor):
            ingestor = ingestor_class(embedding_model=embed_model, **init_kwargs)
        else:
            typer.echo(f"Error: Unknown ingestor type: {ingestor_class}", err=True)
            raise typer.Exit(1)

        # 5. Detect primary key type from dataset
        typer.echo("\nDetecting primary key type from dataset...")
        detected_pkey_type: Literal["bigint", "string"] = ingestor.detect_primary_key_type()
        typer.echo(f"Detected primary key type: {detected_pkey_type}")

        # 6. Create database and schema
        typer.echo(f"\nCreating database schema: {final_db_name}")
        _create_database_and_schema(db_config, final_db_name, embedding_dim, detected_pkey_type)

        # 7. Create session factory and service
        session_factory, schema = _create_session_factory_and_schema(
            db_config, final_db_name, embedding_dim, detected_pkey_type
        )

        if issubclass(ingestor_class, TextEmbeddingDataIngestor):
            from autorag_research.orm.service.text_ingestion import TextDataIngestionService

            text_service = TextDataIngestionService(session_factory, schema)
            ingestor.set_service(text_service)
        else:
            from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService

            mm_service = MultiModalIngestionService(session_factory, schema)
            ingestor.set_service(mm_service)  # ty: ignore[invalid-argument-type]

        # 8. Ingest data
        typer.echo(f"\nIngesting {ingestor_name} dataset...")
        subset_literal: Literal["train", "dev", "test"] = subset  # ty: ignore[invalid-assignment]
        ingestor.ingest(
            subset=subset_literal,
            query_limit=query_limit,
            min_corpus_cnt=min_corpus_cnt,
        )
        typer.echo("Ingestion complete.")

        # 9. Embed data (unless skipped)
        if not skip_embedding:
            typer.echo(f"\nEmbedding all data (batch_size={embed_batch_size}, concurrency={embed_concurrency})...")
            ingestor.embed_all(max_concurrency=embed_concurrency, batch_size=embed_batch_size)
            typer.echo("Embedding complete.")
        else:
            typer.echo("\nSkipping embedding step (--skip-embedding)")

        # 10. Print summary
        typer.echo("\n" + "=" * 60)
        typer.echo("Ingestion Summary:")
        typer.echo(f"  Schema: {final_db_name}")
        typer.echo(f"  Embedding dimension: {embedding_dim}")
        typer.echo(f"  Primary key type: {detected_pkey_type}")
        typer.echo("\nNext steps:")
        typer.echo(f"  autorag-research list --schema={final_db_name}")
        typer.echo(f"  autorag-research run --db-name={final_db_name}")

    # Set docstring dynamically
    example_value = available_values[0] if available_values else "value"
    ingest_command.__doc__ = f"""{meta.description}

    Available values for --{cli_option}: {", ".join(available_values[:5])}{"..." if len(available_values) > 5 else ""}
    Use --list to see all available values.

    Examples:
      autorag-research ingest {ingestor_name} --{cli_option}={example_value}
      autorag-research ingest {ingestor_name} --{cli_option}={example_value} --embedding-model=openai-large
      autorag-research ingest {ingestor_name} --list
    """

    return ingest_command


def _create_database_and_schema(
    db_config: DatabaseConfig,
    schema_name: str,
    embedding_dim: int,
    primary_key_type: Literal["bigint", "string"],
) -> None:
    """Create database, install extensions, and create schema tables."""
    from autorag_research.orm.util import create_database, install_vector_extensions

    create_database(
        db_config.host,
        db_config.user,
        db_config.password,
        schema_name,
        port=db_config.port,
    )
    install_vector_extensions(
        db_config.host,
        db_config.user,
        db_config.password,
        schema_name,
        port=db_config.port,
    )


def _create_session_factory_and_schema(
    db_config: DatabaseConfig,
    schema_name: str,
    embedding_dim: int,
    primary_key_type: Literal["bigint", "string"],
):
    """Create SQLAlchemy session factory and ORM schema."""
    from autorag_research.orm.schema_factory import create_schema

    schema = create_schema(embedding_dim, primary_key_type)
    db_url = (
        f"postgresql+psycopg://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{schema_name}"
    )
    engine = create_engine(db_url, pool_pre_ping=True)
    schema.Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    return session_factory, schema


@ingest_app.callback(invoke_without_command=True)
def ingest_callback(ctx: typer.Context) -> None:
    """Ingest datasets into PostgreSQL.

    Choose an ingestor and specify the dataset to ingest.

    Examples:
      autorag-research ingest beir --dataset-name=scifact
      autorag-research ingest beir --dataset-name=scifact --embedding-model=openai-large
      autorag-research ingest mrtydi --language=english --query-limit=100
      autorag-research ingest ragbench --configs=covidqa,msmarco
      autorag-research ingest beir --dataset-name=scifact --skip-embedding
    """
    if ctx.invoked_subcommand is None:
        from autorag_research.data.registry import get_ingestor_help

        typer.echo(ctx.get_help())
        typer.echo("\n" + get_ingestor_help())

        # Show available embedding configs
        typer.echo("\nAvailable embedding models (--embedding-model):")
        for name, target in discover_embedding_configs().items():
            typer.echo(f"  {name}: {target}")
