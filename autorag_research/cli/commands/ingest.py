"""ingest command - Ingest datasets into PostgreSQL using Click CLI."""

import logging
import os
import sys
import tempfile
from pathlib import Path

import click
import yaml
from huggingface_hub import hf_hub_download

from autorag_research.cli.configs.db import DatabaseConfig
from autorag_research.cli.configs.ingestors import (
    COMMON_OPTIONS,
    INGESTOR_REGISTRY,
    generate_db_name,
    get_ingestor_help,
)

logger = logging.getLogger(__name__)

HF_REPO_ID = "vkehfdl1/autorag-research-datasets"


def load_db_config_from_yaml() -> DatabaseConfig:
    """Load database config from configs/db/default.yaml if exists."""
    from autorag_research.cli.config_path import ConfigPathManager

    config_dir = ConfigPathManager.get_config_dir() if ConfigPathManager.is_initialized() else Path.cwd() / "configs"
    yaml_path = config_dir / "db" / "default.yaml"

    defaults = DatabaseConfig()

    if not yaml_path.exists():
        return defaults

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}

        # Handle OmegaConf-style env var: ${oc.env:PGPASSWORD,postgres}
        password = data.get("password", defaults.password)
        if isinstance(password, str) and password.startswith("${"):
            password = os.environ.get("PGPASSWORD", "postgres")

        return DatabaseConfig(
            host=data.get("host", defaults.host),
            port=data.get("port", defaults.port),
            user=data.get("user", defaults.user),
            password=password,
            database=data.get("database", defaults.database),
        )
    except Exception as e:
        logger.warning(f"Failed to load DB config from YAML: {e}")
        return defaults


class IngestorGroup(click.MultiCommand):  # ty: ignore[invalid-base]
    """Dynamic Click group that creates subcommands for each ingestor."""

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:  # noqa: C901
        if cmd_name not in INGESTOR_REGISTRY:
            return None

        spec = INGESTOR_REGISTRY[cmd_name]

        # Build options dynamically
        params = []

        # Main dataset selection option (not required if --list is used)
        params.append(
            click.Option(
                [f"--{spec.cli_option}"],
                required=False,  # We'll validate manually to allow --list without --dataset
                help=f"Dataset to ingest. Available: {', '.join(spec.available_values[:5])}... (use --list for full list)",
            )
        )

        # Add --list option to show available values
        params.append(
            click.Option(
                ["--list", "show_list"],
                is_flag=True,
                help="Show all available dataset values and exit",
            )
        )

        # Common options
        for opt in COMMON_OPTIONS:
            params.append(
                click.Option(
                    [f"--{opt.name}"],
                    type=opt.type if not isinstance(opt.type, bool) else None,
                    is_flag=isinstance(opt.type, bool),
                    default=opt.default,
                    help=opt.help,
                )
            )

        # Extra options specific to this ingestor
        for opt in spec.extra_options:
            params.append(
                click.Option(
                    [f"--{opt.name}"],
                    type=opt.type if not isinstance(opt.type, bool) else None,
                    is_flag=isinstance(opt.type, bool),
                    default=opt.default,
                    help=opt.help,
                )
            )

        # Database connection options (defaults from configs/db/default.yaml, CLI overrides)
        params.extend([
            click.Option(["--db-host"], default=None, help="Database host (default: from configs/db/default.yaml)"),
            click.Option(
                ["--db-port"], type=int, default=None, help="Database port (default: from configs/db/default.yaml)"
            ),
            click.Option(["--db-user"], default=None, help="Database user (default: from configs/db/default.yaml)"),
            click.Option(["--db-password"], default=None, help="Database password (or set PGPASSWORD)"),
            click.Option(["--db-database"], default=None, help="Database name (default: from configs/db/default.yaml)"),
        ])

        def make_callback(ingestor_name: str, ingestor_spec):  # noqa: C901
            def callback(**kwargs):  # noqa: C901
                # Handle --list flag
                if kwargs.get("show_list"):
                    click.echo(f"\nAvailable values for --{ingestor_spec.cli_option}:")
                    for val in ingestor_spec.available_values:
                        click.echo(f"  {val}")
                    return

                # Extract parameters
                dataset_value = kwargs.get(ingestor_spec.cli_option.replace("-", "_"))
                if not dataset_value:
                    click.echo(f"Error: --{ingestor_spec.cli_option} is required", err=True)
                    click.echo("Use --list to see available values", err=True)
                    sys.exit(1)

                subset = kwargs.get("subset", "test")
                custom_db_name = kwargs.get("db_name")

                # Handle list parameters (comma-separated)
                if ingestor_name in ("ragbench", "bright") and dataset_value:
                    dataset_value = [v.strip() for v in str(dataset_value).split(",")]

                # Generate db_name
                params_for_name = {ingestor_spec.dataset_param: dataset_value}
                db_name = custom_db_name or generate_db_name(ingestor_name, params_for_name, subset)

                # Load DB config from YAML first, then override with CLI options
                db_config = load_db_config_from_yaml()

                # Override with CLI options if provided
                if kwargs.get("db_host") is not None:
                    db_config.host = kwargs["db_host"]
                if kwargs.get("db_port") is not None:
                    db_config.port = kwargs["db_port"]
                if kwargs.get("db_user") is not None:
                    db_config.user = kwargs["db_user"]
                if kwargs.get("db_password") is not None:
                    db_config.password = kwargs["db_password"]
                if kwargs.get("db_database") is not None:
                    db_config.database = kwargs["db_database"]

                click.echo(f"\nIngesting dataset: {ingestor_name}")
                click.echo(f"  {ingestor_spec.cli_option}: {dataset_value}")
                click.echo(f"  subset: {subset}")
                click.echo(f"  target schema: {db_name}")
                click.echo(f"  database: {db_config.host}:{db_config.port}/{db_config.database}")
                click.echo("=" * 60)

                # Try to download pre-built dump first
                dump_file = download_dump(db_name)

                if dump_file:
                    restore_from_dump(db_config, dump_file, db_name)
                else:
                    click.echo("\nNo pre-built dump available.")
                    click.echo("Consider using pre-built dumps from HuggingFace Hub:")
                    click.echo(f"  https://huggingface.co/datasets/{HF_REPO_ID}")

            return callback

        return click.Command(
            name=cmd_name,
            params=params,
            callback=make_callback(cmd_name, spec),
            help=spec.description,
        )


@click.command(cls=IngestorGroup, invoke_without_command=True)
@click.pass_context
def ingest(ctx: click.Context) -> None:
    """Ingest datasets into PostgreSQL.

    Choose an ingestor and specify the dataset to ingest.

    \b
    Examples:
      autorag-research ingest beir --dataset=scifact
      autorag-research ingest mrtydi --language=english --query-limit=100
      autorag-research ingest ragbench --configs=covidqa,msmarco
      autorag-research ingest beir --dataset=scifact --db-name=my_custom_schema
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        click.echo("\n" + get_ingestor_help())


def download_dump(schema_name: str) -> Path | None:
    """Download pre-built pg_dump from HuggingFace Hub."""
    dump_filename = f"{schema_name}.dump"

    click.echo(f"\nLooking for pre-built dump: {dump_filename}")

    try:
        dump_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=dump_filename,
            repo_type="dataset",
            cache_dir=tempfile.gettempdir(),
        )
        click.echo(f"  Downloaded: {dump_path}")
        return Path(dump_path)
    except Exception as e:
        logger.debug(f"Could not download pre-built dump: {e}")
        click.echo("  Not found in HuggingFace Hub")
        return None


def restore_from_dump(db_config: DatabaseConfig, dump_file: Path, schema_name: str) -> None:
    """Restore database from pg_dump file."""
    from autorag_research.data.restore import restore_database

    click.echo(f"\nRestoring from dump file: {dump_file}")

    try:
        restore_database(
            dump_file=str(dump_file),
            host=db_config.host,
            user=db_config.user,
            password=db_config.password,
            database=db_config.database,
            port=db_config.port,
            clean=False,
            create=True,
            no_owner=True,
            install_extensions=True,
        )
        click.echo(f"\nSuccess! Schema '{schema_name}' restored.")
        click.echo("\nNext steps:")
        click.echo(f"  autorag-research info --schema={schema_name}")
        click.echo(f"  autorag-research run --db-name={schema_name}")
    except Exception as e:
        logger.exception("Failed to restore database")
        click.echo(f"\nError restoring database: {e}", err=True)
        click.echo("\nTroubleshooting:", err=True)
        click.echo("  1. Ensure PostgreSQL is running", err=True)
        click.echo("  2. Check database credentials", err=True)
        click.echo("  3. Ensure pg_restore is installed", err=True)
        sys.exit(1)


if __name__ == "__main__":
    ingest()
