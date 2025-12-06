"""Data ingestion script for AutoRAG-Research."""

import os
import subprocess
from pathlib import Path
from typing import Literal

import click
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from autorag_research.data.vidore import ViDoReArxivQAIngestor
from autorag_research.orm.schema_factory import create_schema
from autorag_research.orm.service import MultiModalIngestionService
from autorag_research.orm.util import create_database, install_vector_extensions

SCRIPT_DIR = Path(__file__).parent
DUMP_SCRIPT = SCRIPT_DIR / "dump_postgres.sh"

DATASET_TYPES = {
    "vidore-arxivqa": ViDoReArxivQAIngestor,
}


def create_session_factory_and_schema(
    db_name: str, embedding_dim: int, primary_key_type: Literal["bigint", "string"] = "bigint"
):
    """Create session factory and schema for a given embedding dimension and primary key type.

    Args:
        db_name: Name of the database
        embedding_dim: Dimension of embeddings
        primary_key_type: Type of primary keys ('bigint' or 'string')

    Returns:
        Tuple of (session_factory, schema)
    """
    host = os.getenv("POSTGRES_HOST", "localhost")
    user = os.getenv("POSTGRES_USER", "postgres")
    pwd = os.getenv("POSTGRES_PASSWORD", "postgres")
    port = int(os.getenv("POSTGRES_PORT", "5432"))

    schema = create_schema(embedding_dim, primary_key_type)

    create_database(host, user, pwd, db_name, port=port)
    install_vector_extensions(host, user, pwd, db_name, port=port)
    postgres_url = f"postgresql+psycopg://{user}:{pwd}@{host}:{port}/{db_name}"
    engine = create_engine(postgres_url, pool_pre_ping=True)
    schema.Base.metadata.create_all(engine)
    return sessionmaker(bind=engine), schema


def dump_database(db_name: str, output_path: str) -> None:
    host = os.getenv("POSTGRES_HOST", "localhost")
    user = os.getenv("POSTGRES_USER", "postgres")
    pwd = os.getenv("POSTGRES_PASSWORD", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")

    cmd = [
        str(DUMP_SCRIPT),
        "--host",
        host,
        "--port",
        port,
        "--user",
        user,
        "--password",
        pwd,
        "--dbname",
        db_name,
        "--output",
        output_path,
    ]
    subprocess.run(cmd, check=True)  # noqa: S603


@click.command()
@click.option("--dataset-type", type=click.Choice(list(DATASET_TYPES.keys())), required=True, help="Type of dataset")
@click.option("--subset", type=click.Choice(["train", "dev", "test"]), default="test", help="Dataset subset")
@click.option("--dump-path", type=str, default=None, help="Path to dump the database after ingestion")
@click.option(
    "--embedding-dim",
    type=int,
    default=768,
    help="Pre-made embedding dimension to set up the database schema without actual embeddings",
)
def main(
    dataset_type: str,
    subset: Literal["train", "dev", "test"],
    dump_path: str | None,
    embedding_dim: int = 768,
):
    # Create ingestor first to detect primary key type
    ingestor_class = DATASET_TYPES[dataset_type]
    ingestor = ingestor_class()

    # Auto-detect primary key type from the ingestor
    click.echo("Auto-detecting primary key type from dataset...")
    detected_pkey_type = ingestor.detect_primary_key_type()
    click.echo(f"Detected primary key type: {detected_pkey_type}")

    db_name = f"{dataset_type}_no-embed"
    session_factory, schema = create_session_factory_and_schema(db_name, embedding_dim, detected_pkey_type)
    service = MultiModalIngestionService(session_factory, schema)

    # Set service to the ingestor
    ingestor.set_service(service)

    click.echo(f"Ingesting {dataset_type} ({subset})...")
    ingestor.ingest(subset)
    click.echo("Ingestion complete.")

    if dump_path:
        click.echo(f"Dumping database to {dump_path}...")
        dump_database(db_name, dump_path)
        click.echo("Database dump complete.")


if __name__ == "__main__":
    main()
