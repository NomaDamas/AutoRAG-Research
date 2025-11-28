"""Data ingestion script for AutoRAG-Research."""

import os
import subprocess
from pathlib import Path
from typing import Literal

import click
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from autorag_research.data.beir import BEIRIngestor
from autorag_research.exceptions import EmbeddingNotSetError
from autorag_research.orm.schema_factory import create_schema
from autorag_research.orm.service.text_ingestion import TextDataIngestionService
from autorag_research.orm.util import create_database, install_vector_extensions

SCRIPT_DIR = Path(__file__).parent
DUMP_SCRIPT = SCRIPT_DIR / "dump_postgres.sh"

DATASET_TYPES = {
    "beir": BEIRIngestor,
}


def create_session_factory(db_name: str, embedding_dim: int):
    host = os.getenv("POSTGRES_HOST", "localhost")
    user = os.getenv("POSTGRES_USER", "postgres")
    pwd = os.getenv("POSTGRES_PASSWORD", "postgres")
    port = int(os.getenv("POSTGRES_PORT", "5432"))

    schema = create_schema(embedding_dim)

    create_database(host, user, pwd, db_name, port=port)
    install_vector_extensions(host, user, pwd, db_name, port=port)
    postgres_url = f"postgresql+psycopg://{user}:{pwd}@{host}:{port}/{db_name}"
    engine = create_engine(postgres_url, pool_pre_ping=True)
    schema.Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def health_check_embedding(embedding_model: OpenAILikeEmbedding) -> int:
    """Health check embedding model and return embedding dimension."""
    try:
        embedding = embedding_model.get_text_embedding("health check")
        return len(embedding)
    except Exception as e:
        raise EmbeddingNotSetError from e


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
@click.option("--dataset-name", type=str, required=True, help="Name of the dataset")
@click.option("--embedding-model-name", type=str, required=True, help="Name of the Embedding Model")
@click.option("--subset", type=click.Choice(["train", "dev", "test"]), default="test", help="Dataset subset")
@click.option("--batch-size", type=int, default=128, help="Batch size for embedding")
@click.option("--max-concurrency", type=int, default=16, help="Max concurrent embedding calls")
@click.option("--api-base", type=str, required=True, help="Embedding API base URL")
@click.option("--dump-path", type=str, default=None, help="Path to dump the database after ingestion")
def main(
    dataset_type: str,
    dataset_name: str,
    embedding_model_name: str,
    subset: Literal["train", "dev", "test"],
    batch_size: int,
    max_concurrency: int,
    api_base: str,
    dump_path: str | None,
):
    embedding_model = OpenAILikeEmbedding(
        api_base=api_base,
        api_key="-",
        model_name=embedding_model_name,
    )

    click.echo("Checking embedding model health...")
    embedding_dim = health_check_embedding(embedding_model)
    click.echo(f"Embedding model is healthy. Dimension: {embedding_dim}")

    db_name = f"{dataset_name}_{embedding_model_name}"
    session_factory = create_session_factory(db_name, embedding_dim)
    service = TextDataIngestionService(session_factory)

    ingestor_class = DATASET_TYPES[dataset_type]
    ingestor = ingestor_class(service, embedding_model, dataset_name)

    click.echo(f"Ingesting {dataset_type}/{dataset_name} ({subset})...")
    ingestor.ingest(subset)
    click.echo("Ingestion complete.")

    click.echo("Embedding all data...")
    ingestor.embed_all(max_concurrency=max_concurrency, batch_size=batch_size)
    click.echo("Embedding complete.")

    if dump_path:
        click.echo(f"Dumping database to {dump_path}...")
        dump_database(db_name, dump_path)
        click.echo("Database dump complete.")


if __name__ == "__main__":
    main()
