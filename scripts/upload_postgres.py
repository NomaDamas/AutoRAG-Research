"""Upload PostgreSQL dump files to HuggingFace Hub.

Usage:
    python scripts/upload_postgres.py \\
        --file-path ./scifact.dump \\
        --ingestor beir \\
        --dataset scifact \\
        --embedding-model openai-small

Requires HF_TOKEN environment variable with write access to the target repository.
"""

import click

from autorag_research.data import upload_dump
from autorag_research.data.hf_storage import INGESTOR_TO_REPO


@click.command()
@click.option(
    "--file-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the PostgreSQL dump file to upload.",
)
@click.option(
    "--ingestor",
    required=True,
    type=click.Choice(list(INGESTOR_TO_REPO.keys())),
    help="Ingestor family name (determines target repository).",
)
@click.option(
    "--dataset",
    required=True,
    type=str,
    help="Dataset subset name (e.g., 'scifact', 'arxivqa').",
)
@click.option(
    "--embedding-model",
    required=True,
    type=str,
    help="Embedding model name (e.g., 'openai-small', 'colpali-v1.2').",
)
@click.option(
    "--message",
    type=str,
    default=None,
    help="Custom commit message. Auto-generated if not provided.",
)
def main(
    file_path: str,
    ingestor: str,
    dataset: str,
    embedding_model: str,
    message: str | None,
) -> None:
    """Upload a PostgreSQL dump file to HuggingFace Hub."""
    result = upload_dump(
        file_path=file_path,
        ingestor=ingestor,
        dataset=dataset,
        embedding=embedding_model,
        commit_message=message,
    )
    click.echo(f"Upload successful: {result}")


if __name__ == "__main__":
    main()
