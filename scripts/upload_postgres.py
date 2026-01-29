"""Upload PostgreSQL dump files to HuggingFace Hub.

Usage:
    python scripts/upload_postgres.py \
        --file-path ./scifact_openai-small.dump \
        --ingestor beir \
        --filename scifact_openai-small

Requires HF_TOKEN environment variable with write access to the target repository.
"""

import click

from autorag_research.data import upload_dump
from autorag_research.data.registry import discover_ingestors


def _get_ingestors_with_hf_repo() -> list[str]:
    """Get list of ingestor names that have hf_repo configured."""
    registry = discover_ingestors()
    return sorted(name for name, meta in registry.items() if meta.hf_repo is not None)


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
    type=click.Choice(_get_ingestors_with_hf_repo()),
    help="Ingestor family name (determines target repository).",
)
@click.option(
    "--filename",
    required=True,
    type=str,
    help="Dump filename without .dump extension (e.g., 'scifact_openai-small').",
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
    filename: str,
    message: str | None,
) -> None:
    """Upload a PostgreSQL dump file to HuggingFace Hub."""
    result = upload_dump(
        file_path=file_path,
        ingestor=ingestor,
        filename=filename,
        commit_message=message,
    )
    click.echo(f"Upload successful: {result}")


if __name__ == "__main__":
    main()
