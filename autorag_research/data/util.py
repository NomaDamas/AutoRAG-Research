"""Dataset setup utilities for AutoRAG-Research.

Provides functions to download pre-built dataset dumps from HuggingFace Hub
and restore them to PostgreSQL databases.
"""

import logging

from autorag_research.data.hf_storage import download_dump
from autorag_research.data.registry import discover_ingestors
from autorag_research.data.restore import restore_database

logger = logging.getLogger("AutoRAG-Research")


def _get_ingestors_with_hf_repo() -> dict[str, str]:
    """Get all ingestors that have hf_repo configured.

    Returns:
        Dict mapping ingestor name to hf_repo value.
    """
    registry = discover_ingestors()
    return {name: meta.hf_repo for name, meta in registry.items() if meta.hf_repo is not None}


def setup_dataset(
    ingestor_name: str,
    dataset_name: str,
    embedding_model_name: str,
    host: str,
    user: str,
    password: str,
    port: int = 5432,
    **kwargs,
) -> None:
    """Set up a dataset by downloading and restoring it to a PostgreSQL database.

    Downloads the pre-built dataset dump file from HuggingFace Hub and restores it
    to a PostgreSQL database with the naming convention "{dataset_name}_{embedding_model_name}".

    Args:
        ingestor_name: Name of the ingestor family (e.g., "beir", "mrtydi").
        dataset_name: Name of the dataset to set up (e.g., "scifact").
        embedding_model_name: Name of the embedding model used for the dataset.
        host: PostgreSQL server hostname.
        user: PostgreSQL username.
        password: PostgreSQL password.
        port: PostgreSQL server port. Defaults to 5432.
        **kwargs: Additional keyword arguments passed to restore_database.

    Raises:
        ValueError: If the ingestor_name is not recognized or has no HF repo configured.
    """
    ingestors_with_hf_repo = _get_ingestors_with_hf_repo()
    if ingestor_name not in ingestors_with_hf_repo:
        msg = f"Unknown ingestor or no HF repo configured: '{ingestor_name}'. Available: {', '.join(sorted(ingestors_with_hf_repo.keys()))}"
        raise ValueError(msg)

    logger.info(f"Setting up dataset: {ingestor_name}/{dataset_name} with {embedding_model_name}")

    # Download dump from HuggingFace Hub (uses HF caching)
    filename = f"{dataset_name}_{embedding_model_name}"
    dump_path = download_dump(
        ingestor=ingestor_name,
        filename=filename,
    )

    # Restore to PostgreSQL
    database_name = f"{dataset_name}_{embedding_model_name}"
    logger.info(f"Restoring to database: {database_name}")

    restore_database(
        dump_file=dump_path,
        host=host,
        user=user,
        password=password,
        database=database_name,
        port=port,
        **kwargs,
    )

    logger.info(f"Dataset setup complete: {database_name}")
