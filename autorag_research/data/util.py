"""Dataset setup utilities for AutoRAG-Research.

Provides functions to download pre-built dataset dumps from HuggingFace Hub
and restore them to PostgreSQL databases.
"""

import logging

from autorag_research.data.hf_storage import INGESTOR_TO_REPO, download_dump
from autorag_research.data.restore import restore_database

logger = logging.getLogger("AutoRAG-Research")

# Registry mapping: ingestor -> dataset -> embedding -> filename
# This registry tracks available pre-built dumps
DATASET_REGISTRY: dict[str, dict[str, dict[str, str]]] = {
    "beir": {
        "scifact": {"embeddinggemma-300m": "scifact_embeddinggemma-300m.dump"},
    },
}


def _find_ingestor_for_dataset(dataset_name: str) -> str | None:
    """Find the ingestor that contains a given dataset.

    Args:
        dataset_name: Name of the dataset to find

    Returns:
        Ingestor name if found, None otherwise
    """
    for ingestor, datasets in DATASET_REGISTRY.items():
        if dataset_name in datasets:
            return ingestor
    return None


def setup_dataset(
    dataset_name: str,
    embedding_model_name: str,
    host: str,
    user: str,
    password: str,
    port: int = 5432,
    ingestor_name: str | None = None,
    **kwargs,
) -> None:
    """Set up a dataset by downloading and restoring it to a PostgreSQL database.

    Downloads the pre-built dataset dump file from HuggingFace Hub and restores it
    to a PostgreSQL database with the naming convention "{dataset_name}_{embedding_model_name}".

    Args:
        dataset_name: Name of the dataset to set up (e.g., "scifact").
        embedding_model_name: Name of the embedding model used for the dataset.
        host: PostgreSQL server hostname.
        user: PostgreSQL username.
        password: PostgreSQL password.
        port: PostgreSQL server port. Defaults to 5432.
        ingestor_name: Name of the ingestor family (e.g., "beir", "mrtydi").
            If not provided, will attempt to auto-detect from DATASET_REGISTRY.
        **kwargs: Additional keyword arguments passed to restore_database.

    Raises:
        KeyError: If the dataset cannot be found and ingestor_name is not provided.
        ValueError: If the ingestor_name is not recognized.
    """
    # Determine ingestor
    if ingestor_name is None:
        ingestor_name = _find_ingestor_for_dataset(dataset_name)
        if ingestor_name is None:
            available_datasets = [f"{ing}/{ds}" for ing, datasets in DATASET_REGISTRY.items() for ds in datasets]
            msg = f"Dataset '{dataset_name}' not found. Use one of: {', '.join(available_datasets)}"
            raise KeyError(msg)

    if ingestor_name not in INGESTOR_TO_REPO:
        msg = f"Unknown ingestor '{ingestor_name}'. Available: {', '.join(sorted(INGESTOR_TO_REPO.keys()))}"
        raise ValueError(msg)

    logger.info(f"Setting up dataset: {ingestor_name}/{dataset_name} with {embedding_model_name}")

    # Download dump from HuggingFace Hub (uses HF caching)
    dump_path = download_dump(
        ingestor=ingestor_name,
        dataset=dataset_name,
        embedding=embedding_model_name,
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
