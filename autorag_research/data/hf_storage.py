"""HuggingFace Hub storage integration for PostgreSQL dump files.

This module provides functions to upload, download, and manage PostgreSQL
dump files stored in HuggingFace Hub dataset repositories.
"""

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, repo_exists, upload_file

logger = logging.getLogger("AutoRAG-Research")

HF_ORG = "NomaDamas"

INGESTOR_TO_REPO: dict[str, str] = {
    "beir": "beir-dumps",
    "mrtydi": "mrtydi-dumps",
    "ragbench": "ragbench-dumps",
    "bright": "bright-dumps",
    "visrag": "visrag-dumps",
    "vidore": "vidore-dumps",
    "vidorev2": "vidorev2-dumps",
    "open-ragbench": "openragbench-dumps",
    "mteb": "mteb-dumps",
    "arxivqa": "arxivqa-dumps",
}


def get_repo_id(ingestor: str) -> str:
    """Get the HuggingFace Hub repository ID for an ingestor.

    Args:
        ingestor: Name of the ingestor (e.g., "beir", "mrtydi")

    Returns:
        Full repository ID (e.g., "NomaDamas/beir-dumps")

    Raises:
        KeyError: If ingestor is not recognized
    """
    if ingestor not in INGESTOR_TO_REPO:
        available = ", ".join(sorted(INGESTOR_TO_REPO.keys()))
        raise KeyError(f"Unknown ingestor '{ingestor}'. Available: {available}")  # noqa: TRY003
    return f"{HF_ORG}/{INGESTOR_TO_REPO[ingestor]}"


def make_dump_filename(dataset: str, embedding: str) -> str:
    """Generate a standardized dump filename.

    Args:
        dataset: Dataset subset name (e.g., "scifact", "arxivqa")
        embedding: Embedding model name (e.g., "openai-small", "colpali-v1.2")

    Returns:
        Filename in format "{dataset}_{embedding}.dump"
    """
    return f"{dataset}_{embedding}.dump"


def download_dump(
    ingestor: str,
    dataset: str,
    embedding: str,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download a PostgreSQL dump file from HuggingFace Hub.

    Uses HuggingFace's caching mechanism to avoid re-downloading files.

    Args:
        ingestor: Ingestor family name (e.g., "beir", "mrtydi")
        dataset: Dataset subset name (e.g., "scifact")
        embedding: Embedding model name (e.g., "openai-small")
        revision: Git revision (branch, tag, or commit hash). Defaults to main.
        cache_dir: Optional custom cache directory. Defaults to HF cache.

    Returns:
        Path to the downloaded dump file (in HF cache)

    Raises:
        KeyError: If ingestor is not recognized
        huggingface_hub.utils.EntryNotFoundError: If file doesn't exist
        huggingface_hub.utils.RepositoryNotFoundError: If repo doesn't exist
    """
    repo_id = get_repo_id(ingestor)
    filename = make_dump_filename(dataset, embedding)

    logger.info(f"Downloading dump from HuggingFace Hub: {repo_id}/{filename}")

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    logger.info(f"Downloaded to: {downloaded_path}")
    return Path(downloaded_path)


def upload_dump(
    file_path: str | Path,
    ingestor: str,
    dataset: str,
    embedding: str,
    commit_message: str | None = None,
) -> str:
    """Upload a PostgreSQL dump file to HuggingFace Hub.

    Requires HF_TOKEN environment variable to be set with write access.

    Args:
        file_path: Path to the local dump file
        ingestor: Ingestor family name (e.g., "beir", "mrtydi")
        dataset: Dataset subset name (e.g., "scifact")
        embedding: Embedding model name (e.g., "openai-small")
        commit_message: Optional commit message. Auto-generated if not provided.

    Returns:
        URL of the uploaded file

    Raises:
        KeyError: If ingestor is not recognized
        FileNotFoundError: If file_path doesn't exist
        huggingface_hub.utils.HfHubHTTPError: If upload fails
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dump file not found: {file_path}")  # noqa: TRY003

    repo_id = get_repo_id(ingestor)
    filename = make_dump_filename(dataset, embedding)

    if commit_message is None:
        commit_message = f"Add {dataset} dump with {embedding} embeddings"

    logger.info(f"Uploading dump to HuggingFace Hub: {repo_id}/{filename}")

    result = upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )

    logger.info(f"Uploaded successfully: {result}")
    return result


def list_available_dumps(ingestor: str) -> list[str]:
    """List all dump files available in an ingestor's repository.

    Args:
        ingestor: Ingestor family name (e.g., "beir", "mrtydi")

    Returns:
        List of filenames (e.g., ["scifact_openai-small.dump", "nfcorpus_openai-small.dump"])

    Raises:
        KeyError: If ingestor is not recognized
        huggingface_hub.utils.RepositoryNotFoundError: If repo doesn't exist
    """
    repo_id = get_repo_id(ingestor)

    logger.debug(f"Listing files in HuggingFace Hub repo: {repo_id}")

    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    return [f for f in files if f.endswith(".dump")]


def dump_exists(ingestor: str, dataset: str, embedding: str) -> bool:
    """Check if a specific dump file exists in HuggingFace Hub.

    Args:
        ingestor: Ingestor family name (e.g., "beir", "mrtydi")
        dataset: Dataset subset name (e.g., "scifact")
        embedding: Embedding model name (e.g., "openai-small")

    Returns:
        True if the dump exists, False otherwise

    Raises:
        KeyError: If ingestor is not recognized
    """
    repo_id = get_repo_id(ingestor)

    # First check if repo exists
    if not repo_exists(repo_id=repo_id, repo_type="dataset"):
        return False

    filename = make_dump_filename(dataset, embedding)

    try:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        return False
    else:
        return filename in files
