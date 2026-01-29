"""HuggingFace Hub storage integration for PostgreSQL dump files.

This module provides functions to upload, download, and manage PostgreSQL
dump files stored in HuggingFace Hub dataset repositories.
"""

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, repo_exists, upload_file

logger = logging.getLogger("AutoRAG-Research")

HF_ORG = "NomaDamas"


def get_repo_id(ingestor: str) -> str:
    """Get the HuggingFace Hub repository ID for an ingestor.

    Args:
        ingestor: Name of the ingestor (e.g., "beir", "mrtydi")

    Returns:
        Full repository ID (e.g., "NomaDamas/beir-dumps")

    Raises:
        KeyError: If ingestor is not recognized or has no HF repo configured
    """
    from autorag_research.data.registry import discover_ingestors

    registry = discover_ingestors()
    meta = registry.get(ingestor)
    if meta is None or meta.hf_repo is None:
        available = sorted(name for name, m in registry.items() if m.hf_repo is not None)
        raise KeyError(f"Unknown ingestor or no HF repo configured: '{ingestor}'. Available: {', '.join(available)}")  # noqa: TRY003
    return f"{HF_ORG}/{meta.hf_repo}"


def download_dump(
    ingestor: str,
    filename: str,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download a PostgreSQL dump file from HuggingFace Hub.

    Uses HuggingFace's caching mechanism to avoid re-downloading files.

    Args:
        ingestor: Ingestor family name (e.g., "beir", "mrtydi")
        filename: Dump filename without .dump extension (e.g., "scifact_openai-small").
            Use list_available_dumps() to see available files.
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
    full_filename = f"{filename}.dump"

    logger.info(f"Downloading dump from HuggingFace Hub: {repo_id}/{full_filename}")

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=full_filename,
        revision=revision,
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    logger.info(f"Downloaded to: {downloaded_path}")
    return Path(downloaded_path)


def upload_dump(
    file_path: str | Path,
    ingestor: str,
    filename: str,
    commit_message: str | None = None,
) -> str:
    """Upload a PostgreSQL dump file to HuggingFace Hub.

    Requires HF_TOKEN environment variable to be set with write access.

    Args:
        file_path: Path to the local dump file
        ingestor: Ingestor family name (e.g., "beir", "mrtydi")
        filename: Dump filename without .dump extension (e.g., "scifact_openai-small").
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
    full_filename = f"{filename}.dump"

    if commit_message is None:
        commit_message = f"Add {filename} dump"

    logger.info(f"Uploading dump to HuggingFace Hub: {repo_id}/{full_filename}")

    result = upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=full_filename,
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
        List of filenames without .dump extension (e.g., ["scifact_openai-small", "nfcorpus_openai-small"]).
        These can be passed directly to download_dump().

    Raises:
        KeyError: If ingestor is not recognized
        huggingface_hub.utils.RepositoryNotFoundError: If repo doesn't exist
    """
    repo_id = get_repo_id(ingestor)

    logger.debug(f"Listing files in HuggingFace Hub repo: {repo_id}")

    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    return [f.removesuffix(".dump") for f in files if f.endswith(".dump")]


def dump_exists(ingestor: str, filename: str) -> bool:
    """Check if a specific dump file exists in HuggingFace Hub.

    Args:
        ingestor: Ingestor family name (e.g., "beir", "mrtydi")
        filename: Dump filename without .dump extension (e.g., "scifact_openai-small").

    Returns:
        True if the dump exists, False otherwise

    Raises:
        KeyError: If ingestor is not recognized
    """
    repo_id = get_repo_id(ingestor)

    # First check if repo exists
    if not repo_exists(repo_id=repo_id, repo_type="dataset"):
        return False

    full_filename = f"{filename}.dump"

    try:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        return False
    else:
        return full_filename in files
