"""init command - Download default configuration files."""

import logging
from pathlib import Path

import httpx

logger = logging.getLogger("AutoRAG-Research")

GITHUB_REPO = "NomaDamas/AutoRAG-Research"
GITHUB_BRANCH = "main"
GITHUB_API_BASE = f"https://api.github.com/repos/{GITHUB_REPO}/contents"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}"


def init() -> None:
    """Download default configuration files to the configured directory.

    Downloads configuration files from the AutoRAG-Research GitHub repository
    to your local configs directory. Existing files are not overwritten.

    Examples:
      autorag-research init
      autorag-research --config-path=/my/configs init
    """
    import autorag_research.cli as cli

    config_dir = cli.CONFIG_PATH or Path.cwd() / "configs"
    logger.info(f"Initializing configuration files in {config_dir}")

    downloaded = 0
    skipped = 0
    failed = 0

    with httpx.Client(timeout=30.0) as client:
        # Fetch file list from GitHub API
        files = fetch_config_files_from_github(client)
        if not files:
            raise RuntimeError("Failed to fetch config files from GitHub")  # noqa: TRY003

        logger.info(f"  Found {len(files)} configuration files\n")

        for file_path in files:
            local_path = config_dir / file_path
            url = f"{GITHUB_RAW_BASE}/configs/{file_path}"

            if local_path.exists():
                logger.info(f"  [skip] {file_path} (already exists)")
                skipped += 1
                continue

            response = client.get(url)
            if response.status_code == 200:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_text(response.text)
                logger.info(f"  [ok] {file_path}")
                downloaded += 1
            else:
                logger.error("  [error] %s (HTTP %d)", file_path, response.status_code)
                failed += 1

    logger.info(
        f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed"
        f"\nConfiguration files are in: {config_dir}"
        "\nNext steps:"
        "\n  1. Edit configs/db.yaml with your database credentials"
        "\n  2. Ingest a dataset: autorag-research ingest beir --dataset=scifact"
        "\n  3. Run experiment: autorag-research run --db-name=beir_scifact_test"
    )


def fetch_config_files_from_github(client: httpx.Client) -> list[str]:
    """Fetch list of config files from GitHub API recursively."""
    files = []
    root_prefix = "configs/"

    def fetch_directory(path: str = "configs") -> None:
        url = f"{GITHUB_API_BASE}/{path}"
        response = client.get(url, headers={"Accept": "application/vnd.github.v3+json"})

        if response.status_code != 200:
            logger.warning(f"Failed to fetch directory {path}: HTTP {response.status_code}")
            return

        try:
            items = response.json()
        except Exception as e:
            logger.warning(f"Failed to parse response for {path}: {e}")
            return

        for item in items:
            if item["type"] == "file" and item["name"].endswith((".yaml", ".yml")):
                rel_path = item["path"].removeprefix(root_prefix)
                files.append(rel_path)
            elif item["type"] == "dir":
                fetch_directory(item["path"])

    fetch_directory()
    return sorted(files)
