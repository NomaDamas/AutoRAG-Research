"""Data ingestion and export utilities."""

from platformdirs import user_data_dir

from autorag_research.data.restore import restore_database

PUBLIC_R2_URL = "https://pub-150dd5f5ea254c6699d508a0f11a6d82.r2.dev"  # TODO: Replace to production URL from Cloudflare
USER_DATA_DIR = user_data_dir("autorag_research", "NomaDamas")

__all__ = ["restore_database"]
