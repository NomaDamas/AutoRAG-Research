"""Data ingestion and export utilities."""

from platformdirs import user_data_dir

from autorag_research.data.hf_storage import (
    HF_ORG,
    download_dump,
    dump_exists,
    list_available_dumps,
    upload_dump,
)
from autorag_research.data.restore import restore_database
from autorag_research.data.util import setup_dataset

USER_DATA_DIR = user_data_dir("autorag_research", "NomaDamas")

__all__ = [
    "HF_ORG",
    "USER_DATA_DIR",
    "download_dump",
    "dump_exists",
    "list_available_dumps",
    "restore_database",
    "setup_dataset",
    "upload_dump",
]
