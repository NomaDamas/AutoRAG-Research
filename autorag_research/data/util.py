import asyncio
import os
import tempfile

import aiofiles
import aiohttp

from autorag_research.data import PUBLIC_R2_URL, restore_database

DATASET_TAGS = {
    "scifact": {
        "embeddinggemma-300m": "scifact-embeddinggemma-300m.dump",
    }
}


def setup_dataset(
    dataset_name: str,
    embedding_model_name: str,
    host: str,
    user: str,
    password: str,
    port: int = 5432,
    **kwargs,
):
    """Set up a dataset by downloading and restoring it to a PostgreSQL database.

    Downloads the pre-built dataset dump file from AutoRAG-Research storage and restores it
    to a PostgreSQL database with the naming convention "{dataset_name}_{embedding_model_name}".

    Args:
        dataset_name: Name of the dataset to set up (e.g., "scifact").
        embedding_model_name: Name of the embedding model used for the dataset.
        host: PostgreSQL server hostname.
        user: PostgreSQL username.
        password: PostgreSQL password.
        port: PostgreSQL server port. Defaults to 5432.
        **kwargs: Additional keyword arguments passed to restore_database.

    Raises:
        KeyError: If the dataset_name or embedding_model_name is not found in DATASET_TAGS.
    """
    dump_filename = DATASET_TAGS[dataset_name][embedding_model_name]
    with tempfile.TemporaryDirectory() as tmpdir:
        asyncio.run(download_file_streaming(f"{PUBLIC_R2_URL}/{dump_filename}", os.path.join(tmpdir, dump_filename)))
        restore_database(
            os.path.join(tmpdir, dump_filename),
            host=host,
            user=user,
            password=password,
            database=f"{dataset_name}_{embedding_model_name}",
            port=port,
            **kwargs,
        )


async def download_file_streaming(url, filename):
    async with (
        aiohttp.ClientSession() as session,
        session.get(url) as response,
        aiofiles.open(filename, mode="wb") as f,
    ):
        async for chunk in response.content.iter_chunked(8192):
            await f.write(chunk)
