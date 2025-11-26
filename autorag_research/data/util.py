import aiofiles
import aiohttp


async def download_file_streaming(url, filename):
    async with (
        aiohttp.ClientSession() as session,
        session.get(url) as response,
        aiofiles.open(filename, mode="wb") as f,
    ):
        async for chunk in response.content.iter_chunked(8192):
            await f.write(chunk)
