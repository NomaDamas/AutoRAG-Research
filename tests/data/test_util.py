import os
import tempfile

import pytest

from autorag_research.data.util import download_file_streaming


@pytest.mark.asyncio
async def test_download_file_streaming():
    url = "https://pub-150dd5f5ea254c6699d508a0f11a6d82.r2.dev/test-image.png"

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "downloaded_file.png")

        await download_file_streaming(url, filename)

        # 파일이 생성되었는지 확인
        assert os.path.exists(filename)

        # 파일 크기가 0보다 큰지 확인
        assert os.path.getsize(filename) > 0
