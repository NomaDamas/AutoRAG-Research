import asyncio
import base64
import io
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from autorag_research.util import (
    extract_image_from_data_uri,
    pil_image_to_bytes,
    run_with_concurrency_limit,
)


class TestRunWithConcurrencyLimit:
    @pytest.mark.asyncio
    async def test_basic_functionality(self):
        """Test that the function processes all items and returns results in order."""

        async def double(x: int) -> int:
            return x * 2

        items = [1, 2, 3, 4, 5]
        results = await run_with_concurrency_limit(items, double, max_concurrency=3)

        assert results == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_concurrency_limit_is_respected(self):
        """Test that max_concurrency limit is respected."""
        concurrent_count = 0
        max_concurrent_observed = 0

        async def track_concurrency(x: int) -> int:
            nonlocal concurrent_count, max_concurrent_observed
            concurrent_count += 1
            max_concurrent_observed = max(max_concurrent_observed, concurrent_count)
            await asyncio.sleep(0.05)  # Simulate some async work
            concurrent_count -= 1
            return x

        items = list(range(10))
        max_concurrency = 3

        await run_with_concurrency_limit(items, track_concurrency, max_concurrency=max_concurrency)

        assert max_concurrent_observed <= max_concurrency

    @pytest.mark.asyncio
    async def test_handles_exceptions_gracefully(self):
        """Test that exceptions are caught and None is returned for failed items."""

        async def fail_on_three(x: int) -> int:
            if x == 3:
                raise ValueError
            return x * 2

        items = [1, 2, 3, 4, 5]
        results = await run_with_concurrency_limit(
            items, fail_on_three, max_concurrency=5, error_message="Failed to process"
        )

        assert results == [2, 4, None, 8, 10]

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Test that empty input returns empty list."""

        async def identity(x: int) -> int:
            return x

        results = await run_with_concurrency_limit([], identity, max_concurrency=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_single_item(self):
        """Test with a single item."""

        async def double(x: int) -> int:
            return x * 2

        results = await run_with_concurrency_limit([42], double, max_concurrency=5)

        assert results == [84]

    @pytest.mark.asyncio
    async def test_all_items_fail(self):
        """Test when all items fail."""

        async def always_fail(x: int) -> int:
            raise RuntimeError

        items = [1, 2, 3]
        results = await run_with_concurrency_limit(items, always_fail, max_concurrency=5)

        assert results == [None, None, None]

    @pytest.mark.asyncio
    async def test_with_different_return_types(self):
        """Test with different return types (strings)."""

        async def to_string(x: int) -> str:
            return f"item_{x}"

        items = [1, 2, 3]
        results = await run_with_concurrency_limit(items, to_string, max_concurrency=2)

        assert results == ["item_1", "item_2", "item_3"]

    @pytest.mark.asyncio
    async def test_with_list_return_type(self):
        """Test with list return type (like embeddings)."""

        async def fake_embed(text: str) -> list[float]:
            return [float(len(text)), 0.5, 0.3]

        items = ["hello", "world", "test"]
        results = await run_with_concurrency_limit(items, fake_embed, max_concurrency=2)

        assert results == [[5.0, 0.5, 0.3], [5.0, 0.5, 0.3], [4.0, 0.5, 0.3]]

    @pytest.mark.asyncio
    async def test_preserves_order_with_varying_delays(self):
        """Test that results are returned in the same order as input, even with varying delays."""

        async def delayed_identity(x: int) -> int:
            # Longer delay for smaller numbers to test order preservation
            await asyncio.sleep((5 - x) * 0.01)
            return x

        items = [1, 2, 3, 4, 5]
        results = await run_with_concurrency_limit(items, delayed_identity, max_concurrency=5)

        assert results == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_max_concurrency_one(self):
        """Test with max_concurrency=1 (sequential execution)."""
        execution_order: list[int] = []

        async def track_order(x: int) -> int:
            execution_order.append(x)
            await asyncio.sleep(0.01)
            return x

        items = [1, 2, 3, 4, 5]
        results = await run_with_concurrency_limit(items, track_order, max_concurrency=1)

        assert results == [1, 2, 3, 4, 5]
        # With max_concurrency=1, items should be processed in order
        assert execution_order == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_with_async_mock(self):
        """Test integration with AsyncMock for mocking async functions."""
        mock_func = AsyncMock(side_effect=lambda x: x * 10)

        items = [1, 2, 3]
        results = await run_with_concurrency_limit(items, mock_func, max_concurrency=2)

        assert results == [10, 20, 30]
        assert mock_func.call_count == 3


def mock_func(x: int) -> int:
    return x * 2


@pytest.mark.asyncio
async def test_to_async_func():
    """Test that to_async_func correctly converts a sync function to async."""
    from autorag_research.util import to_async_func

    async_func = to_async_func(mock_func)

    result = await async_func(5)
    assert result == 10


class TestPilImagesToBytes:
    """Test static method pil_images_to_bytes."""

    def test_pil_images_to_bytes_jpeg_rgb(self):
        """Test converting RGB image to JPEG bytes."""
        # Create a simple RGB image
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes, mimetype = pil_image_to_bytes(img)
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0
        assert mimetype == "image/jpeg"

        # Verify bytes can be read back as image
        loaded_img = Image.open(io.BytesIO(img_bytes))
        assert loaded_img.size == (100, 100)

    def test_pil_images_to_bytes_png_rgba(self):
        """Test converting RGBA image to PNG bytes (transparent)."""
        # Create RGBA image with transparency
        img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        img_bytes, mimetype = pil_image_to_bytes(img)
        assert isinstance(img_bytes, bytes)
        assert mimetype == "image/png"

    def test_pil_images_to_bytes_png_la_mode(self):
        """Test converting LA mode (grayscale with alpha) to PNG."""
        img = Image.new("LA", (30, 30), color=(128, 200))
        img_bytes, mimetype = pil_image_to_bytes(img)
        assert mimetype == "image/png"
        assert isinstance(img_bytes, bytes)

    def test_pil_images_to_bytes_png_palette_mode(self):
        """Test converting palette mode (P) to PNG."""
        # Create RGB and convert to palette mode
        img = Image.new("RGB", (20, 20), color="blue")
        img_p = img.convert("P")
        img_bytes, mimetype = pil_image_to_bytes(img_p)
        assert mimetype == "image/png"
        assert isinstance(img_bytes, bytes)


class TestExtractImageFromDataUri:
    def test_extract_image_from_data_uri_jpeg(self):
        # Create minimal valid JPEG header bytes
        jpeg_header = bytes([
            0xFF,
            0xD8,
            0xFF,
            0xE0,
            0x00,
            0x10,
            0x4A,
            0x46,
            0x49,
            0x46,
            0x00,
            0x01,
            0x01,
            0x00,
            0x00,
            0x01,
            0x00,
            0x01,
            0x00,
            0x00,
        ])
        encoded = base64.b64encode(jpeg_header).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{encoded}"

        img_bytes, mimetype = extract_image_from_data_uri(data_uri)

        assert mimetype == "image/jpeg"
        assert isinstance(img_bytes, bytes)
        assert img_bytes == jpeg_header

    def test_extract_image_from_data_uri_png(self):
        # Create minimal PNG header bytes
        png_header = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        encoded = base64.b64encode(png_header).decode("utf-8")
        data_uri = f"data:image/png;base64,{encoded}"

        img_bytes, mimetype = extract_image_from_data_uri(data_uri)

        assert mimetype == "image/png"
        assert isinstance(img_bytes, bytes)
        assert img_bytes == png_header

    def test_extract_image_from_data_uri_gif(self):
        gif_header = b"GIF89a"
        encoded = base64.b64encode(gif_header).decode("utf-8")
        data_uri = f"data:image/gif;base64,{encoded}"

        img_bytes, mimetype = extract_image_from_data_uri(data_uri)

        assert mimetype == "image/gif"
        assert img_bytes == gif_header

    def test_extract_image_from_data_uri_preserves_bytes(self):
        test_data = b"\x00\x01\x02\xff\xfe\xfd"
        encoded = base64.b64encode(test_data).decode("utf-8")
        data_uri = f"data:image/png;base64,{encoded}"

        img_bytes, _ = extract_image_from_data_uri(data_uri)

        assert img_bytes == test_data


class TestNormalizeMinmax:
    def test_basic(self):
        """Test basic min-max normalization."""
        from autorag_research.util import normalize_minmax

        scores = [1.0, 2.0, 3.0]
        result = normalize_minmax(scores)
        assert result == [0.0, 0.5, 1.0]

    def test_negative_values(self):
        """Test with negative values."""
        from autorag_research.util import normalize_minmax

        scores = [-10.0, 0.0, 10.0]
        result = normalize_minmax(scores)
        assert result == [0.0, 0.5, 1.0]

    def test_all_equal(self):
        """Test when all scores are equal."""
        from autorag_research.util import normalize_minmax

        scores = [5.0, 5.0, 5.0]
        result = normalize_minmax(scores)
        assert result == [0.5, 0.5, 0.5]

    def test_empty(self):
        """Test with empty list."""
        from autorag_research.util import normalize_minmax

        result = normalize_minmax([])
        assert result == []

    def test_single_element(self):
        """Test with single element."""
        from autorag_research.util import normalize_minmax

        result = normalize_minmax([5.0])
        assert result == [0.5]


class TestNormalizeTmm:
    def test_basic(self):
        """Test theoretical min-max normalization with actual max."""
        from autorag_research.util import normalize_tmm

        # With theoretical_min=-1.0 and actual_max=0.5
        # score_range = 0.5 - (-1.0) = 1.5
        # -0.5 -> (-0.5 - (-1.0)) / 1.5 = 0.5 / 1.5 = 0.333...
        # 0.0 -> (0.0 - (-1.0)) / 1.5 = 1.0 / 1.5 = 0.666...
        # 0.5 -> (0.5 - (-1.0)) / 1.5 = 1.5 / 1.5 = 1.0
        scores = [-0.5, 0.0, 0.5]
        result = normalize_tmm(scores, theoretical_min=-1.0)
        assert len(result) == 3
        assert result[2] == 1.0  # Max value should be 1.0
        assert result[0] < result[1] < result[2]  # Monotonic increasing

    def test_cosine_similarity_bounds(self):
        """Test with cosine similarity theoretical min -1."""
        from autorag_research.util import normalize_tmm

        scores = [-1.0, 0.0, 1.0]
        result = normalize_tmm(scores, theoretical_min=-1.0)
        # actual_max = 1.0, score_range = 1.0 - (-1.0) = 2.0
        assert result == [0.0, 0.5, 1.0]

    def test_bm25_bounds(self):
        """Test with BM25 theoretical min 0."""
        from autorag_research.util import normalize_tmm

        scores = [0.0, 50.0, 100.0]
        result = normalize_tmm(scores, theoretical_min=0.0)
        # actual_max = 100.0, score_range = 100.0 - 0.0 = 100.0
        assert result == [0.0, 0.5, 1.0]

    def test_empty(self):
        """Test with empty list."""
        from autorag_research.util import normalize_tmm

        result = normalize_tmm([], theoretical_min=0.0)
        assert result == []

    def test_all_equal_at_min(self):
        """Test with all scores equal to theoretical min."""
        from autorag_research.util import normalize_tmm

        result = normalize_tmm([5.0, 5.0], theoretical_min=5.0)
        # actual_max = 5.0, score_range = 5.0 - 5.0 = 0.0
        assert result == [0.5, 0.5]


class TestNormalizeZscore:
    def test_basic(self):
        """Test basic z-score normalization."""
        from autorag_research.util import normalize_zscore

        scores = [1.0, 2.0, 3.0]
        result = normalize_zscore(scores)
        # Mean = 2.0, std = sqrt(2/3) ≈ 0.8165
        assert len(result) == 3
        assert abs(result[1]) < 1e-10  # Middle value should be ~0 (mean)
        assert result[0] < 0  # Below mean
        assert result[2] > 0  # Above mean

    def test_all_equal(self):
        """Test when all scores are equal (std=0)."""
        from autorag_research.util import normalize_zscore

        scores = [5.0, 5.0, 5.0]
        result = normalize_zscore(scores)
        assert result == [0.0, 0.0, 0.0]

    def test_empty(self):
        """Test with empty list."""
        from autorag_research.util import normalize_zscore

        result = normalize_zscore([])
        assert result == []

    def test_symmetry(self):
        """Test symmetry around mean."""
        from autorag_research.util import normalize_zscore

        scores = [0.0, 10.0]
        result = normalize_zscore(scores)
        assert abs(result[0] + result[1]) < 1e-10  # Should be symmetric


class TestNormalizeDbsf:
    def test_basic(self):
        """Test basic DBSF normalization."""
        from autorag_research.util import normalize_dbsf

        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = normalize_dbsf(scores)
        # All values should be in [0, 1]
        assert all(0.0 <= v <= 1.0 for v in result)
        # Values should be in ascending order
        assert result == sorted(result)

    def test_clipping_behavior(self):
        """Test that values are clipped to [0, 1] range."""
        from autorag_research.util import normalize_dbsf

        # With any distribution, values should be clipped to [0, 1]
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = normalize_dbsf(scores)
        # All values should be in [0, 1]
        assert all(0.0 <= v <= 1.0 for v in result)
        # Middle value should be at 0.5 (it's the mean)
        assert result[2] == 0.5

    def test_all_equal(self):
        """Test when all scores are equal."""
        from autorag_research.util import normalize_dbsf

        scores = [5.0, 5.0, 5.0]
        result = normalize_dbsf(scores)
        assert result == [0.5, 0.5, 0.5]

    def test_empty(self):
        """Test with empty list."""
        from autorag_research.util import normalize_dbsf

        result = normalize_dbsf([])
        assert result == []

    def test_negative_values(self):
        """Test with negative values."""
        from autorag_research.util import normalize_dbsf

        scores = [-10.0, -5.0, 0.0, 5.0, 10.0]
        result = normalize_dbsf(scores)
        # All values should be in [0, 1]
        assert all(0.0 <= v <= 1.0 for v in result)
        # Should maintain order
        assert result == sorted(result)
