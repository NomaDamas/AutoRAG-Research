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


class TestBytesToPilImage:
    """Tests for bytes_to_pil_image utility function."""

    def test_bytes_to_pil_image_png(self):
        """Test converting PNG bytes to PIL Image."""
        from autorag_research.util import bytes_to_pil_image

        # Create a test image and convert to bytes
        original = Image.new("RGB", (20, 30), color=(255, 0, 0))
        buffer = io.BytesIO()
        original.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        # Convert back to PIL Image
        result = bytes_to_pil_image(img_bytes)

        assert isinstance(result, Image.Image)
        assert result.size == (20, 30)

    def test_bytes_to_pil_image_jpeg(self):
        """Test converting JPEG bytes to PIL Image."""
        from autorag_research.util import bytes_to_pil_image

        # Create a test image and convert to bytes
        original = Image.new("RGB", (15, 25), color=(0, 255, 0))
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()

        # Convert back to PIL Image
        result = bytes_to_pil_image(img_bytes)

        assert isinstance(result, Image.Image)
        assert result.size == (15, 25)

    def test_bytes_to_pil_image_rgba(self):
        """Test converting RGBA PNG bytes to PIL Image."""
        from autorag_research.util import bytes_to_pil_image

        # Create RGBA image
        original = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
        buffer = io.BytesIO()
        original.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        # Convert back to PIL Image
        result = bytes_to_pil_image(img_bytes)

        assert isinstance(result, Image.Image)
        assert result.mode in ["RGB", "RGBA"]


class TestPilImageToDataUri:
    """Tests for pil_image_to_data_uri utility function."""

    def test_pil_image_to_data_uri_rgb(self):
        """Test converting RGB PIL Image to data URI."""
        from autorag_research.util import pil_image_to_data_uri

        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        data_uri = pil_image_to_data_uri(img)

        assert data_uri.startswith("data:image/")
        assert ";base64," in data_uri
        # Should be JPEG for RGB images
        assert "image/jpeg" in data_uri

    def test_pil_image_to_data_uri_rgba(self):
        """Test converting RGBA PIL Image to data URI (PNG format)."""
        from autorag_research.util import pil_image_to_data_uri

        img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
        data_uri = pil_image_to_data_uri(img)

        assert data_uri.startswith("data:image/png")
        assert ";base64," in data_uri

    def test_pil_image_to_data_uri_roundtrip(self):
        """Test that data URI can be decoded back to original image."""
        from autorag_research.util import bytes_to_pil_image, pil_image_to_data_uri

        original = Image.new("RGB", (15, 20), color=(0, 128, 255))
        data_uri = pil_image_to_data_uri(original)

        # Decode the data URI
        _, base64_part = data_uri.split(";base64,")
        decoded_bytes = base64.b64decode(base64_part)
        result = bytes_to_pil_image(decoded_bytes)

        assert result.size == (15, 20)

    def test_pil_image_to_data_uri_format(self):
        """Test data URI format is correct."""
        from autorag_research.util import pil_image_to_data_uri

        img = Image.new("RGB", (5, 5), color=(0, 0, 255))
        data_uri = pil_image_to_data_uri(img)

        # Verify format: data:image/<format>;base64,<data>
        assert data_uri.startswith("data:")
        assert ";base64," in data_uri

        # Extract and verify parts
        prefix, base64_data = data_uri.split(";base64,")
        assert prefix.startswith("data:image/")
        assert len(base64_data) > 0

        # Verify base64 is valid
        decoded = base64.b64decode(base64_data)
        assert len(decoded) > 0
