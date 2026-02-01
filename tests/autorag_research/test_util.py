import asyncio
import base64
import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from autorag_research.util import extract_image_from_data_uri, pil_image_to_bytes, run_with_concurrency_limit


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


class TestExtractTokenLogprobs:
    """Tests for extract_token_logprobs utility function.

    This function extracts log probabilities from LangChain LLM responses.
    Used by MAIN-RAG pipeline for relevance scoring.
    """

    def _create_mock_response_with_logprobs(
        self,
        content: str,
        logprobs_content: list[dict],
    ) -> MagicMock:
        """Helper to create mock LLM response with logprobs metadata."""
        response = MagicMock()
        response.content = content
        response.response_metadata = {
            "logprobs": {
                "content": logprobs_content,
            }
        }
        return response

    def test_extract_logprobs_returns_dict_when_present(self):
        """Test that logprobs are correctly extracted from response metadata."""
        from autorag_research.util import extract_token_logprobs

        logprobs_content = [
            {"token": "Yes", "logprob": -0.001, "bytes": [89, 101, 115], "top_logprobs": []},
        ]
        response = self._create_mock_response_with_logprobs("Yes", logprobs_content)

        result = extract_token_logprobs(response)

        assert result is not None
        assert "Yes" in result
        assert result["Yes"] == -0.001

    def test_extract_logprobs_returns_none_when_not_present(self):
        """Test that None is returned when response has no logprobs."""
        from autorag_research.util import extract_token_logprobs

        response = MagicMock()
        response.content = "Yes"
        response.response_metadata = {}

        result = extract_token_logprobs(response)

        assert result is None

    def test_extract_logprobs_returns_none_when_no_response_metadata(self):
        """Test that None is returned when response has no response_metadata attribute."""
        from autorag_research.util import extract_token_logprobs

        response = MagicMock(spec=[])  # No attributes

        result = extract_token_logprobs(response)

        assert result is None

    def test_extract_logprobs_filters_by_target_tokens(self):
        """Test that only target tokens are returned when specified."""
        from autorag_research.util import extract_token_logprobs

        logprobs_content = [
            {"token": "Yes", "logprob": -0.001, "bytes": [89, 101, 115], "top_logprobs": []},
            {"token": ",", "logprob": -0.5, "bytes": [44], "top_logprobs": []},
            {"token": " the", "logprob": -0.3, "bytes": [32, 116, 104, 101], "top_logprobs": []},
        ]
        response = self._create_mock_response_with_logprobs("Yes, the", logprobs_content)

        result = extract_token_logprobs(response, target_tokens=["Yes", "No"])

        assert result is not None
        assert "Yes" in result
        assert "," not in result
        assert " the" not in result

    def test_extract_logprobs_case_insensitive_matching(self):
        """Test that target token matching is case-insensitive."""
        from autorag_research.util import extract_token_logprobs

        logprobs_content = [
            {"token": "yes", "logprob": -0.001, "bytes": [121, 101, 115], "top_logprobs": []},
        ]
        response = self._create_mock_response_with_logprobs("yes", logprobs_content)

        result = extract_token_logprobs(response, target_tokens=["Yes", "No"])

        assert result is not None
        assert "yes" in result

    def test_extract_logprobs_finds_target_in_top_logprobs(self):
        """Test that target tokens are found in top_logprobs alternatives."""
        from autorag_research.util import extract_token_logprobs

        logprobs_content = [
            {
                "token": "Yes",
                "logprob": -0.001,
                "bytes": [89, 101, 115],
                "top_logprobs": [
                    {"token": "Yes", "logprob": -0.001, "bytes": [89, 101, 115]},
                    {"token": "No", "logprob": -6.5, "bytes": [78, 111]},
                ],
            },
        ]
        response = self._create_mock_response_with_logprobs("Yes", logprobs_content)

        result = extract_token_logprobs(response, target_tokens=["Yes", "No"])

        assert result is not None
        assert "Yes" in result
        assert "No" in result
        assert result["No"] == -6.5

    def test_extract_logprobs_returns_none_for_empty_content(self):
        """Test that None is returned when logprobs content is empty."""
        from autorag_research.util import extract_token_logprobs

        response = MagicMock()
        response.content = ""
        response.response_metadata = {"logprobs": {"content": []}}

        result = extract_token_logprobs(response)

        assert result is None

    def test_extract_logprobs_handles_missing_logprob_value(self):
        """Test that tokens with None logprob values are skipped."""
        from autorag_research.util import extract_token_logprobs

        logprobs_content = [
            {"token": "Yes", "logprob": None, "bytes": [89, 101, 115], "top_logprobs": []},
            {"token": "!", "logprob": -0.1, "bytes": [33], "top_logprobs": []},
        ]
        response = self._create_mock_response_with_logprobs("Yes!", logprobs_content)

        result = extract_token_logprobs(response)

        assert result is not None
        assert "Yes" not in result
        assert "!" in result

    def test_extract_logprobs_returns_all_tokens_when_no_target_specified(self):
        """Test that all tokens are returned when target_tokens is None."""
        from autorag_research.util import extract_token_logprobs

        logprobs_content = [
            {"token": "The", "logprob": -0.1, "bytes": [84, 104, 101], "top_logprobs": []},
            {"token": " answer", "logprob": -0.2, "bytes": [32, 97, 110], "top_logprobs": []},
            {"token": " is", "logprob": -0.3, "bytes": [32, 105, 115], "top_logprobs": []},
        ]
        response = self._create_mock_response_with_logprobs("The answer is", logprobs_content)

        result = extract_token_logprobs(response, target_tokens=None)

        assert result is not None
        assert len(result) == 3
        assert "The" in result
        assert " answer" in result
        assert " is" in result
