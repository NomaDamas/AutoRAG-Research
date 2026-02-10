import asyncio
import base64
import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from autorag_research.util import (
    TokenUsageTracker,
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


class TestValidatePluginName:
    """Tests for validate_plugin_name."""

    def test_valid_simple(self):
        from autorag_research.util import validate_plugin_name

        assert validate_plugin_name("my_search") is True

    def test_valid_single_letter(self):
        from autorag_research.util import validate_plugin_name

        assert validate_plugin_name("a") is True

    def test_valid_with_digits(self):
        from autorag_research.util import validate_plugin_name

        assert validate_plugin_name("search2") is True

    def test_invalid_starts_with_digit(self):
        from autorag_research.util import validate_plugin_name

        assert validate_plugin_name("2search") is False

    def test_invalid_uppercase(self):
        from autorag_research.util import validate_plugin_name

        assert validate_plugin_name("MySearch") is False

    def test_invalid_hyphen(self):
        from autorag_research.util import validate_plugin_name

        assert validate_plugin_name("my-search") is False

    def test_invalid_path_traversal(self):
        from autorag_research.util import validate_plugin_name

        assert validate_plugin_name("../../../evil") is False

    def test_invalid_empty(self):
        from autorag_research.util import validate_plugin_name

        assert validate_plugin_name("") is False

    def test_invalid_starts_with_underscore(self):
        from autorag_research.util import validate_plugin_name

        assert validate_plugin_name("_private") is False


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


class TestNormalizeMinmaxWithNone:
    """Tests for normalize_minmax handling of None values."""

    def test_none_values_preserved(self):
        """Test that None values are preserved in output."""
        from autorag_research.util import normalize_minmax

        scores: list[float | None] = [1.0, None, 3.0]
        result = normalize_minmax(scores)
        assert result == [0.0, None, 1.0]

    def test_none_excluded_from_stats(self):
        """Test that None values don't affect min/max calculation."""
        from autorag_research.util import normalize_minmax

        # Without None: [1.0, 2.0, 3.0] -> [0.0, 0.5, 1.0]
        # With None: min/max should still be 1.0 and 3.0
        scores: list[float | None] = [1.0, None, 2.0, 3.0]
        result = normalize_minmax(scores)
        assert result[0] == 0.0  # min
        assert result[1] is None  # preserved
        assert result[2] == 0.5  # middle
        assert result[3] == 1.0  # max

    def test_all_none(self):
        """Test with all None values."""
        from autorag_research.util import normalize_minmax

        scores: list[float | None] = [None, None, None]
        result = normalize_minmax(scores)
        assert result == [None, None, None]

    def test_single_value_with_none(self):
        """Test with single valid value and None values."""
        from autorag_research.util import normalize_minmax

        scores: list[float | None] = [None, 5.0, None]
        result = normalize_minmax(scores)
        # Single value -> 0.5 (same as all equal)
        assert result == [None, 0.5, None]

    def test_backward_compatible_no_none(self):
        """Test that function still works without None values."""
        from autorag_research.util import normalize_minmax

        scores: list[float | None] = [1.0, 2.0, 3.0]
        result = normalize_minmax(scores)
        assert result == [0.0, 0.5, 1.0]


class TestNormalizeTmmWithNone:
    """Tests for normalize_tmm handling of None values."""

    def test_none_values_preserved(self):
        """Test that None values are preserved in output."""
        from autorag_research.util import normalize_tmm

        scores: list[float | None] = [0.0, None, 100.0]
        result = normalize_tmm(scores, theoretical_min=0.0)
        assert result == [0.0, None, 1.0]

    def test_none_excluded_from_stats(self):
        """Test that None values don't affect max calculation."""
        from autorag_research.util import normalize_tmm

        # actual_max should be 100.0, not affected by None
        scores: list[float | None] = [0.0, None, 50.0, 100.0]
        result = normalize_tmm(scores, theoretical_min=0.0)
        assert result[0] == 0.0
        assert result[1] is None
        assert result[2] == 0.5
        assert result[3] == 1.0

    def test_all_none(self):
        """Test with all None values."""
        from autorag_research.util import normalize_tmm

        scores: list[float | None] = [None, None]
        result = normalize_tmm(scores, theoretical_min=0.0)
        assert result == [None, None]

    def test_backward_compatible_no_none(self):
        """Test that function still works without None values."""
        from autorag_research.util import normalize_tmm

        scores: list[float | None] = [0.0, 50.0, 100.0]
        result = normalize_tmm(scores, theoretical_min=0.0)
        assert result == [0.0, 0.5, 1.0]


class TestNormalizeZscoreWithNone:
    """Tests for normalize_zscore handling of None values."""

    def test_none_values_preserved(self):
        """Test that None values are preserved in output."""
        from autorag_research.util import normalize_zscore

        scores: list[float | None] = [1.0, None, 3.0]
        result = normalize_zscore(scores)
        # mean = 2.0, std = 1.0 (computed from [1.0, 3.0])
        # (1.0 - 2.0) / 1.0 = -1.0
        # (3.0 - 2.0) / 1.0 = 1.0
        assert result[0] == -1.0
        assert result[1] is None
        assert result[2] == 1.0

    def test_none_excluded_from_stats(self):
        """Test that None values don't affect mean/std calculation."""
        from autorag_research.util import normalize_zscore

        # Without None: mean of [0, 10] = 5, std = 5
        scores: list[float | None] = [0.0, None, 10.0]
        result = normalize_zscore(scores)
        # (0 - 5) / 5 = -1.0, (10 - 5) / 5 = 1.0
        assert result[0] == -1.0
        assert result[1] is None
        assert result[2] == 1.0

    def test_all_none(self):
        """Test with all None values."""
        from autorag_research.util import normalize_zscore

        scores: list[float | None] = [None, None]
        result = normalize_zscore(scores)
        assert result == [None, None]

    def test_all_equal_with_none(self):
        """Test all equal values with None (std=0)."""
        from autorag_research.util import normalize_zscore

        scores: list[float | None] = [5.0, None, 5.0]
        result = normalize_zscore(scores)
        assert result == [0.0, None, 0.0]

    def test_backward_compatible_no_none(self):
        """Test that function still works without None values."""
        from autorag_research.util import normalize_zscore

        scores: list[float | None] = [0.0, 10.0]
        result = normalize_zscore(scores)
        assert result[0] == -1.0
        assert result[1] == 1.0


class TestNormalizeDbsfWithNone:
    """Tests for normalize_dbsf handling of None values."""

    def test_none_values_preserved(self):
        """Test that None values are preserved in output."""
        from autorag_research.util import normalize_dbsf

        scores: list[float | None] = [1.0, None, 3.0, 5.0]
        result = normalize_dbsf(scores)
        assert result[1] is None
        # Other values should be in [0, 1]
        assert all(0.0 <= v <= 1.0 for v in result if v is not None)

    def test_none_excluded_from_stats(self):
        """Test that None values don't affect mean/std calculation."""
        from autorag_research.util import normalize_dbsf

        # Mean and std computed only from valid scores
        scores: list[float | None] = [1.0, 2.0, None, 3.0, 4.0, 5.0]
        result = normalize_dbsf(scores)
        assert result[2] is None
        # Middle value (3.0) should be at 0.5 (it's the mean of [1,2,3,4,5])
        assert result[3] == 0.5

    def test_all_none(self):
        """Test with all None values."""
        from autorag_research.util import normalize_dbsf

        scores: list[float | None] = [None, None]
        result = normalize_dbsf(scores)
        assert result == [None, None]

    def test_all_equal_with_none(self):
        """Test all equal values with None."""
        from autorag_research.util import normalize_dbsf

        scores: list[float | None] = [5.0, None, 5.0]
        result = normalize_dbsf(scores)
        assert result == [0.5, None, 0.5]

    def test_backward_compatible_no_none(self):
        """Test that function still works without None values."""
        from autorag_research.util import normalize_dbsf

        scores: list[float | None] = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = normalize_dbsf(scores)
        assert all(0.0 <= v <= 1.0 for v in result if v is not None)
        assert result[2] == 0.5


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


class TestTokenUsageTracker:
    """Tests for the TokenUsageTracker class."""

    def _make_newer_response(self, input_tokens: int, output_tokens: int, total_tokens: int) -> MagicMock:
        """Create mock response with newer LangChain usage_metadata."""
        response = MagicMock()
        response.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        return response

    def _make_older_response(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> MagicMock:
        """Create mock response with older LangChain response_metadata."""
        response = MagicMock(spec=["response_metadata"])
        response.response_metadata = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        }
        return response

    def _make_no_usage_response(self) -> MagicMock:
        """Create mock response with no token usage info."""
        response = MagicMock(spec=[])
        return response

    # --- extract() tests ---

    def test_extract_newer_langchain_format(self):
        """Test extraction from newer LangChain usage_metadata."""
        response = self._make_newer_response(100, 50, 150)
        result = TokenUsageTracker.extract(response)
        assert result == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

    def test_extract_older_langchain_format(self):
        """Test extraction from older LangChain response_metadata."""
        response = self._make_older_response(200, 80, 280)
        result = TokenUsageTracker.extract(response)
        assert result == {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}

    def test_extract_no_usage_returns_none(self):
        """Test that None is returned when no usage info is available."""
        response = self._make_no_usage_response()
        result = TokenUsageTracker.extract(response)
        assert result is None

    def test_extract_empty_usage_metadata(self):
        """Test that empty usage_metadata falls through to response_metadata."""
        response = MagicMock()
        response.usage_metadata = {}
        response.response_metadata = {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        result = TokenUsageTracker.extract(response)
        assert result == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def test_extract_empty_token_usage_in_response_metadata(self):
        """Test that empty token_usage dict returns None."""
        response = MagicMock(spec=["response_metadata"])
        response.response_metadata = {"token_usage": {}}
        result = TokenUsageTracker.extract(response)
        assert result is None

    # --- aggregate() tests ---

    def test_aggregate_both_none(self):
        """Test that None + None returns None."""
        assert TokenUsageTracker.aggregate(None, None) is None

    def test_aggregate_current_none(self):
        """Test that None + dict returns dict."""
        new = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        result = TokenUsageTracker.aggregate(None, new)
        assert result == new

    def test_aggregate_new_none(self):
        """Test that dict + None returns dict."""
        current = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        result = TokenUsageTracker.aggregate(current, None)
        assert result == current

    def test_aggregate_both_dicts(self):
        """Test that two dicts are summed correctly."""
        current = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        new = {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}
        result = TokenUsageTracker.aggregate(current, new)
        assert result == {"prompt_tokens": 300, "completion_tokens": 130, "total_tokens": 430}

    def test_aggregate_different_keys(self):
        """Test aggregation with different key sets."""
        current = {"prompt_tokens": 100, "total_tokens": 100}
        new = {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
        result = TokenUsageTracker.aggregate(current, new)
        assert result == {"prompt_tokens": 150, "completion_tokens": 20, "total_tokens": 170}

    # --- Instance tracking tests ---

    def test_record_from_response(self):
        """Test record() extracts and accumulates from a response."""
        tracker = TokenUsageTracker()
        response = self._make_newer_response(100, 50, 150)

        usage = tracker.record(response)

        assert usage == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        assert tracker.total == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        assert tracker.call_count == 1

    def test_record_usage_from_dict(self):
        """Test record_usage() accumulates from a pre-extracted dict."""
        tracker = TokenUsageTracker()
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

        tracker.record_usage(usage)

        assert tracker.total == usage
        assert tracker.call_count == 1

    def test_multiple_records(self):
        """Test accumulation across multiple records."""
        tracker = TokenUsageTracker()

        tracker.record_usage({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        tracker.record_usage({"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280})

        assert tracker.total == {"prompt_tokens": 300, "completion_tokens": 130, "total_tokens": 430}
        assert tracker.call_count == 2

    def test_record_none_usage(self):
        """Test that recording None doesn't affect total but doesn't add to history."""
        tracker = TokenUsageTracker()

        tracker.record_usage({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        tracker.record_usage(None)

        assert tracker.total == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        assert tracker.call_count == 1  # None doesn't count

    def test_record_response_with_no_usage(self):
        """Test record() with a response that has no usage info."""
        tracker = TokenUsageTracker()
        response = self._make_no_usage_response()

        usage = tracker.record(response)

        assert usage is None
        assert tracker.total is None
        assert tracker.call_count == 0

    def test_total_initially_none(self):
        """Test that total starts as None."""
        tracker = TokenUsageTracker()
        assert tracker.total is None

    def test_history_returns_defensive_copy(self):
        """Test that history returns a copy, not the internal list."""
        tracker = TokenUsageTracker()
        tracker.record_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

        history = tracker.history
        history.append({"prompt_tokens": 999, "completion_tokens": 999, "total_tokens": 999})

        assert tracker.call_count == 1  # Internal list unaffected

    def test_history_entries_are_copies(self):
        """Test that history entries are copies of the original dicts."""
        tracker = TokenUsageTracker()
        original = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        tracker.record_usage(original)

        # Mutating the original should not affect the tracker
        original["prompt_tokens"] = 999

        assert tracker.history[0]["prompt_tokens"] == 10

    def test_history_contains_per_call_data(self):
        """Test that history contains individual call data."""
        tracker = TokenUsageTracker()
        tracker.record_usage({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        tracker.record_usage({"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280})

        history = tracker.history
        assert len(history) == 2
        assert history[0] == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        assert history[1] == {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}

    def test_reset_clears_all_state(self):
        """Test that reset() clears total and history."""
        tracker = TokenUsageTracker()
        tracker.record_usage({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        tracker.record_usage({"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280})

        tracker.reset()

        assert tracker.total is None
        assert tracker.history == []
        assert tracker.call_count == 0

    def test_mixed_none_and_real_usage(self):
        """Test tracker with a mix of None and real usage responses."""
        tracker = TokenUsageTracker()

        response1 = self._make_newer_response(100, 50, 150)
        response2 = self._make_no_usage_response()
        response3 = self._make_older_response(200, 80, 280)

        tracker.record(response1)
        tracker.record(response2)
        tracker.record(response3)

        assert tracker.total == {"prompt_tokens": 300, "completion_tokens": 130, "total_tokens": 430}
        assert tracker.call_count == 2  # Only non-None entries
