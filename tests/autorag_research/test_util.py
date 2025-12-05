import asyncio
from unittest.mock import AsyncMock

import pytest

from autorag_research.util import run_with_concurrency_limit


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
