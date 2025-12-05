import asyncio
import functools
import itertools
import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypeVar, cast

import numpy as np
import pandas as pd
import tiktoken
from pydantic import BaseModel as BM
from pydantic.v1 import BaseModel

logger = logging.getLogger("AutoRAG-Research")

T = TypeVar("T")
R = TypeVar("R")


def to_list(item: Any) -> Any:
    """Recursively convert collections to Python lists.

    Args:
        item: The item to convert to a list. Can be numpy array, pandas Series,
            or any iterable collection.

    Returns:
        The converted Python list.
    """
    if isinstance(item, np.ndarray):
        # Convert numpy array to list and recursively process each element
        return [to_list(sub_item) for sub_item in item.tolist()]
    elif isinstance(item, pd.Series):
        # Convert pandas Series to list and recursively process each element
        return [to_list(sub_item) for sub_item in item.tolist()]
    elif isinstance(item, Iterable) and not isinstance(item, (str, bytes, BaseModel, BM)):
        # Recursively process each element in other iterables
        return [to_list(sub_item) for sub_item in item]
    else:
        return item


def convert_inputs_to_list(func: Callable) -> Callable:
    """Decorator to convert all function inputs to Python lists.

    Args:
        func: The function to be decorated.

    Returns:
        The wrapped function that converts all inputs to lists.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = [to_list(arg) for arg in args]
        new_kwargs = {k: to_list(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapper


def truncate_texts(str_list: list[str], max_tokens: int) -> list[str]:
    """Truncate each string in the list to a maximum number of tokens using tiktoken.

    Args:
        str_list: List of strings to be truncated.
        max_tokens: Maximum number of tokens allowed per string.

    Returns:
        List of truncated strings.
    """
    encoder = tiktoken.get_encoding("cl100k_base")

    truncated_list = []
    for text in str_list:
        tokens = encoder.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        truncated_text = encoder.decode(tokens)
        truncated_list.append(truncated_text)

    return truncated_list


def unpack_and_run(target_list: list[list[Any]], func: Callable, *args: tuple, **kwargs: Any) -> list[Any]:
    """Unpack each sublist in target_list and run func with the unpacked arguments.

    Args:
        target_list: List of sublists to be unpacked and processed.
        func: Callable function to run on the flattened list.
        *args: Additional positional arguments to pass to func.
        **kwargs: Additional keyword arguments to pass to func.

    Returns:
        List of results grouped by original sublist lengths.
    """
    lengths = list(map(len, target_list))
    flattened = list(itertools.chain.from_iterable(target_list))

    unpacked_results = func(flattened, *args, **kwargs)
    iterator = iter(unpacked_results)
    result = [list(itertools.islice(iterator, length)) for length in lengths]

    return result


async def run_with_concurrency_limit(
    items: Iterable[T],
    async_func: Callable[[T], Awaitable[R]],
    max_concurrency: int,
    error_message: str = "Task failed",
) -> list[R | None]:
    """Run async function on items with concurrency limit using semaphore.

    A generic utility for running async operations with controlled concurrency.
    Each item is processed by the async function, with at most `max_concurrency`
    operations running simultaneously.

    Args:
        items: Iterable of items to process.
        async_func: Async function that takes an item and returns a result.
        max_concurrency: Maximum number of concurrent operations.
        error_message: Message to log when an operation fails.

    Returns:
        List of results (or None if failed) in same order as items.

    Example:
        ```python
        async def embed_text(text: str) -> list[float]:
            return await some_api_call(text)

        texts = ["hello", "world", "test"]
        embeddings = await run_with_concurrency_limit(
            texts,
            embed_text,
            max_concurrency=5,
            error_message="Failed to embed text",
        )
        ```
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_with_semaphore(item: T) -> R | None:
        async with semaphore:
            try:
                return await async_func(item)
            except Exception:
                logger.exception(error_message)
                return None

    tasks = [process_with_semaphore(item) for item in items]
    return await asyncio.gather(*tasks)


def to_async_func(func: Callable[..., R]) -> Callable[..., Awaitable[R]]:
    """Convert a synchronous function to an asynchronous function.

    Args:
        func: The synchronous function to convert.

    Returns:
        An asynchronous function that runs the synchronous function in a thread.
    """

    if asyncio.iscoroutinefunction(func):
        return cast(Callable[..., Awaitable[R]], func)  # Already async

    @functools.wraps(func)
    async def async_func(*args: tuple, **kwargs: Any) -> R:
        return await asyncio.to_thread(func, *args, **kwargs)

    return async_func
