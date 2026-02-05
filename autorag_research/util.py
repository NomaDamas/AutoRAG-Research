import asyncio
import base64
import functools
import io
import itertools
import logging
import re
import string
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypeVar, cast

import numpy as np
import pandas as pd
import tiktoken
from PIL import Image
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


def pil_image_to_bytes(image: Image.Image) -> tuple[bytes, str]:
    """Convert PIL image to bytes with mimetype.

    Args:
        image: PIL Image object.

    Returns:
        Tuple of (image_bytes, mimetype).
    """
    buffer = io.BytesIO()
    # Determine format based on image mode
    img_format = "PNG" if image.mode in ("RGBA", "LA", "P") else "JPEG"
    image.save(buffer, format=img_format)
    mimetype = f"image/{img_format.lower()}"
    return buffer.getvalue(), mimetype


def normalize_string(s: str) -> str:
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles, and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_image_from_data_uri(data_uri: str) -> tuple[bytes, str]:
    """Extract image bytes and mimetype from a data URI."""
    match = re.match(r"data:([^;]+);base64,(.+)", data_uri)
    if not match:
        msg = f"Invalid data URI format: {data_uri[:50]}..."
        raise ValueError(msg)
    mimetype = match.group(1)
    base64_data = match.group(2)
    image_bytes = base64.b64decode(base64_data)
    return image_bytes, mimetype


def normalize_minmax(scores: list[float | None]) -> list[float | None]:
    """Min-max normalization to [0, 1] range.

    Scales scores linearly so that the minimum becomes 0 and maximum becomes 1.
    If all scores are equal, returns a list of 0.5 values.
    None values are preserved and excluded from statistics calculation.

    Args:
        scores: List of numeric scores to normalize. None values are preserved.

    Returns:
        List of normalized scores in [0, 1] range, with None preserved.

    Example:
        >>> normalize_minmax([1.0, 2.0, 3.0])
        [0.0, 0.5, 1.0]
        >>> normalize_minmax([1.0, None, 3.0])
        [0.0, None, 1.0]
    """
    if not scores:
        return []

    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return list(scores)

    min_score = min(valid_scores)
    max_score = max(valid_scores)
    score_range = max_score - min_score

    if score_range == 0:
        return [0.5 if s is not None else None for s in scores]

    return [(s - min_score) / score_range if s is not None else None for s in scores]


def normalize_tmm(
    scores: list[float | None],
    theoretical_min: float,
) -> list[float | None]:
    """Theoretical min-max normalization using theoretical min and actual max.

    Uses the theoretical minimum bound and actual maximum from the data.
    This is useful when the minimum is known (e.g., 0 for BM25) but the
    maximum varies per query.
    None values are preserved and excluded from statistics calculation.

    Args:
        scores: List of numeric scores to normalize. None values are preserved.
        theoretical_min: Known minimum possible score (e.g., 0 for BM25, -1 for cosine).

    Returns:
        List of normalized scores in [0, 1] range, with None preserved.

    Example:
        >>> normalize_tmm([0.0, 50.0, 100.0], theoretical_min=0.0)
        [0.0, 0.5, 1.0]
        >>> normalize_tmm([0.0, None, 100.0], theoretical_min=0.0)
        [0.0, None, 1.0]
    """
    if not scores:
        return []

    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return list(scores)

    actual_max = max(valid_scores)
    score_range = actual_max - theoretical_min
    if score_range == 0:
        return [0.5 if s is not None else None for s in scores]

    return [(s - theoretical_min) / score_range if s is not None else None for s in scores]


def normalize_zscore(scores: list[float | None]) -> list[float | None]:
    """Z-score standardization (mean=0, std=1).

    Centers scores around mean and scales by standard deviation.
    If standard deviation is 0 (all scores equal), returns all zeros.
    None values are preserved and excluded from statistics calculation.

    Args:
        scores: List of numeric scores to normalize. None values are preserved.

    Returns:
        List of z-score normalized values, with None preserved.

    Example:
        >>> normalize_zscore([1.0, 2.0, 3.0])
        [-1.2247..., 0.0, 1.2247...]
        >>> normalize_zscore([1.0, None, 3.0])
        [-1.0, None, 1.0]
    """
    if not scores:
        return []

    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return list(scores)

    n = len(valid_scores)
    mean = sum(valid_scores) / n
    variance = sum((s - mean) ** 2 for s in valid_scores) / n
    std = variance**0.5

    if std == 0:
        return [0.0 if s is not None else None for s in scores]

    return [(s - mean) / std if s is not None else None for s in scores]


def normalize_dbsf(scores: list[float | None]) -> list[float | None]:
    """3-sigma distribution-based score fusion normalization.

    Normalizes using mean ± 3*std as bounds, then clips to [0, 1].
    This method is robust to outliers and works well when combining
    scores from different distributions.
    None values are preserved and excluded from statistics calculation.

    Reference: "Score Normalization in Multi-Engine Text Retrieval"

    Args:
        scores: List of numeric scores to normalize. None values are preserved.

    Returns:
        List of normalized scores clipped to [0, 1] range, with None preserved.

    Example:
        >>> normalize_dbsf([1.0, 2.0, 3.0, 4.0, 5.0])
        [0.0, 0.25, 0.5, 0.75, 1.0]  # approximately
        >>> normalize_dbsf([1.0, None, 3.0, 4.0, 5.0])
        [0.0, None, 0.333..., 0.5, 0.666...]  # approximately
    """
    if not scores:
        return []

    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return list(scores)

    n = len(valid_scores)
    mean = sum(valid_scores) / n
    variance = sum((s - mean) ** 2 for s in valid_scores) / n
    std = variance**0.5

    if std == 0:
        return [0.5 if s is not None else None for s in scores]

    # Use 3-sigma bounds
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    score_range = upper_bound - lower_bound

    def normalize_single(s: float | None) -> float | None:
        if s is None:
            return None
        normalized = (s - lower_bound) / score_range
        return max(0.0, min(1.0, normalized))

    return [normalize_single(s) for s in scores]


def aggregate_token_usage(results: list[dict]) -> tuple[int, int, int, int]:
    """Aggregate token usage from generation results.

    Args:
        results: List of generation result dicts with token_usage and execution_time.

    Returns:
        Tuple of (prompt_tokens, completion_tokens, embedding_tokens, execution_time_ms).
    """
    prompt_tokens = 0
    completion_tokens = 0
    embedding_tokens = 0
    execution_time_ms = 0

    for result in results:
        if result["token_usage"]:
            prompt_tokens += result["token_usage"].get("prompt_tokens", 0)
            completion_tokens += result["token_usage"].get("completion_tokens", 0)
            embedding_tokens += result["token_usage"].get("embedding_tokens", 0)
        execution_time_ms += result["execution_time"]

    return prompt_tokens, completion_tokens, embedding_tokens, execution_time_ms
