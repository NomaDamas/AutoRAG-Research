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


def bytes_to_pil_image(image_bytes: bytes) -> Image.Image:
    """Convert image bytes to PIL Image.

    Args:
        image_bytes: Raw image bytes (PNG, JPEG, etc.)

    Returns:
        PIL Image object.
    """
    return Image.open(io.BytesIO(image_bytes))


def pil_image_to_data_uri(image: Image.Image) -> str:
    """Convert PIL Image to data URI for multi-modal LLMs.

    Args:
        image: PIL Image object.

    Returns:
        Data URI string (e.g., "data:image/png;base64,iVBORw0...").
    """
    img_bytes, mimetype = pil_image_to_bytes(image)
    b64_data = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mimetype};base64,{b64_data}"


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


def extract_langchain_token_usage(response: Any) -> dict[str, int] | None:
    """Extract token usage from a LangChain LLM response.

    Supports multiple LangChain response formats:
    - usage_metadata (newer LangChain style with input_tokens/output_tokens)
    - response_metadata.token_usage (older style with prompt_tokens/completion_tokens)

    Args:
        response: LangChain LLM response object (AIMessage, etc.)

    Returns:
        Dictionary with prompt_tokens, completion_tokens, total_tokens keys,
        or None if token usage is not available.

    Example:
        ```python
        response = llm.invoke("Hello")
        token_usage = extract_langchain_token_usage(response)
        if token_usage:
            print(f"Total tokens: {token_usage['total_tokens']}")
        ```
    """
    # Try newer LangChain style (usage_metadata with input_tokens/output_tokens)
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        return {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    # Try older LangChain style (response_metadata.token_usage)
    if hasattr(response, "response_metadata"):
        usage = response.response_metadata.get("token_usage", {})
        if usage:
            return {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

    return None


def aggregate_token_usage(
    current: dict[str, int] | None,
    new: dict[str, int] | None,
) -> dict[str, int] | None:
    """Aggregate token usage from multiple LLM calls.

    Combines token counts from multiple responses, useful for pipelines
    that make multiple LLM calls (e.g., IRCoT iterative reasoning).

    Args:
        current: Current aggregated token usage (or None).
        new: New token usage to add (or None).

    Returns:
        Aggregated token usage dict, or None if both inputs are None.

    Example:
        ```python
        total = None
        for response in responses:
            usage = extract_langchain_token_usage(response)
            total = aggregate_token_usage(total, usage)
        print(f"Total tokens across all calls: {total['total_tokens']}")
        ```
    """
    if current is None and new is None:
        return None
    if current is None:
        return new
    if new is None:
        return current

    return {key: current.get(key, 0) + new.get(key, 0) for key in {*current, *new}}


def extract_token_logprobs(
    response: Any,
    target_tokens: list[str] | None = None,
) -> dict[str, float] | None:
    """Extract log probabilities from LangChain LLM response.

    Works with any LangChain LLM that stores logprobs in ``response_metadata["logprobs"]["content"]``.
    Compatible providers include:
    - OpenAI (ChatOpenAI, AzureChatOpenAI)
    - Together AI, Fireworks AI, Anyscale
    - Local models via vLLM, text-generation-inference, Ollama (with logprobs enabled)
    - Any OpenAI-compatible API endpoint

    To enable logprobs, use provider-specific configuration:
    - OpenAI/vLLM: llm.bind(logprobs=True, top_logprobs=5)
    - Other providers: Check provider documentation for logprobs support

    LangChain stores logprobs in ``response.response_metadata["logprobs"]["content"]``.
    Each token entry has: token, logprob, bytes, top_logprobs.

    Args:
        response: LangChain AIMessage or similar response object.
        target_tokens: If provided, only return logprobs for these tokens.
            Case-insensitive matching. If None, returns all token logprobs.

    Returns:
        Dict mapping token strings to their log probability values.
        Returns None if logprobs not available in response.

    Example:
        >>> # Enable logprobs on the LLM (OpenAI example)
        >>> llm = ChatOpenAI(model="gpt-4o-mini").bind(logprobs=True, top_logprobs=5)
        >>> response = llm.invoke("Answer Yes or No: Is the sky blue?")
        >>> logprobs = extract_token_logprobs(response, target_tokens=["Yes", "No"])
        >>> # Returns: {"Yes": -0.0001, "No": -9.2} or None if not available

    Note:
        - log probability of 0.0 = 100% confidence
        - More negative = less likely
        - Convert to probability: exp(logprob)
    """
    # Check if response has response_metadata with logprobs
    if not hasattr(response, "response_metadata"):
        return None

    metadata = response.response_metadata
    if not isinstance(metadata, dict) or "logprobs" not in metadata:
        return None

    logprobs_data = metadata["logprobs"]
    if not isinstance(logprobs_data, dict) or "content" not in logprobs_data:
        return None

    content = logprobs_data["content"]
    if not content:
        return None

    result: dict[str, float] = {}

    # Build lowercase target tokens set for case-insensitive matching
    target_tokens_lower = {t.lower() for t in target_tokens} if target_tokens is not None else None

    # Extract logprobs from each token
    for token_data in content:
        token = token_data.get("token", "")
        logprob = token_data.get("logprob")
        top_logprobs = token_data.get("top_logprobs", [])

        if target_tokens_lower is not None:
            # Targeted extraction mode - find all target tokens
            _extract_target_logprobs(result, token, logprob, top_logprobs, target_tokens_lower)
        elif logprob is not None:
            # No target filter - return all tokens with valid logprobs
            result[token] = logprob

    return result if result else None


def _extract_target_logprobs(
    result: dict[str, float],
    token: str,
    logprob: float | None,
    top_logprobs: list[dict],
    target_tokens_lower: set[str],
) -> None:
    """Extract logprobs for target tokens from token data.

    Helper function to reduce complexity in extract_token_logprobs.
    """
    # Check if main token is a target
    if logprob is not None and token.lower() in target_tokens_lower:
        result[token] = logprob

    # Always check top_logprobs for other target tokens
    for alt in top_logprobs:
        alt_token = alt.get("token", "")
        alt_logprob = alt.get("logprob")
        # Only add if target token, has valid logprob, and not already present
        if alt_token.lower() in target_tokens_lower and alt_logprob is not None and alt_token not in result:
            result[alt_token] = alt_logprob


def image_chunk_to_pil_images(image_chunks: list[tuple[bytes, str]]) -> list[Image.Image]:
    """Convert raw image bytes to PIL Images, skipping invalid ones.

    Args:
        image_chunks: List of (bytes, mimetype) tuples.
            It can be a result of the GET operation from the ImageChunk repository.

    Returns:
        List of valid PIL Images.
    """
    images: list[Image.Image] = []
    for img_bytes, _mimetype in image_chunks:
        if img_bytes:
            try:
                img = bytes_to_pil_image(img_bytes)
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                images.append(img)
            except Exception:
                logger.debug("Skipping invalid image during VisRAG-Gen processing")
    return images
