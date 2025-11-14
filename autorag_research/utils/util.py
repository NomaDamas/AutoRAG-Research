import functools
import itertools
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import tiktoken
from pydantic import BaseModel as BM
from pydantic.v1 import BaseModel


def to_list(item):
    """Recursively convert collections to Python lists."""
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


def convert_inputs_to_list(func):
    """Decorator to convert all function inputs to Python lists."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = [to_list(arg) for arg in args]
        new_kwargs = {k: to_list(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapper


def truncate_texts(str_list: list[str], max_tokens: int) -> list[str]:
    """
    Truncate each string in the list to a maximum number of tokens using tiktoken.

    :param str_list: List of strings to be truncated.
    :param max_tokens: Maximum number of tokens allowed per string.
    :return: List of truncated strings.
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


def unpack_and_run(target_list: list[list[Any]], func: callable, *args, **kwargs) -> list[Any]:
    """
    Unpack each sublist in target_list and run func with the unpacked arguments.
    """
    lengths = list(map(len, target_list))
    flattened = list(itertools.chain.from_iterable(target_list))

    unpacked_results = func(flattened, *args, **kwargs)
    iterator = iter(unpacked_results)
    result = [list(itertools.islice(iterator, length)) for length in lengths]

    return result
