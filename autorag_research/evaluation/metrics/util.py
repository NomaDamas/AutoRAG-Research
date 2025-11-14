import functools

import numpy as np

from autorag_research.schema import MetricInput
from autorag_research.utils.util import convert_inputs_to_list


def calculate_cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        The cosine similarity score.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity


def calculate_l2_distance(a, b):
    """Calculate L2 distance between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        The L2 distance.
    """
    return np.linalg.norm(a - b)


def calculate_inner_product(a, b):
    """Calculate inner product between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        The inner product.
    """
    return np.dot(a, b)


def metric(fields_to_check: list[str]):
    """Decorator for AutoRAG metrics that processes each metric input individually.

    Use 'for loop' to run each metric input.
    Put the single metric input into the metric function.

    Args:
        fields_to_check: List of field names to check for validity.

    Returns:
        The decorated metric function.
    """

    def decorator_autorag_metric(func):
        @functools.wraps(func)
        @convert_inputs_to_list
        def wrapper(metric_inputs: list[MetricInput], **kwargs) -> list[float]:
            """Wrapper function for metric decorator.

            Args:
                metric_inputs: A list of MetricInput schema for AutoRAG metric.
                **kwargs: The additional arguments for metric function.

            Returns:
                A list of computed metric scores.
            """
            results = []
            for metric_input in metric_inputs:
                if metric_input.is_fields_notnone(fields_to_check=fields_to_check):
                    results.append(func(metric_input, **kwargs))
                else:
                    results.append(None)
            return results

        return wrapper

    return decorator_autorag_metric


def metric_loop(fields_to_check: list[str]):
    """Decorator for AutoRAG metrics that processes all metric inputs at once.

    Put the list of metric inputs into the metric function.

    Args:
        fields_to_check: List of field names to check for validity.

    Returns:
        The decorated metric function.
    """

    def decorator_metric_loop(func):
        @functools.wraps(func)
        @convert_inputs_to_list
        def wrapper(metric_inputs: list[MetricInput], **kwargs) -> list[float | None]:
            """Wrapper function for metric_loop decorator.

            Args:
                metric_inputs: A list of MetricInput schema for AutoRAG metric.
                **kwargs: The additional arguments for metric function.

            Returns:
                A list of computed metric scores.
            """
            bool_list: list[bool] = [
                metric_input.is_fields_notnone(fields_to_check=fields_to_check) for metric_input in metric_inputs
            ]
            valid_inputs = [
                metric_input for metric_input, is_valid in zip(metric_inputs, bool_list, strict=True) if is_valid
            ]

            results = [None] * len(metric_inputs)
            if valid_inputs:
                processed_valid = func(valid_inputs, **kwargs)

                valid_index = 0
                for i, is_valid in enumerate(bool_list):
                    if is_valid:
                        results[i] = processed_valid[valid_index]
                        valid_index += 1

            return results

        return wrapper

    return decorator_metric_loop
