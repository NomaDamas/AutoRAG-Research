from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MetricInput:
    query: str | None = None
    queries: list[str] | None = None
    retrieval_gt_contents: list[list[str]] | None = None
    retrieved_contents: list[str] | None = None
    retrieval_gt: list[list[str]] | None = None
    retrieved_ids: list[str] | None = None
    relevance_scores: dict[str, int] | None = None  # Maps prefixed_id -> graded relevance score
    prompt: str | None = None
    generated_texts: str | None = None
    generation_gt: list[str] | None = None
    generated_log_probs: list[float] | None = None

    def is_fields_notnone(self, fields_to_check: list[str]) -> bool:
        for field in fields_to_check:
            actual_value = getattr(self, field)

            if actual_value is None:
                return False

            try:
                if not type_checks.get(type(actual_value), lambda _: False)(actual_value):
                    return False
            except Exception:
                return False

        return True

    @classmethod
    def from_dataframe(cls, qa_data: pd.DataFrame) -> list["MetricInput"]:
        """
        Convert a pandas DataFrame into a list of MetricInput instances.
        qa_data: pd.DataFrame: qa_data DataFrame containing metric data.

        :returns: List[MetricInput]: List of MetricInput objects created from DataFrame rows.
        """
        instances = []

        for _, row in qa_data.iterrows():
            instance = cls()

            for attr_name in cls.__annotations__:
                if attr_name in row:
                    value = row[attr_name]

                    if isinstance(value, str):
                        setattr(
                            instance,
                            attr_name,
                            value.strip() if value.strip() != "" else None,
                        )
                    elif isinstance(value, list):
                        setattr(instance, attr_name, value if len(value) > 0 else None)
                    else:
                        setattr(instance, attr_name, value)

            instances.append(instance)

        return instances

    @staticmethod
    def _check_list(lst_or_arr: list[Any] | np.ndarray) -> bool:
        if isinstance(lst_or_arr, np.ndarray):
            lst_or_arr = lst_or_arr.flatten().tolist()  # ty: ignore[invalid-argument-type]

        if len(lst_or_arr) == 0:
            return False

        for item in lst_or_arr:
            if item is None:
                return False

            item_type = type(item)

            if item_type in type_checks:
                if not type_checks[item_type](item):
                    return False
            else:
                return False

        return True


type_checks: dict[type, Callable[[Any], bool]] = {
    str: lambda x: len(x.strip()) > 0,
    list: MetricInput._check_list,
    np.ndarray: MetricInput._check_list,
    int: lambda _: True,
    float: lambda _: True,
}
