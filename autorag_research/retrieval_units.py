"""Shared retrieval-unit validation helpers."""

from typing import Literal, cast

RetrievalUnit = Literal["chunk", "image_chunk", "mixed"]
VALID_RETRIEVAL_UNITS: frozenset[str] = frozenset({"chunk", "image_chunk", "mixed"})


def coerce_retrieval_unit(value: object) -> RetrievalUnit | None:
    """Return a valid retrieval unit or None for missing values."""
    if value is None:
        return None
    if isinstance(value, str) and value in VALID_RETRIEVAL_UNITS:
        return cast("RetrievalUnit", value)
    return None


def _invalid_retrieval_unit_message(value: object) -> str:
    """Return a consistent error message for invalid retrieval-unit values."""
    valid_values = ", ".join(sorted(VALID_RETRIEVAL_UNITS))
    return f"Invalid retrieval_unit {value!r}. Expected one of: {valid_values}."


def require_retrieval_unit(value: object, *, default: RetrievalUnit | None = None) -> RetrievalUnit | None:
    """Return a valid retrieval unit, default only missing values, and reject explicit invalid values."""
    unit = coerce_retrieval_unit(value)
    if unit is not None:
        return unit
    if value is None:
        return default

    msg = _invalid_retrieval_unit_message(value)
    raise ValueError(msg)
