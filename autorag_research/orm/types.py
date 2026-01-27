"""Custom SQLAlchemy types for AutoRAG-Research.

Provides custom types for handling PostgreSQL vector arrays with VectorChord.
"""

import re
from typing import Any

from sqlalchemy import Dialect, TypeDecorator
from sqlalchemy.types import UserDefinedType


class VectorArrayType(UserDefinedType):
    """Custom SQLAlchemy type for array of vectors compatible with VectorChord.

    This type handles serialization/deserialization of multi-vector embeddings
    stored as PostgreSQL `vector(dim)[]` arrays. It bypasses pgvector's default
    array processing which fails with nested arrays.

    The type works with VectorChord's MaxSim operator (@#) for late interaction
    retrieval models like ColPali/ColBERT.

    Usage:
        embeddings: Mapped[list[list[float]] | None] = mapped_column(VectorArrayType(768))

    Database representation:
        VECTOR(768)[] - Array of 768-dimensional vectors
    """

    cache_ok = True

    def __init__(self, dim: int):
        """Initialize VectorArrayType with vector dimension.

        Args:
            dim: The dimension of each vector in the array.
        """
        self.dim = dim

    def get_col_spec(self, **kw: Any) -> str:
        """Return the column specification for PostgreSQL.

        Args:
            **kw: Additional keyword arguments (unused, for SQLAlchemy compatibility).

        Returns:
            PostgreSQL type specification string.
        """
        return f"VECTOR({self.dim})[]"

    def bind_processor(self, dialect: Dialect) -> Any:
        """Return a processor for binding Python values to database.

        Converts Python list[list[float]] to PostgreSQL array literal format:
        ARRAY['[1.0,2.0,3.0]'::vector(dim), '[4.0,5.0,6.0]'::vector(dim)]

        Args:
            dialect: The SQLAlchemy dialect.

        Returns:
            A callable that processes values for database insertion.
        """

        def process(value: list[list[float]] | None) -> str | None:
            if value is None:
                return None
            if not value:
                return "{}"

            # Convert each vector to PostgreSQL vector literal format
            vector_literals = []
            for vec in value:
                # Format: "[1.0,2.0,3.0]"
                vec_str = "[" + ",".join(str(float(x)) for x in vec) + "]"
                vector_literals.append(f'"{vec_str}"')

            # Return PostgreSQL array literal: {"[1,2,3]","[4,5,6]"}
            return "{" + ",".join(vector_literals) + "}"

        return process

    def result_processor(self, dialect: Dialect, coltype: Any) -> Any:
        """Return a processor for converting database values to Python.

        Parses PostgreSQL vector array format back to Python list[list[float]].

        Args:
            dialect: The SQLAlchemy dialect.
            coltype: The column type.

        Returns:
            A callable that processes database values to Python objects.
        """

        def process(value: Any) -> list[list[float]] | None:
            if value is None:
                return None

            # If already a list (psycopg may return pre-parsed), convert to float lists
            if isinstance(value, list):
                result = []
                for vec in value:
                    if isinstance(vec, (list, tuple)):
                        result.append([float(x) for x in vec])
                    elif hasattr(vec, "tolist"):
                        result.append([float(x) for x in vec.tolist()])
                    else:
                        result.append([float(x) for x in vec])
                return result

            # Handle string format from PostgreSQL: {"[1,2,3]","[4,5,6]"}
            if isinstance(value, str):
                if value == "{}" or value == "{}":
                    return []

                result = []
                vector_pattern = r"\[([^\]]+)\]"
                matches = re.findall(vector_pattern, value)

                for match in matches:
                    floats = [float(x.strip()) for x in match.split(",")]
                    result.append(floats)

                return result

            return None

        return process

    def bind_expression(self, bindvalue: Any) -> Any:
        """Generate a SQL expression for binding.

        This ensures proper casting to vector[] type.

        Args:
            bindvalue: The bound value.

        Returns:
            SQL expression with proper casting.
        """
        from sqlalchemy import type_coerce

        return type_coerce(bindvalue, self)


class VectorArray(TypeDecorator):
    """TypeDecorator wrapper for VectorArrayType.

    This provides a simpler interface while delegating to VectorArrayType
    for the actual serialization/deserialization logic.

    Usage:
        embeddings: Mapped[list[list[float]] | None] = mapped_column(VectorArray(768))
    """

    impl = VectorArrayType
    cache_ok = True

    def __init__(self, dim: int):
        """Initialize VectorArray with vector dimension.

        Args:
            dim: The dimension of each vector in the array.
        """
        self.dim = dim
        super().__init__(dim)

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        """Return the dialect-specific implementation.

        Args:
            dialect: The SQLAlchemy dialect.

        Returns:
            The dialect-specific type implementation.
        """
        return dialect.type_descriptor(VectorArrayType(self.dim))

    def process_bind_param(self, value: list[list[float]] | None, dialect: Dialect) -> str | None:
        """Process a bound parameter value.

        Args:
            value: The Python value to convert.
            dialect: The SQLAlchemy dialect.

        Returns:
            The converted database value.
        """
        if value is None:
            return None
        if not value:
            return "{}"

        # Convert each vector to PostgreSQL vector literal format
        vector_literals = []
        for vec in value:
            vec_str = "[" + ",".join(str(float(x)) for x in vec) + "]"
            vector_literals.append(f'"{vec_str}"')

        return "{" + ",".join(vector_literals) + "}"

    def process_result_value(self, value: Any, dialect: Dialect) -> list[list[float]] | None:
        """Process a result value from the database.

        Args:
            value: The database value to convert (string or already-parsed list).
            dialect: The SQLAlchemy dialect.

        Returns:
            The converted Python value.
        """
        if value is None:
            return None

        # If already a list (psycopg may return pre-parsed), convert to float lists
        if isinstance(value, list):
            result = []
            for vec in value:
                if isinstance(vec, (list, tuple)):
                    # Already a sequence of numbers
                    result.append([float(x) for x in vec])
                elif hasattr(vec, "tolist"):
                    # numpy array
                    result.append([float(x) for x in vec.tolist()])
                else:
                    # Assume it's a pgvector Vector object or similar
                    result.append([float(x) for x in vec])
            return result

        # Handle string format from PostgreSQL: {"[1,2,3]","[4,5,6]"}
        if isinstance(value, str):
            if value == "{}" or value == "{}":
                return []

            result = []
            vector_pattern = r"\[([^\]]+)\]"
            matches = re.findall(vector_pattern, value)

            for match in matches:
                floats = [float(x.strip()) for x in match.split(",")]
                result.append(floats)

            return result

        return None
