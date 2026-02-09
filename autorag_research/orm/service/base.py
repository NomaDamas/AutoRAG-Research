from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.exceptions import SessionNotSetError


class BaseService(ABC):
    """Abstract base class for all service implementations.

    Provides common patterns for all services:
    - Session factory and schema management
    - Abstract methods for UoW creation and schema class retrieval
    - Helper method for adding objects to database
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        schema: Any | None = None,
    ):
        """Initialize the service.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            schema: Schema namespace from create_schema(). If None, uses default 768-dim schema.
        """
        self.session_factory = session_factory
        self._schema = schema

    @abstractmethod
    def _create_uow(self) -> Any:
        """Create a new Unit of Work instance.

        Subclasses must implement this to return their specific UoW type.
        """
        ...

    @abstractmethod
    def _get_schema_classes(self) -> dict[str, Any]:
        """Get schema classes from the schema namespace.

        Returns:
            Dictionary mapping class names to ORM classes.
        """
        ...

    def _add(self, obj: list[dict], table_name: str, repository_property: str) -> list[int]:
        """
        Add objects to the specified table.

        Args:
            obj: The object(s) to add.
                It should be a list of dictionaries representing the records to insert.
            table_name: The table name.
                It should be the same as the class name in the schema.
                For example, "Document", "ImageChunk", "Chunk", etc.
            repository_property: The repository property name in the UoW.
                It should be the same as the repository property name in the UoW.

        Returns:
            The id of the added object(s).
        """
        classes = self._get_schema_classes()
        cls = classes.get(table_name)
        if cls is None:
            raise ValueError(f"Table '{table_name}' not found in schema.")  # noqa: TRY003

        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            entities = [cls(**item) for item in obj]
            getattr(uow, repository_property).add_all(entities)
            uow.flush()
            ids = [entity.id for entity in entities]
            uow.commit()
            return ids

    def _add_bulk(self, items: list[dict], repository_property: str, skip_duplicates: bool = False) -> list[int | str]:
        """Memory-efficient bulk insert using repository's add_bulk method.

        Unlike _add(), this method does not create ORM objects in Python memory.
        Instead, it uses SQLAlchemy Core's insert() which generates a single
        multi-row INSERT statement, significantly reducing memory usage and
        improving performance for large batch inserts.

        Args:
            items: List of dictionaries representing records to insert.
            repository_property: The repository property name in the UoW (e.g., "chunks", "queries").
            skip_duplicates: If True, uses ON CONFLICT DO NOTHING to skip rows with
                duplicate primary keys instead of raising an IntegrityError.

        Returns:
            List of inserted IDs (excludes skipped duplicates when skip_duplicates=True).

        Note:
            For 1000 records, this method uses ~3-5x less memory than _add()
            because it bypasses ORM object creation and identity map tracking.
        """
        if not items:
            return []

        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            repository = getattr(uow, repository_property)
            ids = repository.add_bulk_skip_duplicates(items) if skip_duplicates else repository.add_bulk(items)
            uow.commit()
            return ids
