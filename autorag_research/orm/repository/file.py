"""File repository for AutoRAG-Research.

Implements file-specific CRUD operations and queries extending
the generic repository pattern.
"""

from typing import Any

from sqlalchemy import ColumnElement, select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository.base import GenericRepository


class FileRepository(GenericRepository):
    """Repository for File entity with specialized queries."""

    def __init__(self, session: Session, model_cls: type | None = None):
        """Initialize file repository.

        Args:
            session: SQLAlchemy session for database operations.
            model_cls: The File model class. If None, uses default schema.
        """
        if model_cls is None:
            from autorag_research.orm.schema import File

            model_cls = File
        super().__init__(session, model_cls)

    def get_by_path(self, path: str) -> Any | None:
        """Retrieve a file by its path.

        Args:
            path: The file path to search for.

        Returns:
            The file if found, None otherwise.
        """
        stmt = select(self.model_cls).where(self.model_cls.path == path)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_type(self, file_type: str) -> list[Any]:
        """Retrieve all files of a specific type.

        Args:
            file_type: The file type (raw, image, audio, video).

        Returns:
            List of files of the specified type.
        """
        stmt = select(self.model_cls).where(self.model_cls.type == file_type)
        return list(self.session.execute(stmt).scalars().all())

    def get_with_documents(self, file_id: int) -> Any | None:
        """Retrieve a file with its documents eagerly loaded.

        Args:
            file_id: The file ID.

        Returns:
            The file with documents loaded, None if not found.
        """
        stmt = select(self.model_cls).where(self.model_cls.id == file_id).options(joinedload(self.model_cls.documents))
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_pages(self, file_id: int) -> Any | None:
        """Retrieve a file with its pages eagerly loaded.

        Args:
            file_id: The file ID.

        Returns:
            The file with pages loaded, None if not found.
        """
        stmt = select(self.model_cls).where(self.model_cls.id == file_id).options(joinedload(self.model_cls.pages))
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_image_chunks(self, file_id: int) -> Any | None:
        """Retrieve a file with its image chunks eagerly loaded.

        Args:
            file_id: The file ID.

        Returns:
            The file with image chunks loaded, None if not found.
        """
        stmt = (
            select(self.model_cls).where(self.model_cls.id == file_id).options(joinedload(self.model_cls.image_chunks))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_all_by_type(self, file_type: str, limit: int | None = None, offset: int | None = None) -> list[Any]:
        """Retrieve all files of a specific type with pagination.

        Args:
            file_type: The file type to filter by.
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of files of the specified type.
        """
        stmt = select(self.model_cls).where(self.model_cls.type == file_type)
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def search_by_path_pattern(self, pattern: str) -> list[Any]:
        """Search files by path pattern using SQL LIKE.

        Args:
            pattern: The pattern to search for (use % as wildcard).

        Returns:
            List of matching files.
        """
        stmt = select(self.model_cls).where(self.model_cls.path.like(pattern))
        return list(self.session.execute(stmt).scalars().all())

    def count_by_type(self, file_type: str) -> int:
        """Count the number of files of a specific type.

        Args:
            file_type: The file type to count.

        Returns:
            Number of files of the specified type.
        """
        return self.session.query(self.model_cls).filter(self.model_cls.type == file_type).count()

    def get_all_types(self) -> list[ColumnElement[Any]]:
        """Get all unique file types in the database.

        Returns:
            List of unique file types.
        """
        result = self.session.query(self.model_cls.type).distinct().all()
        return [row[0] for row in result]
