class EnvNotFoundError(Exception):
    """Raised when a required environment variable is not found."""

    def __init__(self, env_var_name: str):
        super().__init__(f"Environment variable '{env_var_name}' not found.")


class NoSessionError(Exception):
    """Raised when there is no active database session."""

    def __init__(self):
        super().__init__("No active database session found.")


class UnsupportedDataSubsetError(Exception):
    """Raised when an unsupported data subset is requested."""

    def __init__(self, subsets: list[str]):
        subsets_str = ", ".join(subsets)
        super().__init__(f"Data subset '{subsets_str}' are not supported.")


class EmbeddingNotSetError(Exception):
    """Raised when the embedding model is not set."""

    def __init__(self):
        super().__init__("Embedding model is not set.")


class SessionNotSetError(Exception):
    """Raised when the database session is not set."""

    def __init__(self):
        super().__init__("Database session is not set.")


class LengthMismatchError(Exception):
    """Raised when there is a length mismatch between two related lists."""

    def __init__(self, list1_name: str, list2_name: str):
        super().__init__(f"Length mismatch between '{list1_name}' and '{list2_name}'.")


class InvalidDatasetNameError(NameError):
    """Raised when an invalid dataset name is provided."""

    def __init__(self, dataset_name: str):
        super().__init__(f"Invalid dataset name '{dataset_name}' provided.")


class RepositoryNotSupportedError(Exception):
    """Raised when a repository is not supported by the current UoW."""

    def __init__(self, repository_name: str, uow_type: str):
        super().__init__(f"Repository '{repository_name}' is not supported by '{uow_type}'.")


class EmptyIterableError(Exception):
    """Raised when an iterable is empty but should contain items."""

    def __init__(self, iterable_name: str):
        super().__init__(f"The iterable '{iterable_name}' is empty but should contain items.")


class DuplicateRetrievalGTError(Exception):
    """Raised when retrieval GT already exists for a query and upsert is False."""

    def __init__(self, query_ids: list[int]):
        ids_str = ", ".join(str(qid) for qid in query_ids)
        super().__init__(f"Retrieval GT already exists for query IDs: {ids_str}. Use upsert=True to overwrite.")
