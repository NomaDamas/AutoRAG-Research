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
