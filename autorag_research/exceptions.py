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


class EmbeddingError(Exception):
    """Raised when the embedding model is not set."""

    def __init__(self):
        super().__init__(
            "We don't know what is wrong, but your Embedding model is criminal. Scroll up to see the traceback and find your real reason."
        )


class LLMError(Exception):
    """Raised when the LLM model is not set."""

    def __init__(self):
        super().__init__(
            "We don't know what is wrong, but your LLM model is criminal. Scroll up to see the traceback and find your real reason."
        )


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


class SchemaNotFoundError(Exception):
    """Raised when a schema is not found."""

    def __init__(self, schema_name: str):
        super().__init__(f"Schema '{schema_name}' not found.")


class EmptyIterableError(Exception):
    """Raised when an iterable is empty but should contain items."""

    def __init__(self, iterable_name: str):
        super().__init__(f"The iterable '{iterable_name}' is empty but should contain items.")


class DuplicateRetrievalGTError(Exception):
    """Raised when retrieval GT already exists for a query and upsert is False."""

    def __init__(self, query_ids: list[int | str]):
        ids_str = ", ".join(str(qid) for qid in query_ids)
        super().__init__(f"Retrieval GT already exists for query IDs: {ids_str}. Use upsert=True to overwrite.")


class MissingRequiredParameterError(Exception):
    """Raised when required parameters are missing."""

    def __init__(self, param_names: list[str]):
        params_str = ", ".join(f"'{p}'" for p in param_names)
        super().__init__(f"At least one of the following parameters must be provided: {params_str}.")


class ServiceNotSetError(Exception):
    """Raised when the service is not set."""

    def __init__(self):
        super().__init__("Service is not set.")


class NoQueryInDBError(Exception):
    """Raised when there are no queries in the database."""

    def __init__(self):
        super().__init__("No queries found in the database.")


# Executor exceptions


class ExecutorError(Exception):
    """Base exception for Executor errors."""

    pass


class PipelineExecutionError(ExecutorError):
    """Raised when a pipeline fails to execute."""

    def __init__(self, pipeline_name: str, reason: str):
        super().__init__(f"Pipeline '{pipeline_name}' failed: {reason}")
        self.pipeline_name = pipeline_name
        self.reason = reason


class PipelineVerificationError(ExecutorError):
    """Raised when pipeline results fail verification."""

    def __init__(self, pipeline_name: str, expected: int, actual: int):
        super().__init__(f"Pipeline '{pipeline_name}' verification failed: expected {expected} results, got {actual}")
        self.pipeline_name = pipeline_name
        self.expected = expected
        self.actual = actual


class MaxRetriesExceededError(ExecutorError):
    """Raised when max retries are exceeded."""

    def __init__(self, pipeline_name: str, max_retries: int):
        super().__init__(f"Pipeline '{pipeline_name}' failed after {max_retries} retries")
        self.pipeline_name = pipeline_name
        self.max_retries = max_retries


class EvaluationError(ExecutorError):
    """Raised when evaluation fails."""

    def __init__(self, metric_name: str, pipeline_id: int, reason: str):
        super().__init__(f"Evaluation failed for metric '{metric_name}' on pipeline {pipeline_id}: {reason}")
        self.metric_name = metric_name
        self.pipeline_id = pipeline_id
        self.reason = reason


# MTEB exceptions
class UnsupportedMTEBTaskTypeError(Exception):
    """Raised when an unsupported MTEB task type is provided."""

    def __init__(self, task_name: str, task_type: str, supported_task_types: list[str]):
        super().__init__(
            f"Task '{task_name}' has type '{task_type}' which is not supported. "
            f"Supported types: {', '.join(sorted(supported_task_types))}"
        )


class UnsupportedLanguageError(Exception):
    """Raised when an unsupported language is specified."""

    def __init__(self, language_code: str, supported_languages: list[str] | None = None):
        if supported_languages:
            languages_str = ", ".join(supported_languages)
            super().__init__(
                f"Unsupported language code '{language_code}' specified. Supported languages are: {languages_str}."
            )
        else:
            super().__init__(f"Unsupported language code '{language_code}' specified.")


class MissingDBNameError(Exception):
    """Raised when the database name is missing in the configuration."""

    def __init__(self):
        super().__init__("Database name is missing in the configuration.")
