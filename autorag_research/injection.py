import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from hydra.utils import instantiate
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from omegaconf import OmegaConf

from autorag_research.cli.utils import get_config_dir
from autorag_research.embeddings.base import MultiVectorBaseEmbedding
from autorag_research.rerankers.base import BaseReranker

logger = logging.getLogger("AutoRAG-Research")

EMBEDDING_MODEL_TYPES = Embeddings | MultiVectorBaseEmbedding
RERANKER_MODEL_TYPES = BaseReranker

T = TypeVar("T")


def health_check_embedding(model: EMBEDDING_MODEL_TYPES) -> int:
    """Health check embedding model and return embedding dimension.

    Args:
        model: LangChain Embeddings or MultiVectorBaseEmbedding instance.

    Returns:
        Embedding dimension (length of embedding vector).

    Raises:
        EmbeddingError: If health check fails.
    """
    from autorag_research.exceptions import EmbeddingError

    try:
        embedding = model.embed_query("health check")
        # For multi-vector models, embedding is list[list[float]], get the inner dimension
        if embedding and isinstance(embedding[0], list):
            return len(embedding[0])
        return len(embedding)
    except Exception as e:
        raise EmbeddingError from e


def health_check_llm(model: BaseLanguageModel) -> None:
    """Health check LLM by making a test call.

    Args:
        model: LangChain BaseLanguageModel instance.

    Raises:
        LLMError: If health check fails.
    """
    from autorag_research.exceptions import LLMError

    try:
        model.invoke("Hello, world!")
    except Exception as e:
        raise LLMError from e


def health_check_reranker(model: BaseReranker) -> None:
    """Health check reranker model by making a test rerank call.

    Args:
        model: BaseReranker instance.

    Raises:
        RerankerError: If health check fails.
    """
    from autorag_research.exceptions import RerankerError

    try:
        model.rerank("test query", ["document 1", "document 2"], top_k=1)
    except Exception as e:
        raise RerankerError from e


class ModelManager(Generic[T]):
    """Generic model cache manager for embedding models and LLMs."""

    def __init__(
        self,
        config_subdir: str,
        expected_types: tuple[type, ...],
        health_check_func: Callable[[Any], Any],
    ) -> None:
        """Initialize the model manager.

        Args:
            config_subdir: Subdirectory under config dir (e.g., "embedding", "llm").
            expected_types: Tuple of valid types for type checking.
        """
        self._config_subdir = config_subdir
        self._expected_types = expected_types
        self._cache: dict[str, T] = {}
        self._health_check_func = health_check_func

    @property
    def _type_name(self) -> str:
        """Generate human-readable type name from expected_types."""
        return " or ".join(t.__name__ for t in self._expected_types)

    def load(self, config_name: str) -> T:
        """Load model from YAML config (no caching).

        Args:
            config_name: Name of the config file (without .yaml extension).

        Returns:
            Instantiated model instance.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            TypeError: If instantiated object is not of expected type.
        """
        yaml_path = get_config_dir() / self._config_subdir / f"{config_name}.yaml"
        if not yaml_path.exists():
            yaml_path = get_config_dir() / self._config_subdir / f"{config_name}.yml"
        if not yaml_path.exists():
            msg = f"Config file not found: {yaml_path}"
            raise FileNotFoundError(msg)

        cfg = OmegaConf.load(yaml_path)
        model = instantiate(cfg)

        if not isinstance(model, self._expected_types):
            raise TypeError(f"Expected {self._type_name}, got {type(model)}")  # noqa: TRY003

        self._health_check_func(model)

        return model

    def get_cached(self, config_name: str) -> T:
        """Get or create a cached model instance.

        Args:
            config_name: Name of the config file (without .yaml extension).

        Returns:
            Cached model instance.
        """
        if config_name not in self._cache:
            self._cache[config_name] = self.load(config_name)
            logger.debug(f"Created and cached {self._config_subdir} model: {config_name}")
        return self._cache[config_name]

    def clear_cache(self) -> None:
        """Clear all cached instances."""
        self._cache.clear()


def _create_with_model_decorator(
    manager: ModelManager[T],
    valid_types: tuple[type, ...],
    param_name: str,
) -> Callable:
    """Factory for creating model injection decorators.

    Args:
        manager: The ModelManager instance to use for caching.
        valid_types: Tuple of valid types for passthrough.
        param_name: Parameter name to inject.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        if param_name not in sig.parameters:
            func_name = getattr(func, "__name__", repr(func))
            raise ValueError(f"Parameter '{param_name}' not found in '{func_name}'")  # noqa: TRY003

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            if param_name in bound.arguments:
                value = bound.arguments[param_name]
                if isinstance(value, str):
                    bound.arguments[param_name] = manager.get_cached(value)
                elif not isinstance(value, valid_types):
                    type_names = " or ".join(t.__name__ for t in valid_types)
                    raise TypeError(  # noqa: TRY003
                        f"'{param_name}' must be string, {type_names}, got {type(value).__name__}"
                    )
            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


# Manager instances
_embedding_manager: ModelManager[EMBEDDING_MODEL_TYPES] = ModelManager(
    config_subdir="embedding",
    expected_types=(Embeddings, MultiVectorBaseEmbedding),
    health_check_func=health_check_embedding,
)

_llm_manager: ModelManager[BaseLanguageModel] = ModelManager(
    config_subdir="llm",
    expected_types=(BaseLanguageModel,),
    health_check_func=health_check_llm,
)

_reranker_manager: ModelManager[BaseReranker] = ModelManager(
    config_subdir="reranker",
    expected_types=(BaseReranker,),
    health_check_func=health_check_reranker,
)


# ============================================================================
# Embedding Functions (existing API preserved)
# ============================================================================


def load_embedding_model(config_name: str) -> EMBEDDING_MODEL_TYPES:
    """Load LangChain embedding model directly from YAML via Hydra instantiate.

    Args:
        config_name: Name of the embedding config file (without .yaml extension).
                    e.g., "openai-small", "openai-large", "openai-like"

    Returns:
        LangChain Embeddings or MultiVectorBaseEmbedding instance.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    return _embedding_manager.load(config_name)


def with_embedding(param_name: str = "embedding_model"):
    """Decorator that injects cached embedding model instances.

    Args:
        param_name: Parameter name to inject. Defaults to "embedding_model".

    Behavior:
        - String config name → cached Embeddings instance (via load_embedding_model)
        - Embeddings instance → pass through unchanged

    Example:
        @with_embedding()
        def my_func(embedding_model: Embeddings | str):
            ...

        # Can be called with string (converted to cached instance):
        my_func(embedding_model="openai-large")

        # Or with instance (passed through):
        my_func(embedding_model=some_embedding_instance)

    Note:
        Uses existing load_embedding_model() which handles OmegaConf + Hydra instantiate.
    """
    return _create_with_model_decorator(
        manager=_embedding_manager,
        valid_types=(Embeddings, MultiVectorBaseEmbedding),
        param_name=param_name,
    )


# ============================================================================
# LLM Functions (parallel API to embedding)
# ============================================================================


def load_llm(config_name: str) -> BaseLanguageModel:
    """Load LangChain LLM directly from YAML via Hydra instantiate.

    Args:
        config_name: Name of the LLM config file (without .yaml extension).
                    e.g., "openai-gpt4", "mock"

    Returns:
        LangChain BaseLanguageModel instance.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        TypeError: If instantiated object is not a BaseLanguageModel.
    """
    return _llm_manager.load(config_name)


def with_llm(param_name: str = "llm"):
    """Decorator that injects cached LLM instances.

    Args:
        param_name: Parameter name to inject. Defaults to "llm".

    Behavior:
        - String config name → cached BaseLanguageModel instance (via get_cached_llm)
        - BaseLanguageModel instance → pass through unchanged

    Example:
        @with_llm()
        def my_func(llm: BaseLanguageModel | str):
            ...

        # Can be called with string (converted to cached instance):
        my_func(llm="openai-gpt4")

        # Or with instance (passed through):
        my_func(llm=some_llm_instance)
    """
    return _create_with_model_decorator(
        manager=_llm_manager,
        valid_types=(BaseLanguageModel,),
        param_name=param_name,
    )


# ============================================================================
# Reranker Functions (parallel API to embedding/llm)
# ============================================================================


def load_reranker(config_name: str) -> BaseReranker:
    """Load reranker model directly from YAML via Hydra instantiate.

    Args:
        config_name: Name of the reranker config file (without .yaml extension).
                    e.g., "cohere", "jina", "voyageai"

    Returns:
        BaseReranker instance.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        TypeError: If instantiated object is not a BaseReranker.
    """
    return _reranker_manager.load(config_name)


def with_reranker(param_name: str = "reranker"):
    """Decorator that injects cached reranker instances.

    Args:
        param_name: Parameter name to inject. Defaults to "reranker".

    Behavior:
        - String config name → cached BaseReranker instance (via load_reranker)
        - BaseReranker instance → pass through unchanged

    Example:
        @with_reranker()
        def my_func(reranker: BaseReranker | str):
            ...

        # Can be called with string (converted to cached instance):
        my_func(reranker="cohere")

        # Or with instance (passed through):
        my_func(reranker=some_reranker_instance)
    """
    return _create_with_model_decorator(
        manager=_reranker_manager,
        valid_types=(BaseReranker,),
        param_name=param_name,
    )
