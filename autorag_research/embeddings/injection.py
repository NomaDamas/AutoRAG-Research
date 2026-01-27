import logging

from llama_index.core.base.embeddings.base import BaseEmbedding
from omegaconf import OmegaConf

from autorag_research.cli.utils import get_config_dir

logger = logging.getLogger("AutoRAG-Research")

# Module-level cache for embedding model instances
_embedding_model_cache: dict[str, "BaseEmbedding"] = {}


def load_embedding_model(config_name: str) -> "BaseEmbedding":
    """Load LlamaIndex embedding model directly from YAML via Hydra instantiate.

    Args:
        config_name: Name of the embedding config file (without .yaml extension).
                    e.g., "openai-small", "openai-large", "openai-like"

    Returns:
        LlamaIndex BaseEmbedding instance.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    from hydra.utils import instantiate

    yaml_path = get_config_dir() / "embedding" / f"{config_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError

    cfg = OmegaConf.load(yaml_path)
    model = instantiate(cfg)

    if not isinstance(model, BaseEmbedding):
        raise TypeError(f"Expected BaseEmbedding, got {type(model)}")  # noqa: TRY003

    return model


def get_cached_embedding_model(config_name: str) -> "BaseEmbedding":
    """Get or create a cached embedding model instance.

    Args:
        config_name: Name of the embedding config file (without .yaml extension).

    Returns:
        Cached BaseEmbedding instance.
    """
    if config_name not in _embedding_model_cache:
        _embedding_model_cache[config_name] = load_embedding_model(config_name)
        logger.debug(f"Created and cached embedding model: {config_name}")
    return _embedding_model_cache[config_name]


def clear_embedding_cache() -> None:
    """Clear the embedding model cache (for testing)."""
    _embedding_model_cache.clear()


def with_embedding(param_name: str = "embedding_model"):
    """Decorator that injects cached embedding model instances.

    Args:
        param_name: Parameter name to inject. Defaults to "embedding_model".

    Behavior:
        - String config name → cached BaseEmbedding instance (via load_embedding_model)
        - BaseEmbedding instance → pass through unchanged

    Example:
        @with_embedding()
        def my_func(embedding_model: BaseEmbedding | str):
            ...

        # Can be called with string (converted to cached instance):
        my_func(embedding_model="openai-large")

        # Or with instance (passed through):
        my_func(embedding_model=some_embedding_instance)

    Note:
        Uses existing load_embedding_model() which handles OmegaConf + Hydra instantiate.
    """
    import functools
    import inspect

    def decorator(func):
        sig = inspect.signature(func)
        if param_name not in sig.parameters:
            raise ValueError(f"Parameter '{param_name}' not found in '{func.__name__}'")  # noqa: TRY003

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            if param_name in bound.arguments:
                value = bound.arguments[param_name]
                if isinstance(value, str):
                    bound.arguments[param_name] = get_cached_embedding_model(value)
                elif not isinstance(value, BaseEmbedding):
                    raise TypeError(f"'{param_name}' must be string or BaseEmbedding, got {type(value).__name__}")  # noqa: TRY003
                # BaseEmbedding instance: pass through unchanged
            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


def health_check_embedding(model: "BaseEmbedding") -> int:
    """Health check embedding model and return embedding dimension.

    Args:
        model: LlamaIndex BaseEmbedding instance.

    Returns:
        Embedding dimension (length of embedding vector).

    Raises:
        EmbeddingNotSetError: If health check fails.
    """
    from autorag_research.exceptions import EmbeddingNotSetError

    try:
        embedding = model.get_text_embedding("health check")
        return len(embedding)
    except Exception as e:
        raise EmbeddingNotSetError from e
