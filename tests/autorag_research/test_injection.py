from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.mock_embed_model import MockEmbedding

from autorag_research import cli
from autorag_research.injection import (
    _embedding_model_cache,
    clear_embedding_cache,
    get_cached_embedding_model,
    health_check_embedding,
    load_embedding_model,
    with_embedding,
)

cli.CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs"


class TestLoadEmbeddingModel:
    """Tests for load_embedding_model function."""

    def test_raises_file_not_found_for_missing_config(self) -> None:
        """Raises FileNotFoundError when config doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_embedding_model("nonexistent")

    @patch("hydra.utils.instantiate")
    def test_raises_type_error_for_wrong_type(self, mock_instantiate: MagicMock) -> None:
        """Raises TypeError when instantiated object is not BaseEmbedding."""
        mock_instantiate.return_value = "not an embedding"

        with pytest.raises(TypeError, match="BaseEmbedding"):
            load_embedding_model("openai-small")

    def test_returns_embedding_instance(self) -> None:
        """Returns BaseEmbedding instance when config is valid."""
        from llama_index.core.base.embeddings.base import BaseEmbedding

        result = load_embedding_model("mock")

        assert isinstance(result, BaseEmbedding)


class TestHealthCheckEmbedding:
    """Tests for health_check_embedding function.

    Uses mock embedding model to avoid real API calls.
    """

    def test_returns_dimension_on_success(self) -> None:
        """Returns embedding dimension on success."""
        from llama_index.core.embeddings.mock_embed_model import MockEmbedding

        mock_embedding = MockEmbedding(384)

        result = health_check_embedding(mock_embedding)

        assert result == 384

    def test_raises_on_embedding_failure(self) -> None:
        """Raises EmbeddingNotSetError when embedding fails."""
        from autorag_research.exceptions import EmbeddingNotSetError

        mock_model = MagicMock()
        mock_model.get_text_embedding.side_effect = Exception("API Error")

        with pytest.raises(EmbeddingNotSetError):
            health_check_embedding(mock_model)


class TestGetCachedEmbeddingModel:
    """Tests for get_cached_embedding_model function."""

    def test_caches_embedding_model(self) -> None:
        """Same config name returns same cached instance."""

        clear_embedding_cache()

        model1 = get_cached_embedding_model("mock")
        model2 = get_cached_embedding_model("mock")

        assert model1 is model2
        assert "mock" in _embedding_model_cache
        clear_embedding_cache()

    def test_clear_cache_removes_all(self) -> None:
        """clear_embedding_cache removes all cached models."""
        clear_embedding_cache()
        get_cached_embedding_model("mock")
        assert len(_embedding_model_cache) == 1

        clear_embedding_cache()
        assert len(_embedding_model_cache) == 0


class TestWithEmbeddingDecorator:
    """Tests for with_embedding decorator."""

    def test_string_to_instance_conversion(self) -> None:
        """Decorator converts string config name to BaseEmbedding instance."""

        clear_embedding_cache()

        @with_embedding()
        def my_func(embedding_model: BaseEmbedding | str) -> BaseEmbedding:
            return embedding_model  # type: ignore[return-value]

        result = my_func(embedding_model="mock")
        assert isinstance(result, BaseEmbedding)
        clear_embedding_cache()

    def test_instance_passthrough(self) -> None:
        """Decorator passes through BaseEmbedding instances unchanged."""
        from llama_index.core.embeddings.mock_embed_model import MockEmbedding

        @with_embedding()
        def my_func(embedding_model) -> object:
            return embedding_model

        mock_model = MockEmbedding(embed_dim=384)
        result = my_func(embedding_model=mock_model)
        assert result is mock_model

    def test_caching_same_config(self) -> None:
        """Decorator uses cached model for same config name."""
        clear_embedding_cache()
        results = []

        @with_embedding()
        def my_func(embedding_model) -> object:
            return embedding_model

        results.append(my_func(embedding_model="mock"))
        results.append(my_func(embedding_model="mock"))

        assert results[0] is results[1]
        clear_embedding_cache()

    def test_invalid_type_raises_error(self) -> None:
        """Decorator raises TypeError for invalid embedding_model type."""

        @with_embedding()
        def my_func(embedding_model) -> None:
            pass

        with pytest.raises(TypeError, match="must be string, BaseEmbedding or MultiVectorBaseEmbedding"):
            my_func(embedding_model=123)  # type: ignore[arg-type]

    def test_invalid_param_name_raises_error(self) -> None:
        """Decorator raises ValueError when param_name doesn't exist."""
        with pytest.raises(ValueError, match="Parameter 'nonexistent' not found"):

            @with_embedding(param_name="nonexistent")
            def my_func(embedding_model) -> None:
                pass

    def test_custom_param_name(self) -> None:
        """Decorator works with custom param_name."""

        @with_embedding(param_name="model")
        @with_embedding("model2")
        def my_func(model, model2, ho) -> object:
            return model, model2, ho

        mock_model = MockEmbedding(embed_dim=384)
        mock_model2 = MockEmbedding(embed_dim=768)
        result1, result2, result3 = my_func(model=mock_model, model2=mock_model2, ho=3)
        assert result1 is mock_model
        assert result2 is mock_model2
        assert result3 == 3
