import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.llms.mock import MockLLM

from autorag_research import cli
from autorag_research.injection import (
    _llm_manager,
    health_check_embedding,
    health_check_llm,
    load_embedding_model,
    load_llm,
    with_embedding,
    with_llm,
)

cli.CONFIG_PATH = Path(__file__).parent.parent.parent / "configs"


# ============================================================================
# Health Check Function Tests (unit tests for the functions themselves)
# ============================================================================


class TestHealthCheckEmbedding:
    """Tests for health_check_embedding function."""

    def test_returns_dimension_on_success(self) -> None:
        """Returns embedding dimension on success."""
        mock_embedding = MockEmbedding(384)

        result = health_check_embedding(mock_embedding)

        assert result == 384

    def test_raises_on_embedding_failure(self) -> None:
        """Raises EmbeddingNotSetError when embedding fails."""
        from llama_index.embeddings.openai import OpenAIEmbedding

        from autorag_research.exceptions import EmbeddingNotSetError

        original_api_key = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "havertz"

        embedding_model = OpenAIEmbedding()
        with pytest.raises(EmbeddingNotSetError):
            health_check_embedding(embedding_model)

        if original_api_key:
            os.environ["OPENAI_API_KEY"] = original_api_key


class TestHealthCheckLlm:
    """Tests for health_check_llm function."""

    def test_returns_true_on_success(self) -> None:
        """Returns True when LLM responds successfully."""
        mock_llm = MockLLM()

        result = health_check_llm(mock_llm)

        assert result is True

    def test_raises_on_llm_failure(self) -> None:
        """Raises LLMNotSetError when LLM fails."""
        from llama_index.llms.openai import OpenAI

        from autorag_research.exceptions import LLMNotSetError

        original_api_key = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "havertz"
        model = OpenAI()

        with pytest.raises(LLMNotSetError):
            health_check_llm(model)

        if original_api_key:
            os.environ["OPENAI_API_KEY"] = original_api_key


# ============================================================================
# Embedding Tests
# ============================================================================


class TestLoadEmbeddingModel:
    """Tests for load_embedding_model function."""

    def test_raises_file_not_found_for_missing_config(self) -> None:
        """Raises FileNotFoundError when config doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_embedding_model("nonexistent")

    def test_returns_embedding_instance(self) -> None:
        """Returns BaseEmbedding instance when config is valid."""
        result = load_embedding_model("mock")

        assert isinstance(result, BaseEmbedding)


class TestWithEmbeddingDecorator:
    """Tests for with_embedding decorator."""

    def test_string_to_instance_conversion(self) -> None:
        """Decorator converts string config name to BaseEmbedding instance."""

        @with_embedding()
        def my_func(embedding_model: BaseEmbedding | str) -> BaseEmbedding:
            return embedding_model  # type: ignore[return-value]

        result = my_func(embedding_model="mock")
        assert isinstance(result, BaseEmbedding)

    def test_instance_passthrough(self) -> None:
        """Decorator passes through BaseEmbedding instances unchanged."""

        @with_embedding()
        def my_func(embedding_model) -> object:
            return embedding_model

        mock_model = MockEmbedding(embed_dim=384)
        result = my_func(embedding_model=mock_model)
        assert result is mock_model

    def test_caching_same_config(self) -> None:
        """Decorator uses cached model for same config name."""
        results = []

        @with_embedding()
        def my_func(embedding_model) -> object:
            return embedding_model

        results.append(my_func(embedding_model="mock"))
        results.append(my_func(embedding_model="mock"))

        assert results[0] is results[1]

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
        def my_func(model, model2, other) -> object:
            return model, model2, other

        mock_model = MockEmbedding(embed_dim=384)
        mock_model2 = MockEmbedding(embed_dim=768)
        result1, result2, result3 = my_func(model=mock_model, model2=mock_model2, other=3)
        assert result1 is mock_model
        assert result2 is mock_model2
        assert result3 == 3


# ============================================================================
# LLM Tests
# ============================================================================


class TestLoadLlm:
    """Tests for load_llm function."""

    def test_raises_file_not_found_for_missing_config(self) -> None:
        """Raises FileNotFoundError when config doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_llm("nonexistent")

    def test_returns_llm_instance(self) -> None:
        """Returns BaseLLM instance when config is valid."""
        result = load_llm("mock")

        assert isinstance(result, BaseLLM)

    @patch("autorag_research.injection.instantiate")
    def test_calls_health_check_on_load(self, mock_instantiate: MagicMock) -> None:
        """Health check is called during load."""
        mock_model = MockLLM()
        mock_instantiate.return_value = mock_model
        mock_health_check = MagicMock()

        original_func = _llm_manager._health_check_func
        _llm_manager._health_check_func = mock_health_check
        try:
            load_llm("mock")
            mock_health_check.assert_called_once_with(mock_model)
        finally:
            _llm_manager._health_check_func = original_func

    @patch("autorag_research.injection.instantiate")
    def test_health_check_failure_propagates(self, mock_instantiate: MagicMock) -> None:
        """Health check failure raises LLMNotSetError."""
        from autorag_research.exceptions import LLMNotSetError

        mock_instantiate.return_value = MockLLM()

        original_func = _llm_manager._health_check_func
        _llm_manager._health_check_func = MagicMock(side_effect=LLMNotSetError())
        try:
            with pytest.raises(LLMNotSetError):
                load_llm("mock")
        finally:
            _llm_manager._health_check_func = original_func


class TestWithLlmDecorator:
    """Tests for with_llm decorator."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        _llm_manager.clear_cache()

    def teardown_method(self) -> None:
        """Clear cache after each test."""
        _llm_manager.clear_cache()

    def test_string_to_instance_conversion(self) -> None:
        """Decorator converts string config name to BaseLLM instance."""

        @with_llm()
        def my_func(llm: BaseLLM | str) -> BaseLLM:
            return llm  # type: ignore[return-value]

        result = my_func(llm="mock")
        assert isinstance(result, BaseLLM)

    def test_instance_passthrough(self) -> None:
        """Decorator passes through BaseLLM instances unchanged."""

        @with_llm()
        def my_func(llm) -> object:
            return llm

        mock_model = MockLLM()
        result = my_func(llm=mock_model)
        assert result is mock_model

    def test_caching_same_config(self) -> None:
        """Decorator uses cached model for same config name."""
        results = []

        @with_llm()
        def my_func(llm) -> object:
            return llm

        results.append(my_func(llm="mock"))
        results.append(my_func(llm="mock"))

        assert results[0] is results[1]

    def test_invalid_type_raises_error(self) -> None:
        """Decorator raises TypeError for invalid llm type."""

        @with_llm()
        def my_func(llm) -> None:
            pass

        with pytest.raises(TypeError, match="must be string, BaseLLM"):
            my_func(llm=123)  # type: ignore[arg-type]

    def test_invalid_param_name_raises_error(self) -> None:
        """Decorator raises ValueError when param_name doesn't exist."""
        with pytest.raises(ValueError, match="Parameter 'nonexistent' not found"):

            @with_llm(param_name="nonexistent")
            def my_func(llm) -> None:
                pass

    def test_custom_param_name(self) -> None:
        """Decorator works with custom param_name."""

        @with_llm(param_name="model")
        def my_func(model, other_arg) -> object:
            return model, other_arg

        mock_model = MockLLM()
        result1, result2 = my_func(model=mock_model, other_arg=42)
        assert result1 is mock_model
        assert result2 == 42
