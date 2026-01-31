import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.fake import FakeListLLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
        mock_embedding = FakeEmbeddings(size=384)

        result = health_check_embedding(mock_embedding)

        assert result == 384

    def test_raises_on_embedding_failure(self) -> None:
        """Raises EmbeddingNotSetError when embedding fails."""
        from autorag_research.exceptions import EmbeddingError

        original_api_key = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "havertz"

        embedding_model = OpenAIEmbeddings()
        with pytest.raises(EmbeddingError):
            health_check_embedding(embedding_model)

        if original_api_key:
            os.environ["OPENAI_API_KEY"] = original_api_key


class TestHealthCheckLlm:
    """Tests for health_check_llm function."""

    def test_returns_true_on_success(self) -> None:
        """Returns True when LLM responds successfully."""
        mock_llm = FakeListLLM(responses=["Mock response"])

        health_check_llm(mock_llm)

    def test_raises_on_llm_failure(self) -> None:
        """Raises LLMNotSetError when LLM fails."""
        from autorag_research.exceptions import LLMError

        original_api_key = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "havertz"
        model = ChatOpenAI()

        with pytest.raises(LLMError):
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
        """Returns Embeddings instance when config is valid."""
        result = load_embedding_model("mock")

        assert isinstance(result, Embeddings)


class TestWithEmbeddingDecorator:
    """Tests for with_embedding decorator."""

    def test_string_to_instance_conversion(self) -> None:
        """Decorator converts string config name to Embeddings instance."""

        @with_embedding()
        def my_func(embedding_model: Embeddings | str) -> Embeddings:
            return embedding_model  # type: ignore[return-value]

        result = my_func(embedding_model="mock")
        assert isinstance(result, Embeddings)

    def test_instance_passthrough(self) -> None:
        """Decorator passes through Embeddings instances unchanged."""

        @with_embedding()
        def my_func(embedding_model) -> object:
            return embedding_model

        mock_model = FakeEmbeddings(size=384)
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

        with pytest.raises(TypeError, match="must be string, Embeddings or MultiVectorBaseEmbedding"):
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

        mock_model = FakeEmbeddings(size=384)
        mock_model2 = FakeEmbeddings(size=768)
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
        """Returns BaseLanguageModel instance when config is valid."""
        result = load_llm("mock")

        assert isinstance(result, BaseLanguageModel)

    @patch("autorag_research.injection.instantiate")
    def test_calls_health_check_on_load(self, mock_instantiate: MagicMock) -> None:
        """Health check is called during load."""
        mock_model = FakeListLLM(responses=["Mock response"])
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
        from autorag_research.exceptions import LLMError

        mock_instantiate.return_value = FakeListLLM(responses=["Mock response"])

        original_func = _llm_manager._health_check_func
        _llm_manager._health_check_func = MagicMock(side_effect=LLMError())
        try:
            with pytest.raises(LLMError):
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
        """Decorator converts string config name to BaseLanguageModel instance."""

        @with_llm()
        def my_func(llm: BaseLanguageModel | str) -> BaseLanguageModel:
            return llm  # type: ignore[return-value]

        result = my_func(llm="mock")
        assert isinstance(result, BaseLanguageModel)

    def test_instance_passthrough(self) -> None:
        """Decorator passes through BaseLanguageModel instances unchanged."""

        @with_llm()
        def my_func(llm) -> object:
            return llm

        mock_model = FakeListLLM(responses=["Mock response"])
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

        with pytest.raises(TypeError, match="must be string, BaseLanguageModel"):
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

        mock_model = FakeListLLM(responses=["Mock response"])
        result1, result2 = my_func(model=mock_model, other_arg=42)
        assert result1 is mock_model
        assert result2 == 42
