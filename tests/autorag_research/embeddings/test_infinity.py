"""Tests for InfinityEmbeddings client."""

import io
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from autorag_research.embeddings.infinity import InfinityEmbeddings
from autorag_research.util import load_image

# --- Constants ---

HIDDEN_DIM = 128
NUM_TOKENS = 4  # Number of token vectors per embedding


def _make_ndarray(num_tokens: int = NUM_TOKENS, hidden_dim: int = HIDDEN_DIM) -> np.ndarray:
    """Create a numpy array simulating a multi-vector embedding."""
    return np.random.default_rng(42).standard_normal((num_tokens, hidden_dim)).astype(np.float32)


def _make_mock_future(embeddings: list[np.ndarray], total_tokens: int = 10) -> MagicMock:
    """Create a mock Future whose .result() returns (embeddings, total_tokens)."""
    future = MagicMock()
    future.result.return_value = (embeddings, total_tokens)
    return future


VISION_CLIENT_PATH = "infinity_client.vision_client.InfinityVisionAPI"


def _create_embeddings_with_mock_client() -> InfinityEmbeddings:
    """Create an InfinityEmbeddings instance with a mocked vision client (skip network call)."""
    with patch(VISION_CLIENT_PATH):
        emb = InfinityEmbeddings()
    emb._vision_client = MagicMock()
    return emb


# --- Fixtures ---


@pytest.fixture
def sample_image_path(tmp_path: Path) -> str:
    img = Image.new("RGB", (224, 224), color="green")
    img_path = tmp_path / "sample.png"
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def sample_image_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color="purple")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_image_bytesio() -> BytesIO:
    img = Image.new("RGB", (224, 224), color="blue")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


@pytest.fixture
def embeddings() -> InfinityEmbeddings:
    """Create an InfinityEmbeddings instance with a mocked vision client."""
    return _create_embeddings_with_mock_client()


# --- Test Classes ---


@pytest.mark.api
class TestInfinityEmbeddingsInit:
    def test_default_values(self, embeddings: InfinityEmbeddings):
        assert embeddings.url == "http://localhost:7997"
        assert embeddings.model_name == "michaelfeil/colqwen2-v0.1"
        assert embeddings.encoding == "base64"

    def test_custom_values(self):
        with patch(VISION_CLIENT_PATH):
            emb = InfinityEmbeddings(
                url="http://custom:8080",
                model_name="custom/model",
                encoding="float",
            )
        assert emb.url == "http://custom:8080"
        assert emb.model_name == "custom/model"
        assert emb.encoding == "float"

    def test_vision_client_initialized(self):
        with patch(VISION_CLIENT_PATH) as mock_cls:
            emb = InfinityEmbeddings(url="http://test:9999", encoding="float", model_name="test/model")
            mock_cls.assert_called_once_with(url="http://test:9999", format="float", model="test/model")
            assert emb._vision_client is mock_cls.return_value


@pytest.mark.api
class TestLoadImage:
    def test_from_str_path(self, sample_image_path: str):
        result = load_image(sample_image_path)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_from_path_object(self, sample_image_path: str):
        result = load_image(Path(sample_image_path))
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_from_bytes(self, sample_image_bytes: bytes):
        result = load_image(sample_image_bytes)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_from_bytesio(self, sample_image_bytesio: BytesIO):
        result = load_image(sample_image_bytesio)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_unsupported_type_raises_error(self):
        with pytest.raises(TypeError, match="Unsupported image type"):
            load_image(12345)  # type: ignore[arg-type]


@pytest.mark.api
class TestEmbedText:
    def test_embed_text_returns_multi_vector(self, embeddings: InfinityEmbeddings):
        mock_emb = _make_ndarray()
        embeddings._vision_client.embed.return_value = _make_mock_future([mock_emb])

        result = embeddings.embed_text("test query")
        assert len(result) == NUM_TOKENS
        assert len(result[0]) == HIDDEN_DIM

        embeddings._vision_client.embed.assert_called_once_with("michaelfeil/colqwen2-v0.1", ["test query"])

    def test_embed_query_delegates_to_text(self, embeddings: InfinityEmbeddings):
        mock_emb = _make_ndarray()
        embeddings._vision_client.embed.return_value = _make_mock_future([mock_emb])

        result = embeddings.embed_query("test query")
        assert len(result) == NUM_TOKENS
        assert len(result[0]) == HIDDEN_DIM

    def test_embed_documents_single_api_call(self, embeddings: InfinityEmbeddings):
        mock_embs = [_make_ndarray(), _make_ndarray()]
        embeddings._vision_client.embed.return_value = _make_mock_future(mock_embs)

        results = embeddings.embed_documents(["text1", "text2"])
        assert len(results) == 2
        assert embeddings._vision_client.embed.call_count == 1
        embeddings._vision_client.embed.assert_called_once_with("michaelfeil/colqwen2-v0.1", ["text1", "text2"])

    def test_embed_documents_empty_list(self, embeddings: InfinityEmbeddings):
        result = embeddings.embed_documents([])
        assert result == []


@pytest.mark.api
class TestEmbedImage:
    def test_embed_image_from_path(self, embeddings: InfinityEmbeddings, sample_image_path: str):
        mock_emb = _make_ndarray()
        embeddings._vision_client.image_embed.return_value = _make_mock_future([mock_emb])

        result = embeddings.embed_image(sample_image_path)
        assert len(result) == NUM_TOKENS
        assert len(result[0]) == HIDDEN_DIM

        # Verify image_embed was called with PIL Image
        call_args = embeddings._vision_client.image_embed.call_args
        assert call_args[0][0] == "michaelfeil/colqwen2-v0.1"
        assert isinstance(call_args[0][1][0], Image.Image)

    def test_embed_image_from_bytes(self, embeddings: InfinityEmbeddings, sample_image_bytes: bytes):
        mock_emb = _make_ndarray()
        embeddings._vision_client.image_embed.return_value = _make_mock_future([mock_emb])

        result = embeddings.embed_image(sample_image_bytes)
        assert len(result) == NUM_TOKENS
        assert len(result[0]) == HIDDEN_DIM

    def test_embed_images_single_api_call(self, embeddings: InfinityEmbeddings, sample_image_path: str):
        mock_embs = [_make_ndarray(), _make_ndarray()]
        embeddings._vision_client.image_embed.return_value = _make_mock_future(mock_embs)

        results = embeddings.embed_images([sample_image_path, sample_image_path])
        assert len(results) == 2
        assert embeddings._vision_client.image_embed.call_count == 1

    def test_embed_images_empty_list(self, embeddings: InfinityEmbeddings):
        result = embeddings.embed_images([])
        assert result == []


@pytest.mark.api
@pytest.mark.asyncio
class TestAsyncMethods:
    async def test_aembed_text(self, embeddings: InfinityEmbeddings):
        mock_emb = _make_ndarray()
        embeddings._vision_client.embed.return_value = _make_mock_future([mock_emb])

        result = await embeddings.aembed_text("async text")
        assert len(result) == NUM_TOKENS
        assert len(result[0]) == HIDDEN_DIM

    async def test_aembed_query(self, embeddings: InfinityEmbeddings):
        mock_emb = _make_ndarray()
        embeddings._vision_client.embed.return_value = _make_mock_future([mock_emb])

        result = await embeddings.aembed_query("async query")
        assert len(result) == NUM_TOKENS
        assert len(result[0]) == HIDDEN_DIM

    async def test_aembed_image(self, embeddings: InfinityEmbeddings, sample_image_path: str):
        mock_emb = _make_ndarray()
        embeddings._vision_client.image_embed.return_value = _make_mock_future([mock_emb])

        result = await embeddings.aembed_image(sample_image_path)
        assert len(result) == NUM_TOKENS
        assert len(result[0]) == HIDDEN_DIM

    async def test_aembed_documents(self, embeddings: InfinityEmbeddings):
        mock_embs = [_make_ndarray(), _make_ndarray()]
        embeddings._vision_client.embed.return_value = _make_mock_future(mock_embs)

        results = await embeddings.aembed_documents(["text1", "text2"])
        assert len(results) == 2
        assert embeddings._vision_client.embed.call_count == 1

    async def test_aembed_documents_empty(self, embeddings: InfinityEmbeddings):
        result = await embeddings.aembed_documents([])
        assert result == []

    async def test_aembed_images(self, embeddings: InfinityEmbeddings, sample_image_path: str):
        mock_embs = [_make_ndarray(), _make_ndarray()]
        embeddings._vision_client.image_embed.return_value = _make_mock_future(mock_embs)

        results = await embeddings.aembed_images([sample_image_path, sample_image_path])
        assert len(results) == 2
        assert embeddings._vision_client.image_embed.call_count == 1

    async def test_aembed_images_empty(self, embeddings: InfinityEmbeddings):
        result = await embeddings.aembed_images([])
        assert result == []
