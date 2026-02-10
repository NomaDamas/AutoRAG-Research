"""Tests for InfinityEmbeddings client."""

import base64
import io
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import numpy as np
import pytest
from PIL import Image

from autorag_research.embeddings.infinity import InfinityEmbeddings

# --- Fixtures ---

HIDDEN_DIM = 128
NUM_TOKENS = 4  # Number of token vectors per embedding


def _make_base64_embedding(num_tokens: int = NUM_TOKENS, hidden_dim: int = HIDDEN_DIM) -> str:
    """Create a base64-encoded flat float32 array simulating a multi-vector embedding."""
    flat = np.random.default_rng(42).standard_normal(num_tokens * hidden_dim).astype(np.float32)
    return base64.b64encode(flat.tobytes()).decode("utf-8")


def _make_float_embedding(num_tokens: int = NUM_TOKENS, hidden_dim: int = HIDDEN_DIM) -> list[float]:
    """Create a flat list of floats simulating a multi-vector embedding."""
    return np.random.default_rng(42).standard_normal(num_tokens * hidden_dim).astype(np.float32).tolist()


def _make_api_response(embeddings_data: list, encoding: str = "base64") -> dict:
    """Build a mock API response dict."""
    return {
        "data": [{"embedding": emb, "index": i} for i, emb in enumerate(embeddings_data)],
        "model": "michaelfeil/colqwen2-v0.1",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }


@pytest.fixture
def mock_response_single_base64() -> dict:
    return _make_api_response([_make_base64_embedding()])


@pytest.fixture
def mock_response_batch_base64() -> dict:
    return _make_api_response([_make_base64_embedding(), _make_base64_embedding()])


@pytest.fixture
def mock_response_single_float() -> dict:
    return _make_api_response([_make_float_embedding()], encoding="float")


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
    """Create an InfinityEmbeddings instance with real httpx clients."""
    return InfinityEmbeddings()


# --- Test Classes ---


@pytest.mark.api
class TestInfinityEmbeddingsInit:
    def test_default_values(self, embeddings: InfinityEmbeddings):
        assert embeddings.url == "http://localhost:7997"
        assert embeddings.model_name == "michaelfeil/colqwen2-v0.1"
        assert embeddings.encoding == "base64"
        assert embeddings.hidden_dim == 128
        assert embeddings.timeout == 60.0
        assert embeddings.max_retries == 3

    def test_custom_values(self):
        emb = InfinityEmbeddings(
            url="http://custom:8080",
            model_name="custom/model",
            encoding="float",
            hidden_dim=256,
            timeout=30.0,
            max_retries=5,
        )
        assert emb.url == "http://custom:8080"
        assert emb.model_name == "custom/model"
        assert emb.encoding == "float"
        assert emb.hidden_dim == 256
        assert emb.timeout == 30.0
        assert emb.max_retries == 5

    def test_httpx_clients_initialized(self, embeddings: InfinityEmbeddings):
        assert embeddings._client is not None
        assert embeddings._async_client is not None
        assert isinstance(embeddings._client, httpx.Client)
        assert isinstance(embeddings._async_client, httpx.AsyncClient)


@pytest.mark.api
class TestImageToBase64Jpeg:
    def test_from_str_path(self, sample_image_path: str):
        result = InfinityEmbeddings._image_to_base64_jpeg(sample_image_path)
        assert isinstance(result, str)
        # Verify it's valid base64 that decodes to a JPEG
        decoded = base64.b64decode(result)
        assert decoded[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_from_path_object(self, sample_image_path: str):
        result = InfinityEmbeddings._image_to_base64_jpeg(Path(sample_image_path))
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded[:2] == b"\xff\xd8"

    def test_from_bytes(self, sample_image_bytes: bytes):
        result = InfinityEmbeddings._image_to_base64_jpeg(sample_image_bytes)
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded[:2] == b"\xff\xd8"

    def test_from_bytesio(self, sample_image_bytesio: BytesIO):
        result = InfinityEmbeddings._image_to_base64_jpeg(sample_image_bytesio)
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded[:2] == b"\xff\xd8"

    def test_unsupported_type_raises_error(self):
        with pytest.raises(TypeError, match="Unsupported image type"):
            InfinityEmbeddings._image_to_base64_jpeg(12345)  # type: ignore[arg-type]


@pytest.mark.api
class TestParseEmbeddings:
    def test_base64_single(self, embeddings: InfinityEmbeddings, mock_response_single_base64: dict):
        results = embeddings._parse_embeddings(mock_response_single_base64)
        assert len(results) == 1
        assert len(results[0]) == NUM_TOKENS
        assert len(results[0][0]) == HIDDEN_DIM

    def test_base64_multiple(self, embeddings: InfinityEmbeddings, mock_response_batch_base64: dict):
        results = embeddings._parse_embeddings(mock_response_batch_base64)
        assert len(results) == 2
        for result in results:
            assert len(result) == NUM_TOKENS
            assert len(result[0]) == HIDDEN_DIM

    def test_float_encoding(self, mock_response_single_float: dict):
        emb = InfinityEmbeddings(encoding="float")
        results = emb._parse_embeddings(mock_response_single_float)
        assert len(results) == 1
        assert len(results[0]) == NUM_TOKENS
        assert len(results[0][0]) == HIDDEN_DIM

    def test_correct_reshape(self, embeddings: InfinityEmbeddings):
        """Verify that the flat array is correctly reshaped to (num_tokens, hidden_dim)."""
        num_tokens = 6
        flat = np.arange(num_tokens * HIDDEN_DIM, dtype=np.float32)
        b64 = base64.b64encode(flat.tobytes()).decode("utf-8")
        response = _make_api_response([b64])
        results = embeddings._parse_embeddings(response)
        assert len(results[0]) == num_tokens
        # First vector should start at 0.0
        assert results[0][0][0] == pytest.approx(0.0)
        # Second vector should start at HIDDEN_DIM
        assert results[0][1][0] == pytest.approx(float(HIDDEN_DIM))


@pytest.mark.api
class TestEmbedText:
    def test_embed_text_returns_multi_vector(self, embeddings: InfinityEmbeddings, mock_response_single_base64: dict):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_single_base64
        mock_resp.raise_for_status = MagicMock()
        embeddings._client.post = MagicMock(return_value=mock_resp)

        result = embeddings.embed_text("test query")
        assert len(result) == NUM_TOKENS
        assert len(result[0]) == HIDDEN_DIM

        # Verify API was called with correct payload
        call_args = embeddings._client.post.call_args
        assert call_args[0][0] == "http://localhost:7997/embeddings"
        payload = call_args[1]["json"]
        assert payload["input"] == ["test query"]
        assert payload["modality"] == "text"

    def test_embed_query_delegates_to_text(self, embeddings: InfinityEmbeddings, mock_response_single_base64: dict):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_single_base64
        mock_resp.raise_for_status = MagicMock()
        embeddings._client.post = MagicMock(return_value=mock_resp)

        result = embeddings.embed_query("test query")
        assert len(result) == NUM_TOKENS
        assert len(result[0]) == HIDDEN_DIM

    def test_embed_documents_single_api_call(self, embeddings: InfinityEmbeddings, mock_response_batch_base64: dict):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_batch_base64
        mock_resp.raise_for_status = MagicMock()
        embeddings._client.post = MagicMock(return_value=mock_resp)

        results = embeddings.embed_documents(["text1", "text2"])
        assert len(results) == 2
        # Verify single API call (batch)
        assert embeddings._client.post.call_count == 1
        payload = embeddings._client.post.call_args[1]["json"]
        assert payload["input"] == ["text1", "text2"]

    def test_embed_documents_empty_list(self, embeddings: InfinityEmbeddings):
        result = embeddings.embed_documents([])
        assert result == []


@pytest.mark.api
class TestEmbedImage:
    def test_embed_image_from_path(
        self, embeddings: InfinityEmbeddings, mock_response_single_base64: dict, sample_image_path: str
    ):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_single_base64
        mock_resp.raise_for_status = MagicMock()
        embeddings._client.post = MagicMock(return_value=mock_resp)

        result = embeddings.embed_image(sample_image_path)
        assert len(result) == NUM_TOKENS
        assert len(result[0]) == HIDDEN_DIM

        # Verify modality is "image"
        payload = embeddings._client.post.call_args[1]["json"]
        assert payload["modality"] == "image"

    def test_embed_image_from_bytes(
        self, embeddings: InfinityEmbeddings, mock_response_single_base64: dict, sample_image_bytes: bytes
    ):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_single_base64
        mock_resp.raise_for_status = MagicMock()
        embeddings._client.post = MagicMock(return_value=mock_resp)

        result = embeddings.embed_image(sample_image_bytes)
        assert len(result) == NUM_TOKENS
        assert len(result[0]) == HIDDEN_DIM

    def test_embed_images_single_api_call(
        self, embeddings: InfinityEmbeddings, mock_response_batch_base64: dict, sample_image_path: str
    ):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_batch_base64
        mock_resp.raise_for_status = MagicMock()
        embeddings._client.post = MagicMock(return_value=mock_resp)

        results = embeddings.embed_images([sample_image_path, sample_image_path])
        assert len(results) == 2
        assert embeddings._client.post.call_count == 1

    def test_embed_images_empty_list(self, embeddings: InfinityEmbeddings):
        result = embeddings.embed_images([])
        assert result == []


@pytest.mark.api
@pytest.mark.asyncio
class TestAsyncMethods:
    async def test_aembed_text(self, embeddings: InfinityEmbeddings, mock_response_single_base64: dict):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_single_base64
        mock_resp.raise_for_status = MagicMock()
        embeddings._async_client.post = AsyncMock(return_value=mock_resp)

        result = await embeddings.aembed_text("async text")
        assert len(result) == NUM_TOKENS
        assert len(result[0]) == HIDDEN_DIM

    async def test_aembed_query(self, embeddings: InfinityEmbeddings, mock_response_single_base64: dict):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_single_base64
        mock_resp.raise_for_status = MagicMock()
        embeddings._async_client.post = AsyncMock(return_value=mock_resp)

        result = await embeddings.aembed_query("async query")
        assert len(result) == NUM_TOKENS
        assert len(result[0]) == HIDDEN_DIM

    async def test_aembed_image(
        self, embeddings: InfinityEmbeddings, mock_response_single_base64: dict, sample_image_path: str
    ):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_single_base64
        mock_resp.raise_for_status = MagicMock()
        embeddings._async_client.post = AsyncMock(return_value=mock_resp)

        result = await embeddings.aembed_image(sample_image_path)
        assert len(result) == NUM_TOKENS
        assert len(result[0]) == HIDDEN_DIM

    async def test_aembed_documents(self, embeddings: InfinityEmbeddings, mock_response_batch_base64: dict):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_batch_base64
        mock_resp.raise_for_status = MagicMock()
        embeddings._async_client.post = AsyncMock(return_value=mock_resp)

        results = await embeddings.aembed_documents(["text1", "text2"])
        assert len(results) == 2
        assert embeddings._async_client.post.call_count == 1

    async def test_aembed_documents_empty(self, embeddings: InfinityEmbeddings):
        result = await embeddings.aembed_documents([])
        assert result == []

    async def test_aembed_images(
        self, embeddings: InfinityEmbeddings, mock_response_batch_base64: dict, sample_image_path: str
    ):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_batch_base64
        mock_resp.raise_for_status = MagicMock()
        embeddings._async_client.post = AsyncMock(return_value=mock_resp)

        results = await embeddings.aembed_images([sample_image_path, sample_image_path])
        assert len(results) == 2
        assert embeddings._async_client.post.call_count == 1

    async def test_aembed_images_empty(self, embeddings: InfinityEmbeddings):
        result = await embeddings.aembed_images([])
        assert result == []


@pytest.mark.api
class TestRetryBehavior:
    def test_retries_on_server_error(self, embeddings: InfinityEmbeddings, mock_response_single_base64: dict):
        """Verify that the client retries on HTTP 500 errors."""
        error_resp = MagicMock()
        error_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=MagicMock(status_code=500)
        )

        success_resp = MagicMock()
        success_resp.json.return_value = mock_response_single_base64
        success_resp.raise_for_status = MagicMock()

        embeddings._client.post = MagicMock(side_effect=[error_resp, success_resp])

        result = embeddings.embed_text("retry test")
        assert len(result) == NUM_TOKENS
        assert embeddings._client.post.call_count == 2

    def test_raises_after_max_retries(self, embeddings: InfinityEmbeddings):
        """Verify that the client raises after exhausting retries."""
        error_resp = MagicMock()
        error_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=MagicMock(status_code=500)
        )

        # Set max_retries to 2 for faster test
        embeddings.max_retries = 2
        embeddings._client.post = MagicMock(return_value=error_resp)

        with pytest.raises(httpx.HTTPStatusError):
            embeddings.embed_text("fail test")

        assert embeddings._client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_async_retries_on_server_error(
        self, embeddings: InfinityEmbeddings, mock_response_single_base64: dict
    ):
        """Verify that the async client retries on HTTP 500 errors."""
        error_resp = MagicMock()
        error_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=MagicMock(status_code=500)
        )

        success_resp = MagicMock()
        success_resp.json.return_value = mock_response_single_base64
        success_resp.raise_for_status = MagicMock()

        embeddings._async_client.post = AsyncMock(side_effect=[error_resp, success_resp])

        result = await embeddings.aembed_text("async retry test")
        assert len(result) == NUM_TOKENS
        assert embeddings._async_client.post.call_count == 2
