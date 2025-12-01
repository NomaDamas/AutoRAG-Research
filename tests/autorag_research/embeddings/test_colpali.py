import io
from pathlib import Path

import pytest
import torch
from PIL import Image

from autorag_research.embeddings.colpali import (
    ColPaliEmbeddings,
)

MODEL_NAME = "ModernVBERT/colmodernvbert"
MODEL_TYPE = "modernvbert"


@pytest.fixture(scope="module")
def colpali_embeddings() -> ColPaliEmbeddings:
    return ColPaliEmbeddings(
        model_name=MODEL_NAME,
        model_type=MODEL_TYPE,
        device="mps" if not torch.cuda.is_available() else "cuda:0",
        torch_dtype=torch.bfloat16,
    )


@pytest.fixture
def sample_image(tmp_path: Path) -> str:
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


@pytest.mark.gpu
class TestGetTextEmbedding:
    def test_get_text_embedding_returns_multi_vector(self, colpali_embeddings: ColPaliEmbeddings):
        embedding = colpali_embeddings.get_text_embedding("Hello world")

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(vec, list) for vec in embedding)
        assert all(isinstance(v, float) for vec in embedding for v in vec)

    def test_get_query_embedding_same_as_text(self, colpali_embeddings: ColPaliEmbeddings):
        text = "Test query"
        text_emb = colpali_embeddings.get_text_embedding(text)
        query_emb = colpali_embeddings.get_query_embedding(text)

        assert text_emb == query_emb


@pytest.mark.gpu
class TestGetImageEmbedding:
    def test_get_image_embedding_from_path(self, colpali_embeddings: ColPaliEmbeddings, sample_image: str):
        embedding = colpali_embeddings.get_image_embedding(sample_image)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(vec, list) for vec in embedding)
        assert all(isinstance(v, float) for vec in embedding for v in vec)

    def test_get_image_embedding_from_bytes(self, colpali_embeddings: ColPaliEmbeddings, sample_image_bytes: bytes):
        embedding = colpali_embeddings.get_image_embedding(sample_image_bytes)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(vec, list) for vec in embedding)


@pytest.mark.gpu
class TestGetTextEmbeddings:
    def test_get_text_embeddings(self, colpali_embeddings: ColPaliEmbeddings):
        texts = ["First text", "Second text", "Third text"]

        embeddings = colpali_embeddings.get_text_embeddings(texts)

        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
        assert all(isinstance(vec, list) for emb in embeddings for vec in emb)

    def test_get_text_embeddings_empty_list(self, colpali_embeddings: ColPaliEmbeddings):
        embeddings = colpali_embeddings.get_text_embeddings([])

        assert embeddings == []


@pytest.mark.gpu
class TestGetTextEmbeddingBatch:
    def test_get_text_embedding_batch(self, colpali_embeddings: ColPaliEmbeddings):
        texts = ["First text", "Second text", "Third text"]

        embeddings = colpali_embeddings.get_text_embedding_batch(texts)

        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)

    def test_get_text_embedding_batch_respects_batch_size(self, colpali_embeddings: ColPaliEmbeddings):
        colpali_embeddings.embed_batch_size = 2
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]

        embeddings = colpali_embeddings.get_text_embedding_batch(texts)

        assert len(embeddings) == 5


@pytest.mark.gpu
class TestGetImageEmbeddings:
    def test_get_image_embeddings(self, colpali_embeddings: ColPaliEmbeddings, tmp_path: Path):
        img_paths = []
        for i in range(3):
            img = Image.new("RGB", (224, 224), color=(i * 50, i * 50, i * 50))
            img_path = tmp_path / f"img_{i}.png"
            img.save(img_path)
            img_paths.append(str(img_path))

        embeddings = colpali_embeddings.get_image_embeddings(img_paths)

        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
        assert all(isinstance(vec, list) for emb in embeddings for vec in emb)

    def test_get_image_embeddings_with_bytes(self, colpali_embeddings: ColPaliEmbeddings):
        img_bytes_list = []
        for i in range(2):
            img = Image.new("RGB", (224, 224), color=(i * 100, 0, 0))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes_list.append(buffer.getvalue())

        embeddings = colpali_embeddings.get_image_embeddings(img_bytes_list)

        assert len(embeddings) == 2

    def test_get_image_embeddings_empty_list(self, colpali_embeddings: ColPaliEmbeddings):
        embeddings = colpali_embeddings.get_image_embeddings([])

        assert embeddings == []


@pytest.mark.gpu
class TestGetImageEmbeddingBatch:
    def test_get_image_embedding_batch(self, colpali_embeddings: ColPaliEmbeddings, tmp_path: Path):
        img_paths = []
        for i in range(3):
            img = Image.new("RGB", (224, 224), color=(i * 50, i * 50, i * 50))
            img_path = tmp_path / f"img_{i}.png"
            img.save(img_path)
            img_paths.append(str(img_path))

        embeddings = colpali_embeddings.get_image_embedding_batch(img_paths)

        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)

    def test_get_image_embedding_batch_respects_batch_size(self, colpali_embeddings: ColPaliEmbeddings, tmp_path: Path):
        colpali_embeddings.embed_batch_size = 2
        img_paths = []
        for i in range(5):
            img = Image.new("RGB", (224, 224), color=(i * 40, 0, 0))
            img_path = tmp_path / f"batch_img_{i}.png"
            img.save(img_path)
            img_paths.append(str(img_path))

        embeddings = colpali_embeddings.get_image_embedding_batch(img_paths)

        assert len(embeddings) == 5


@pytest.mark.gpu
@pytest.mark.asyncio
class TestAsyncEmbeddings:
    async def test_aget_text_embedding(self, colpali_embeddings: ColPaliEmbeddings):
        embedding = await colpali_embeddings.aget_text_embedding("Async text")

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(vec, list) for vec in embedding)

    async def test_aget_query_embedding(self, colpali_embeddings: ColPaliEmbeddings):
        embedding = await colpali_embeddings.aget_query_embedding("Async query")

        assert isinstance(embedding, list)
        assert len(embedding) > 0

    async def test_aget_image_embedding(self, colpali_embeddings: ColPaliEmbeddings, sample_image: str):
        embedding = await colpali_embeddings.aget_image_embedding(sample_image)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(vec, list) for vec in embedding)
