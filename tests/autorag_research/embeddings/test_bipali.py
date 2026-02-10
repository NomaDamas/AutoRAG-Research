import io
from pathlib import Path

import pytest
import torch
from PIL import Image

from autorag_research.embeddings.bipali import BiPaliEmbeddings
from autorag_research.util import load_image

MODEL_NAME = "ModernVBERT/bimodernvbert"
MODEL_TYPE = "modernvbert"


class TestLoadImage:
    def test_load_image_from_file_path(self, tmp_path: Path):
        img = Image.new("RGB", (100, 100), color="red")
        img_path = tmp_path / "test_image.png"
        img.save(img_path)

        result = load_image(str(img_path))

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (100, 100)

    def test_load_image_from_bytes(self):
        img = Image.new("RGB", (50, 50), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        result = load_image(img_bytes)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (50, 50)

    def test_load_image_converts_to_rgb(self, tmp_path: Path):
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        img_path = tmp_path / "rgba_image.png"
        img.save(img_path)

        result = load_image(str(img_path))

        assert result.mode == "RGB"

    def test_load_image_unsupported_type_raises_error(self):
        with pytest.raises(TypeError):
            load_image(12345)


@pytest.fixture(scope="module")
def bipali_embeddings() -> BiPaliEmbeddings:
    return BiPaliEmbeddings(
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
class TestEmbedQuery:
    def test_embed_query_returns_list(self, bipali_embeddings: BiPaliEmbeddings):
        embedding = bipali_embeddings.embed_query("Hello world")

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(v, float) for v in embedding)


@pytest.mark.gpu
class TestEmbedImage:
    def test_embed_image_from_path(self, bipali_embeddings: BiPaliEmbeddings, sample_image: str):
        embedding = bipali_embeddings.embed_image(sample_image)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(v, float) for v in embedding)

    def test_embed_image_from_bytes(self, bipali_embeddings: BiPaliEmbeddings, sample_image_bytes: bytes):
        embedding = bipali_embeddings.embed_image(sample_image_bytes)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(v, float) for v in embedding)


@pytest.mark.gpu
class TestEmbedDocuments:
    def test_embed_documents(self, bipali_embeddings: BiPaliEmbeddings):
        texts = ["First text", "Second text", "Third text"]

        embeddings = bipali_embeddings.embed_documents(texts)

        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)

    def test_embed_documents_respects_batch_size(self, bipali_embeddings: BiPaliEmbeddings):
        bipali_embeddings.embed_batch_size = 2
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]

        embeddings = bipali_embeddings.embed_documents(texts)

        assert len(embeddings) == 5

    def test_embed_documents_empty_list(self, bipali_embeddings: BiPaliEmbeddings):
        embeddings = bipali_embeddings.embed_documents([])

        assert embeddings == []


@pytest.mark.gpu
class TestEmbedImages:
    def test_embed_images(self, bipali_embeddings: BiPaliEmbeddings, tmp_path: Path):
        img_paths = []
        for i in range(3):
            img = Image.new("RGB", (224, 224), color=(i * 50, i * 50, i * 50))
            img_path = tmp_path / f"img_{i}.png"
            img.save(img_path)
            img_paths.append(str(img_path))

        embeddings = bipali_embeddings.embed_images(img_paths)

        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)

    def test_embed_images_with_bytes(self, bipali_embeddings: BiPaliEmbeddings):
        img_bytes_list = []
        for i in range(2):
            img = Image.new("RGB", (224, 224), color=(i * 100, 0, 0))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes_list.append(buffer.getvalue())

        embeddings = bipali_embeddings.embed_images(img_bytes_list)

        assert len(embeddings) == 2

    def test_embed_images_empty_list(self, bipali_embeddings: BiPaliEmbeddings):
        embeddings = bipali_embeddings.embed_images([])

        assert embeddings == []


@pytest.mark.gpu
@pytest.mark.asyncio
class TestAsyncEmbeddings:
    async def test_aembed_query(self, bipali_embeddings: BiPaliEmbeddings):
        embedding = await bipali_embeddings.aembed_query("Async text")

        assert isinstance(embedding, list)
        assert len(embedding) > 0

    async def test_aembed_documents(self, bipali_embeddings: BiPaliEmbeddings):
        embeddings = await bipali_embeddings.aembed_documents(["Async query"])

        assert isinstance(embeddings, list)
        assert len(embeddings) == 1

    async def test_aembed_image(self, bipali_embeddings: BiPaliEmbeddings, sample_image: str):
        embedding = await bipali_embeddings.aembed_image(sample_image)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
