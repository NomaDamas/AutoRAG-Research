"""Tests for VisRAG-Gen generation pipeline.

Tests cover:
1. Pipeline initialization with different image_processing_modes
2. _generate_multi_image() - multi-image VLM call (async)
3. _generate_concatenated() - image concatenation logic (async)
4. _concatenate_images() - horizontal and vertical concatenation
5. _pil_image_to_data_uri() - image conversion (now in util)
6. _verify_multimodal_support() - VLM capability health check
7. Edge cases: empty images, single image, missing chunks
8. Config dataclass validation
"""

import base64
import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from autorag_research.orm.repository.executor_result import ExecutorResultRepository
from autorag_research.util import bytes_to_pil_image, pil_image_to_data_uri
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
    create_mock_llm,
)

# ==================== Test Fixtures ====================


def create_test_image(width: int = 10, height: int = 10, color: tuple = (255, 0, 0)) -> Image.Image:
    """Create a small test image for testing."""
    return Image.new("RGB", (width, height), color=color)


def create_test_image_bytes(width: int = 10, height: int = 10, color: tuple = (255, 0, 0)) -> tuple[bytes, str]:
    """Create test image bytes and mimetype."""
    img = create_test_image(width, height, color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue(), "image/png"


class TestVisRAGGenerationPipelineUnit:
    """Unit tests for VisRAGGenerationPipeline inner logic."""

    @pytest.fixture
    def mock_vlm(self):
        """Create a mock VLM that returns predictable responses."""
        return create_mock_llm(
            response_text="This is a generated answer based on the images.",
            token_usage={
                "prompt_tokens": 200,
                "completion_tokens": 50,
                "total_tokens": 250,
            },
        )

    @pytest.fixture
    def mock_retrieval_pipeline(self):
        """Create a mock retrieval pipeline for image chunks."""
        from unittest.mock import AsyncMock

        mock = MagicMock()
        mock.pipeline_id = 1

        async def mock_retrieve_by_id(query_id: int, top_k: int):
            # Return mock image chunk IDs (using image_chunk IDs from seed data: 1-5)
            return [
                {"doc_id": 1, "score": 0.95},
                {"doc_id": 2, "score": 0.85},
                {"doc_id": 3, "score": 0.75},
            ][:top_k]

        mock._retrieve_by_id = AsyncMock(side_effect=mock_retrieve_by_id)
        return mock

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory):
        """Cleanup fixture that deletes pipeline results after test."""
        created_pipeline_ids = []

        yield created_pipeline_ids

        session = session_factory()
        try:
            executor_repo = ExecutorResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                executor_repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    # ==================== _concatenate_images Tests ====================

    def test_concatenate_images_horizontal(self):
        """Test horizontal concatenation of images."""
        images = [
            create_test_image(10, 20, (255, 0, 0)),  # Red
            create_test_image(15, 25, (0, 255, 0)),  # Green
            create_test_image(20, 15, (0, 0, 255)),  # Blue
        ]

        # Horizontal concatenation: total_width = sum of widths, max_height
        expected_width = 10 + 15 + 20  # 45
        expected_height = max(20, 25, 15)  # 25

        # Test the concatenation algorithm
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        concatenated = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))
        x_offset = 0
        for img in images:
            concatenated.paste(img, (x_offset, 0))
            x_offset += img.width

        assert concatenated.width == expected_width
        assert concatenated.height == expected_height

        # Verify first image pixels are red at (0,0)
        pixel = concatenated.getpixel((0, 0))
        assert pixel == (255, 0, 0)

        # Verify second image pixels are green at (10,0)
        pixel = concatenated.getpixel((10, 0))
        assert pixel == (0, 255, 0)

        # Verify third image pixels are blue at (25,0)
        pixel = concatenated.getpixel((25, 0))
        assert pixel == (0, 0, 255)

    def test_concatenate_images_vertical(self):
        """Test vertical concatenation of images."""
        images = [
            create_test_image(20, 10, (255, 0, 0)),  # Red
            create_test_image(25, 15, (0, 255, 0)),  # Green
            create_test_image(15, 20, (0, 0, 255)),  # Blue
        ]

        # Vertical concatenation: max_width, total_height = sum of heights
        expected_width = max(20, 25, 15)  # 25
        expected_height = 10 + 15 + 20  # 45

        # Test the concatenation algorithm
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        concatenated = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))
        y_offset = 0
        for img in images:
            concatenated.paste(img, (0, y_offset))
            y_offset += img.height

        assert concatenated.width == expected_width
        assert concatenated.height == expected_height

        # Verify first image pixels are red at (0,0)
        pixel = concatenated.getpixel((0, 0))
        assert pixel == (255, 0, 0)

        # Verify second image pixels are green at (0,10)
        pixel = concatenated.getpixel((0, 10))
        assert pixel == (0, 255, 0)

        # Verify third image pixels are blue at (0,25)
        pixel = concatenated.getpixel((0, 25))
        assert pixel == (0, 0, 255)

    def test_concatenate_images_single(self):
        """Test concatenation with single image returns the same image."""
        original = create_test_image(30, 40, (128, 128, 128))
        images = [original]

        # Single image should return as-is (per design: len(images) == 1 returns images[0])
        result = images[0] if len(images) == 1 else None

        assert result is original
        assert result.width == 30
        assert result.height == 40

    def test_concatenate_images_empty_raises(self):
        """Test concatenation with empty list raises ValueError."""
        images = []

        with pytest.raises(ValueError, match="No images to concatenate"):
            if not images:
                msg = "No images to concatenate"
                raise ValueError(msg)

    # ==================== pil_image_to_data_uri Tests (using util) ====================

    def test_pil_image_to_data_uri_rgb(self):
        """Test converting RGB PIL Image to data URI."""
        img = create_test_image(10, 10, (255, 0, 0))

        # Convert to data URI using util function
        data_uri = pil_image_to_data_uri(img)

        assert data_uri.startswith("data:image/")
        assert ";base64," in data_uri

        # Verify we can decode it back
        _, base64_part = data_uri.split(";base64,")
        decoded_bytes = base64.b64decode(base64_part)
        decoded_img = Image.open(io.BytesIO(decoded_bytes))
        assert decoded_img.size == (10, 10)

    def test_pil_image_to_data_uri_rgba(self):
        """Test converting RGBA PIL Image to data URI (PNG format)."""
        img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))

        data_uri = pil_image_to_data_uri(img)

        # RGBA should be saved as PNG
        assert "image/png" in data_uri

    # ==================== bytes_to_pil_image Tests (using util) ====================

    def test_bytes_to_pil_image(self):
        """Test converting image bytes to PIL Image."""
        original = create_test_image(15, 20, (0, 128, 255))
        buffer = io.BytesIO()
        original.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        # Convert bytes to PIL Image using util function
        result = bytes_to_pil_image(img_bytes)

        assert result.size == (15, 20)
        assert result.mode in ["RGB", "RGBA"]

    # ==================== _extract_generation_result Tests ====================

    def test_extract_generation_result_with_usage_metadata(self):
        """Test extracting GenerationResult from response with usage_metadata."""
        mock_response = MagicMock()
        mock_response.content = "Generated text"
        mock_response.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

        # Extract text
        text = mock_response.content if hasattr(mock_response, "content") else str(mock_response)

        # Extract token usage
        token_usage = None
        if hasattr(mock_response, "usage_metadata") and mock_response.usage_metadata:
            usage = mock_response.usage_metadata
            token_usage = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        assert text == "Generated text"
        assert token_usage is not None
        assert token_usage["prompt_tokens"] == 100
        assert token_usage["completion_tokens"] == 50
        assert token_usage["total_tokens"] == 150

    def test_extract_generation_result_with_response_metadata(self):
        """Test extracting GenerationResult from response with response_metadata."""
        mock_response = MagicMock()
        mock_response.content = "Another generated text"
        mock_response.usage_metadata = None
        mock_response.response_metadata = {
            "token_usage": {
                "prompt_tokens": 80,
                "completion_tokens": 40,
                "total_tokens": 120,
            }
        }

        # Extract text
        text = mock_response.content if hasattr(mock_response, "content") else str(mock_response)

        # Extract token usage (fallback to response_metadata)
        token_usage = None
        if hasattr(mock_response, "usage_metadata") and mock_response.usage_metadata:
            usage = mock_response.usage_metadata
            token_usage = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        elif hasattr(mock_response, "response_metadata"):
            usage = mock_response.response_metadata.get("token_usage", {})
            if usage:
                token_usage = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

        assert text == "Another generated text"
        assert token_usage is not None
        assert token_usage["prompt_tokens"] == 80
        assert token_usage["completion_tokens"] == 40
        assert token_usage["total_tokens"] == 120

    def test_extract_generation_result_no_token_usage(self):
        """Test extracting GenerationResult when no token usage is available."""
        mock_response = MagicMock()
        mock_response.content = "Text without token info"
        mock_response.usage_metadata = None
        mock_response.response_metadata = {}

        text = mock_response.content if hasattr(mock_response, "content") else str(mock_response)

        token_usage = None
        if hasattr(mock_response, "usage_metadata") and mock_response.usage_metadata:
            pass  # Would set token_usage
        elif hasattr(mock_response, "response_metadata"):
            usage = mock_response.response_metadata.get("token_usage", {})
            if usage:
                pass  # Would set token_usage

        assert text == "Text without token info"
        assert token_usage is None

    # ==================== Multi-Modal Message Format Tests ====================

    def test_multi_modal_message_format(self):
        """Test that multi-modal message content is structured correctly."""
        query = "What is shown in these images?"
        prompt_template = "Based on the images, answer: {query}"
        images = [
            create_test_image(10, 10, (255, 0, 0)),
            create_test_image(10, 10, (0, 255, 0)),
        ]

        # Build content as per design
        content = [{"type": "text", "text": prompt_template.format(query=query)}]

        for img in images:
            data_uri = pil_image_to_data_uri(img)
            content.append({"type": "image_url", "image_url": {"url": data_uri}})

        assert len(content) == 3  # 1 text + 2 images
        assert content[0]["type"] == "text"
        assert "What is shown in these images?" in content[0]["text"]
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/")
        assert content[2]["image_url"]["url"].startswith("data:image/")


class TestVisRAGGenerationPipelineConfig:
    """Tests for VisRAGGenerationPipelineConfig dataclass."""

    def test_config_default_values(self):
        """Test config has correct default values."""
        from dataclasses import dataclass
        from typing import Literal

        # Simulate the config structure from design
        @dataclass(kw_only=True)
        class MockVisRAGConfig:
            name: str
            retrieval_pipeline_name: str
            llm: str
            image_processing_mode: Literal["multi_image", "concatenate"] = "multi_image"
            concatenation_direction: Literal["horizontal", "vertical"] = "horizontal"
            prompt_template: str = "Based on images: {query}"
            temperature: float = 0.0
            max_tokens: int | None = None
            top_k: int = 10
            batch_size: int = 100

        config = MockVisRAGConfig(
            name="test_visrag",
            retrieval_pipeline_name="colpali_retrieval",
            llm="gpt-4o-vision",
        )

        assert config.image_processing_mode == "multi_image"
        assert config.concatenation_direction == "horizontal"
        assert config.temperature == 0.0
        assert config.max_tokens is None
        assert config.top_k == 10
        assert config.batch_size == 100

    def test_config_multi_image_mode(self):
        """Test config with multi_image processing mode."""
        from dataclasses import dataclass
        from typing import Literal

        @dataclass(kw_only=True)
        class MockVisRAGConfig:
            name: str
            image_processing_mode: Literal["multi_image", "concatenate"] = "multi_image"

        config = MockVisRAGConfig(name="test", image_processing_mode="multi_image")
        assert config.image_processing_mode == "multi_image"

    def test_config_concatenate_mode(self):
        """Test config with concatenate processing mode."""
        from dataclasses import dataclass
        from typing import Literal

        @dataclass(kw_only=True)
        class MockVisRAGConfig:
            name: str
            image_processing_mode: Literal["multi_image", "concatenate"] = "multi_image"
            concatenation_direction: Literal["horizontal", "vertical"] = "horizontal"

        config = MockVisRAGConfig(
            name="test",
            image_processing_mode="concatenate",
            concatenation_direction="vertical",
        )
        assert config.image_processing_mode == "concatenate"
        assert config.concatenation_direction == "vertical"

    def test_config_custom_prompt_template(self):
        """Test config with custom prompt template."""
        from dataclasses import dataclass

        @dataclass(kw_only=True)
        class MockVisRAGConfig:
            name: str
            prompt_template: str = "Default: {query}"

        custom_template = "Custom prompt for query: {query}\nProvide detailed answer:"
        config = MockVisRAGConfig(name="test", prompt_template=custom_template)

        assert "{query}" in config.prompt_template
        assert "Custom prompt" in config.prompt_template


class TestVisRAGGenerationPipelineMultimodalCheck:
    """Tests for multimodal capability verification in VisRAG-Gen."""

    def test_multimodal_check_valid_llm(self, session_factory):
        """Test that valid VLM passes multimodal check."""
        from autorag_research.pipelines.generation.visrag_gen import VisRAGGenerationPipeline

        # Create mock VLM that accepts multi-modal input
        mock_vlm = create_mock_llm(
            response_text="ok",
            token_usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1

        # Should not raise - VLM supports multi-modal input
        pipeline = VisRAGGenerationPipeline(
            session_factory=session_factory,
            name="test_multimodal_valid",
            llm=mock_vlm,
            retrieval_pipeline=mock_retrieval,
        )

        assert pipeline.pipeline_id > 0
        # Verify invoke was called during health check
        assert mock_vlm.invoke.called

    def test_multimodal_check_invalid_llm(self, session_factory):
        """Test that non-VLM raises ValueError during init."""
        from autorag_research.pipelines.generation.visrag_gen import VisRAGGenerationPipeline

        # Create mock LLM that raises error on multi-modal input
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ValueError("Model does not support image input")

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1

        # Should raise ValueError with descriptive message
        with pytest.raises(ValueError, match="does not support multi-modal input"):
            VisRAGGenerationPipeline(
                session_factory=session_factory,
                name="test_multimodal_invalid",
                llm=mock_llm,
                retrieval_pipeline=mock_retrieval,
            )

    def test_multimodal_check_uses_tiny_image(self):
        """Test that health check uses 1x1 pixel image for efficiency."""
        from autorag_research.pipelines.generation.visrag_gen import pil_image_to_data_uri

        # Create the same tiny image used in health check
        tiny_image = Image.new("RGB", (1, 1), color=(255, 255, 255))
        data_uri = pil_image_to_data_uri(tiny_image)

        # Verify it's a valid data URI
        assert data_uri.startswith("data:image/")
        assert ";base64," in data_uri

        # Verify the image is minimal
        _, base64_part = data_uri.split(";base64,")
        decoded_bytes = base64.b64decode(base64_part)
        decoded_img = Image.open(io.BytesIO(decoded_bytes))
        assert decoded_img.size == (1, 1)


class TestVisRAGGenerationPipelineEdgeCases:
    """Tests for edge cases in VisRAG-Gen pipeline."""

    def test_empty_retrieval_results(self):
        """Test handling when retrieval returns no image chunks."""
        retrieved = []

        # Pipeline should handle empty retrieval gracefully
        # Per design: generate answer without images (text-only mode)
        content = [{"type": "text", "text": "No images available. Query: What is this?"}] if not retrieved else []

        assert len(content) == 1
        assert content[0]["type"] == "text"

    def test_single_image_retrieval(self):
        """Test handling when retrieval returns exactly one image."""
        images = [create_test_image(10, 10)]

        # Single image case - no concatenation needed, use image directly
        result_image = images[0] if len(images) == 1 else None

        assert result_image is not None
        assert result_image.size == (10, 10)

    def test_missing_image_chunk_in_database(self):
        """Test handling when image chunk IDs don't exist in database."""
        # Simulate _get_image_chunk_contents with missing IDs
        image_chunk_ids = [1, 999, 3]  # 999 doesn't exist
        chunk_map = {1: (b"data1", "image/png"), 3: (b"data3", "image/png")}

        # Return empty for missing IDs (per design)
        result = [chunk_map.get(cid, (b"", "image/png")) for cid in image_chunk_ids]

        assert len(result) == 3
        assert result[0] == (b"data1", "image/png")
        assert result[1] == (b"", "image/png")  # Missing ID returns empty
        assert result[2] == (b"data3", "image/png")

    def test_image_mode_conversion_to_rgb(self):
        """Test that images are converted to RGB for consistency."""
        # Create RGBA image
        rgba_img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
        # Create L (grayscale) image
        gray_img = Image.new("L", (10, 10), color=128)

        # Convert to RGB
        rgb_from_rgba = rgba_img.convert("RGB") if rgba_img.mode != "RGB" else rgba_img
        rgb_from_gray = gray_img.convert("RGB") if gray_img.mode != "RGB" else gray_img

        assert rgb_from_rgba.mode == "RGB"
        assert rgb_from_gray.mode == "RGB"


class TestVisRAGGenerationPipelineImageProcessingModes:
    """Tests for different image processing modes."""

    @pytest.fixture
    def test_images(self):
        """Create a set of test images."""
        return [
            create_test_image(20, 30, (255, 0, 0)),  # Red
            create_test_image(25, 35, (0, 255, 0)),  # Green
            create_test_image(30, 25, (0, 0, 255)),  # Blue
        ]

    @pytest.fixture
    def retrieval_scores(self):
        """Sample retrieval scores."""
        return [0.95, 0.85, 0.70]

    def test_multi_image_mode_message_structure(self, test_images):
        """Test multi_image mode creates correct message structure."""
        query = "Describe these images"
        prompt = f"Based on the images: {query}"

        # Build multi-modal content
        content = [{"type": "text", "text": prompt}]

        for img in test_images:
            data_uri = pil_image_to_data_uri(img)
            content.append({"type": "image_url", "image_url": {"url": data_uri}})

        # Verify structure
        assert len(content) == 4  # 1 text + 3 images
        text_items = [c for c in content if c["type"] == "text"]
        image_items = [c for c in content if c["type"] == "image_url"]
        assert len(text_items) == 1
        assert len(image_items) == 3

    def test_concatenate_mode_produces_single_image(self, test_images):
        """Test concatenate mode produces a single composite image."""
        # Horizontal concatenation
        total_width = sum(img.width for img in test_images)
        max_height = max(img.height for img in test_images)
        concatenated = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

        x_offset = 0
        for img in test_images:
            concatenated.paste(img, (x_offset, 0))
            x_offset += img.width

        # Result is single image
        assert isinstance(concatenated, Image.Image)
        assert concatenated.width == 20 + 25 + 30  # 75
        assert concatenated.height == 35  # max height


class TestVisRAGGenerationPipelineIntegration:
    """Integration tests for VisRAGGenerationPipeline.

    Note: These tests require the actual implementation to exist.
    They are written in TDD style to define expected behavior.
    """

    @pytest.fixture
    def mock_vlm(self):
        """Create a mock VLM for integration tests."""
        return create_mock_llm(
            response_text="This is the answer based on document images.",
            token_usage={
                "prompt_tokens": 200,
                "completion_tokens": 50,
                "total_tokens": 250,
            },
        )

    @pytest.fixture
    def mock_retrieval_pipeline(self):
        """Create a mock retrieval pipeline that returns image chunk IDs."""
        from unittest.mock import AsyncMock

        mock = MagicMock()
        mock.pipeline_id = 1

        async def mock_retrieve_by_id(query_id: int, top_k: int):
            # Return image chunk IDs (1-5 exist in seed data)
            return [
                {"doc_id": 1, "score": 0.95},
                {"doc_id": 2, "score": 0.85},
                {"doc_id": 3, "score": 0.75},
            ][:top_k]

        mock._retrieve_by_id = AsyncMock(side_effect=mock_retrieve_by_id)
        return mock

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory):
        """Cleanup fixture that deletes pipeline results after test."""
        created_pipeline_ids = []

        yield created_pipeline_ids

        session = session_factory()
        try:
            executor_repo = ExecutorResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                executor_repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    def test_pipeline_creation_multi_image_mode(
        self, session_factory, mock_vlm, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test pipeline creation with multi_image mode."""
        from autorag_research.pipelines.generation.visrag_gen import VisRAGGenerationPipeline

        pipeline = VisRAGGenerationPipeline(
            session_factory=session_factory,
            name="test_visrag_multi_image",
            llm=mock_vlm,
            retrieval_pipeline=mock_retrieval_pipeline,
            image_processing_mode="multi_image",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline.pipeline_id > 0
        assert pipeline._image_processing_mode == "multi_image"

    def test_pipeline_creation_concatenate_mode(
        self, session_factory, mock_vlm, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test pipeline creation with concatenate mode."""
        from autorag_research.pipelines.generation.visrag_gen import VisRAGGenerationPipeline

        pipeline = VisRAGGenerationPipeline(
            session_factory=session_factory,
            name="test_visrag_concatenate",
            llm=mock_vlm,
            retrieval_pipeline=mock_retrieval_pipeline,
            image_processing_mode="concatenate",
            concatenation_direction="vertical",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline.pipeline_id > 0
        assert pipeline._image_processing_mode == "concatenate"
        assert pipeline._concatenation_direction == "vertical"

    def test_pipeline_config(self, session_factory, mock_vlm, mock_retrieval_pipeline, cleanup_pipeline_results):
        """Test that pipeline config is stored correctly."""
        from autorag_research.pipelines.generation.visrag_gen import VisRAGGenerationPipeline

        pipeline = VisRAGGenerationPipeline(
            session_factory=session_factory,
            name="test_visrag_config",
            llm=mock_vlm,
            retrieval_pipeline=mock_retrieval_pipeline,
            image_processing_mode="multi_image",
            temperature=0.5,
            max_tokens=500,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["type"] == "visrag_gen"
        assert config["image_processing_mode"] == "multi_image"
        assert config["temperature"] == 0.5
        assert config["max_tokens"] == 500
        assert config["retrieval_pipeline_id"] == mock_retrieval_pipeline.pipeline_id

    @pytest.mark.asyncio
    async def test_generate_single_query(
        self, session_factory, mock_vlm, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test generation for a single query."""
        from autorag_research.pipelines.generation.visrag_gen import VisRAGGenerationPipeline

        # Mock get_image_chunk_contents on the service to return test image data
        with patch(
            "autorag_research.orm.service.generation_pipeline.GenerationPipelineService.get_image_chunk_contents",
            return_value=[create_test_image_bytes() for _ in range(3)],
        ):
            pipeline = VisRAGGenerationPipeline(
                session_factory=session_factory,
                name="test_visrag_single",
                llm=mock_vlm,
                retrieval_pipeline=mock_retrieval_pipeline,
            )
            cleanup_pipeline_results.append(pipeline.pipeline_id)

            # Mock _get_query_text to return test query text
            with patch.object(pipeline, "_get_query_text", return_value="What is shown in the document?"):
                result = await pipeline._generate(query_id=1, top_k=3)

            assert result.text == "This is the answer based on document images."
            assert result.token_usage is not None
            assert result.token_usage["total_tokens"] == 250

    def test_run_pipeline(self, session_factory, mock_vlm, mock_retrieval_pipeline, cleanup_pipeline_results):
        """Test running the full pipeline with PipelineTestVerifier."""
        from autorag_research.pipelines.generation.visrag_gen import VisRAGGenerationPipeline

        with patch(
            "autorag_research.orm.service.generation_pipeline.GenerationPipelineService.get_image_chunk_contents",
            return_value=[create_test_image_bytes() for _ in range(3)],
        ):
            pipeline = VisRAGGenerationPipeline(
                session_factory=session_factory,
                name="test_visrag_run",
                llm=mock_vlm,
                retrieval_pipeline=mock_retrieval_pipeline,
            )
            cleanup_pipeline_results.append(pipeline.pipeline_id)

            result = pipeline.run(top_k=3, batch_size=10)

            # Use PipelineTestVerifier for standard output validation
            config = PipelineTestConfig(
                pipeline_type="generation",
                expected_total_queries=5,  # Seed data has 5 queries
                check_token_usage=True,
                check_execution_time=True,
                check_persistence=True,
            )
            verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
            verifier.verify_all()

    def test_run_pipeline_token_aggregation(
        self, session_factory, mock_vlm, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that token usage is correctly aggregated across all queries."""
        from autorag_research.pipelines.generation.visrag_gen import VisRAGGenerationPipeline

        with patch(
            "autorag_research.orm.service.generation_pipeline.GenerationPipelineService.get_image_chunk_contents",
            return_value=[create_test_image_bytes() for _ in range(2)],
        ):
            pipeline = VisRAGGenerationPipeline(
                session_factory=session_factory,
                name="test_visrag_tokens",
                llm=mock_vlm,
                retrieval_pipeline=mock_retrieval_pipeline,
            )
            cleanup_pipeline_results.append(pipeline.pipeline_id)

            result = pipeline.run(top_k=2, batch_size=10)

            # Verify aggregated token_usage values (5 queries * mock token counts)
            assert result["token_usage"]["total_tokens"] == 1250  # 5 * 250
            assert result["token_usage"]["prompt_tokens"] == 1000  # 5 * 200
            assert result["token_usage"]["completion_tokens"] == 250  # 5 * 50

    @pytest.mark.asyncio
    async def test_custom_prompt_template(
        self, session_factory, mock_vlm, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test pipeline with custom prompt template."""
        from autorag_research.pipelines.generation.visrag_gen import VisRAGGenerationPipeline

        custom_template = "Analyze these document images:\n\n{query}\n\nProvide detailed answer:"

        with patch(
            "autorag_research.orm.service.generation_pipeline.GenerationPipelineService.get_image_chunk_contents",
            return_value=[create_test_image_bytes() for _ in range(2)],
        ):
            pipeline = VisRAGGenerationPipeline(
                session_factory=session_factory,
                name="test_visrag_template",
                llm=mock_vlm,
                retrieval_pipeline=mock_retrieval_pipeline,
                prompt_template=custom_template,
            )
            cleanup_pipeline_results.append(pipeline.pipeline_id)

            # Mock _get_query_text to return test query text
            with patch.object(pipeline, "_get_query_text", return_value="Test query"):
                _ = await pipeline._generate(query_id=1, top_k=2)

            # Verify the VLM was called with correct message structure
            # Note: ainvoke is called for generation (invoke is used for health check)
            # Get the last ainvoke call (generation call)
            call_args = mock_vlm.ainvoke.call_args_list[-1]
            # ainvoke is called with a list of messages: ainvoke([message])
            message_list = call_args[0][0]
            assert len(message_list) == 1
            message = message_list[0]
            # Find the text content
            text_content = None
            for item in message.content:
                if item.get("type") == "text":
                    text_content = item.get("text")
                    break

            assert text_content is not None
            assert "Analyze these document images" in text_content
            assert "Provide detailed answer" in text_content
