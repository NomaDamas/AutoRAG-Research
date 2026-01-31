"""BiPali embeddings for multi-modal retrieval."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import torch
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from langchain_core.embeddings import Embeddings
from PIL import Image
from pydantic import ConfigDict, Field, PrivateAttr
from transformers import PreTrainedModel

from autorag_research.types import ImageType

# Model type registry: maps model_type to (model_class, processor_class)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "modernvbert": ("BiModernVBert", "BiModernVBertProcessor"),
    "smolvlm": ("BiIdefics3", "BiIdefics3Processor"),
    "pali": ("BiPali", "BiPaliProcessor"),
    "qwen2": ("BiQwen2", "BiQwen2Processor"),
    "qwen2_5": ("BiQwen2_5", "BiQwen2_5_Processor"),
}


def _load_model_classes(model_type: str) -> tuple[PreTrainedModel, BaseVisualRetrieverProcessor]:
    """Dynamically load model and processor classes from colpali_engine."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'. Supported: {list(MODEL_REGISTRY.keys())}")  # noqa: TRY003

    model_class_name, processor_class_name = MODEL_REGISTRY[model_type]

    try:
        import colpali_engine.models as models_module

        model_class = getattr(models_module, model_class_name)
        processor_class = getattr(models_module, processor_class_name)
    except ImportError as e:
        raise ImportError(  # noqa: TRY003
            "colpali_engine is required for BiPaliEmbeddings. Install it with: pip install colpali-engine"
        ) from e
    except AttributeError as e:
        raise AttributeError(  # noqa: TRY003
            f"Could not find {model_class_name} or {processor_class_name} in colpali_engine.models"
        ) from e

    return model_class, processor_class


def _load_image(img_file_path: ImageType) -> Image.Image:
    """Load an image from file path or bytes."""
    if isinstance(img_file_path, str):
        return Image.open(img_file_path).convert("RGB")
    elif isinstance(img_file_path, bytes):
        import io

        return Image.open(io.BytesIO(img_file_path)).convert("RGB")
    else:
        raise TypeError(img_file_path)


class BiPaliEmbeddings(Embeddings):
    """BiPali-style embeddings supporting multiple model types.

    This class provides a unified interface for BiEncoder models from colpali_engine
    that produce single-vector embeddings for both text and images.

    Implements the LangChain Embeddings interface with additional image embedding support.

    Supported model types:
    - "modernvbert": BiModernVBert
    - "smolvlm": BiIdefics3
    - "pali": BiPali
    - "qwen2": BiQwen2
    - "qwen2_5": BiQwen2_5

    Example:
        >>> embeddings = BiPaliEmbeddings(
        ...     model_name="ModernVBERT/bimodernvbert",
        ...     model_type="modernvbert",
        ... )
        >>> text_emb = embeddings.embed_query("Hello world")
        >>> image_emb = embeddings.embed_image("image.png")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = Field(description="HuggingFace model ID")
    model_type: str = Field(description="Model type (e.g., 'modernvbert')")
    device: str = Field(default="cpu", description="Device to run the model on")
    torch_dtype: torch.dtype = Field(default=torch.bfloat16, description="Torch dtype for model weights")
    embed_batch_size: int = Field(default=10, description="Batch size for embedding")

    _model: Any = PrivateAttr(default=None)
    _processor: Any = PrivateAttr(default=None)

    # Class variable for supported model types
    SUPPORTED_MODEL_TYPES: ClassVar[list[str]] = list(MODEL_REGISTRY.keys())

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._load_model()

    def _load_model(self) -> None:
        """Load the model and processor based on model_type."""
        model_class, processor_class = _load_model_classes(self.model_type)

        self._processor = processor_class.from_pretrained(self.model_name)  # ty: ignore
        self._model = model_class.from_pretrained(
            self.model_name,
            dtype=self.torch_dtype,
            trust_remote_code=True,
        )
        self._model.to(self.device)
        self._model.eval()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query/text string.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        return self._embed_text(text)

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single query/text string asynchronously.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        return await asyncio.to_thread(self.embed_query, text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents/texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return self._embed_texts_batch(texts)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents/texts asynchronously.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return await asyncio.to_thread(self.embed_documents, texts)

    def embed_image(self, img_file_path: ImageType) -> list[float]:
        """Embed a single image.

        Args:
            img_file_path: Path to image file or bytes.

        Returns:
            Embedding vector as list of floats.
        """
        image = _load_image(img_file_path)
        image_inputs = self._processor.process_images([image])

        # Move inputs to device
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        with torch.no_grad():
            embeddings = self._model(**image_inputs)

        # Shape: (batch, hidden_dim) -> take first item
        return embeddings[0].cpu().tolist()

    async def aembed_image(self, img_file_path: ImageType) -> list[float]:
        """Embed a single image asynchronously.

        Args:
            img_file_path: Path to image file or bytes.

        Returns:
            Embedding vector as list of floats.
        """
        return await asyncio.to_thread(self.embed_image, img_file_path)

    def embed_images(self, img_file_paths: list[ImageType]) -> list[list[float]]:
        """Embed multiple images with batching.

        Args:
            img_file_paths: List of paths to image files or bytes.

        Returns:
            List of embedding vectors.
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(img_file_paths), self.embed_batch_size):
            batch_paths = img_file_paths[i : i + self.embed_batch_size]
            images = [_load_image(p) for p in batch_paths]

            image_inputs = self._processor.process_images(images)
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

            with torch.no_grad():
                embeddings = self._model(**image_inputs)

            # Shape: (batch, hidden_dim)
            batch_embeddings = embeddings.cpu().tolist()
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _embed_text(self, text: str) -> list[float]:
        """Internal method to embed a single text."""
        text_inputs = self._processor.process_texts([text])

        # Move inputs to device
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            embeddings = self._model(**text_inputs)

        # Shape: (batch, hidden_dim) -> take first item
        return embeddings[0].cpu().tolist()

    def _embed_texts_batch(self, texts: list[str]) -> list[list[float]]:
        """Internal method to embed multiple texts with batching."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.embed_batch_size):
            batch_texts = texts[i : i + self.embed_batch_size]

            text_inputs = self._processor.process_texts(batch_texts)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

            with torch.no_grad():
                embeddings = self._model(**text_inputs)

            # Shape: (batch, hidden_dim)
            batch_embeddings = embeddings.cpu().tolist()
            all_embeddings.extend(batch_embeddings)

        return all_embeddings
