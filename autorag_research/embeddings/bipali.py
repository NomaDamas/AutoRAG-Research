"""BiPali embeddings for multi-modal retrieval."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import torch
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.core.schema import ImageType
from PIL import Image
from pydantic import Field, PrivateAttr

# Model type registry: maps model_type to (model_class, processor_class)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "modernvbert": ("BiModernVBert", "BiModernVBertProcessor"),
    "smolvlm": ("BiIdefics3", "BiIdefics3Processor"),
    "pali": ("BiPali", "BiPaliProcessor"),
    "qwen2": ("BiQwen2", "BiQwen2Processor"),
    "qwen2_5": ("BiQwen2_5", "BiQwen2_5_Processor"),
}


def _load_model_classes(model_type: str) -> tuple[type, type]:
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


class BiPaliEmbeddings(MultiModalEmbedding):
    """BiPali-style embeddings supporting multiple model types.

    This class provides a unified interface for BiEncoder models from colpali_engine
    that produce single-vector embeddings for both text and images.

    Supported model types:
    - "modernvbert": BiModernVBert
    - "idefics3": BiIdefics3
    - "pali": BiPali
    - "qwen2": BiQwen2
    - "qwen2_5": BiQwen2_5

    Example:
        >>> embeddings = BiPaliEmbeddings(
        ...     model_name="ModernVBERT/bimodernvbert",
        ...     model_type="modernvbert",
        ... )
        >>> text_emb = embeddings.get_text_embedding("Hello world")
        >>> image_emb = embeddings.get_image_embedding("image.png")
    """

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

        self._processor = processor_class.from_pretrained(self.model_name)
        self._model = model_class.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        )
        self._model.to(self.device)
        self._model.eval()

    def _get_image_embedding(self, img_file_path: ImageType) -> Embedding:
        """Get embedding for a single image."""
        image = _load_image(img_file_path)
        image_inputs = self._processor.process_images([image])

        # Move inputs to device
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        with torch.no_grad():
            embeddings = self._model(**image_inputs)

        # Shape: (batch, hidden_dim) -> take first item
        return embeddings[0].cpu().tolist()

    async def _aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        """Get image embedding asynchronously."""
        return await asyncio.to_thread(self._get_image_embedding, img_file_path)

    def _get_query_embedding(self, query: str) -> Embedding:
        """Get embedding for a query string."""
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Get query embedding asynchronously."""
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> Embedding:
        """Get embedding for a text string."""
        text_inputs = self._processor.process_texts([text])

        # Move inputs to device
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            embeddings = self._model(**text_inputs)

        # Shape: (batch, hidden_dim) -> take first item
        return embeddings[0].cpu().tolist()

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Get text embedding asynchronously."""
        return await asyncio.to_thread(self._get_text_embedding, text)

    def get_image_embedding_batch(
        self, img_file_paths: list[ImageType], show_progress: bool = False
    ) -> list[Embedding]:
        """Get embeddings for multiple images with batching."""
        all_embeddings: list[Embedding] = []

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

    def get_text_embedding_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[Embedding]:
        """Get embeddings for multiple texts with batching."""
        all_embeddings: list[Embedding] = []

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
