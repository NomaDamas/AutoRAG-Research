"""ColPali embeddings for multi-vector multi-modal retrieval."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import torch
from llama_index.core.schema import ImageType
from pydantic import Field, PrivateAttr

from autorag_research.embeddings.base import (
    MultiVectorEmbedding,
    MultiVectorMultiModalEmbedding,
)
from autorag_research.embeddings.bipali import _load_image

# Model type registry for Col* models: maps model_type to (model_class, processor_class)
COL_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "modernvbert": ("ColModernVBert", "ColModernVBertProcessor"),
    "smolvlm": ("ColIdefics3", "ColIdefics3Processor"),
    "pali": ("ColPali", "ColPaliProcessor"),
    "qwen2": ("ColQwen2", "ColQwen2Processor"),
    "qwen2_5": ("ColQwen2_5", "ColQwen2_5_Processor"),
}


def _load_col_model_classes(model_type: str) -> tuple[type, type]:
    """Dynamically load Col* model and processor classes from colpali_engine."""
    if model_type not in COL_MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'. Supported: {list(COL_MODEL_REGISTRY.keys())}")  # noqa: TRY003

    model_class_name, processor_class_name = COL_MODEL_REGISTRY[model_type]

    try:
        import colpali_engine.models as models_module

        model_class = getattr(models_module, model_class_name)
        processor_class = getattr(models_module, processor_class_name)
    except ImportError as e:
        raise ImportError(  # noqa: TRY003
            "colpali_engine is required for ColPaliEmbeddings. Install it with: pip install colpali-engine"
        ) from e
    except AttributeError as e:
        raise AttributeError(  # noqa: TRY003
            f"Could not find {model_class_name} or {processor_class_name} in colpali_engine.models"
        ) from e

    return model_class, processor_class


class ColPaliEmbeddings(MultiVectorMultiModalEmbedding):
    """ColPali-style embeddings supporting multiple model types.

    This class provides a unified interface for ColEncoder models from colpali_engine
    that produce multi-vector embeddings (one vector per token/patch) for late interaction retrieval.

    Supported model types:
    - "modernvbert": ColModernVBert
    - "smolvlm": ColIdefics3
    - "pali": ColPali
    - "qwen2": ColQwen2
    - "qwen2_5": ColQwen2_5

    Example:
        >>> embeddings = ColPaliEmbeddings(
        ...     model_name="vidore/colpali-v1.3",
        ...     model_type="pali",
        ... )
        >>> text_emb = embeddings.get_text_embedding("Hello world")  # list[list[float]]
        >>> image_emb = embeddings.get_image_embedding("image.png")  # list[list[float]]
    """

    model_name: str = Field(description="HuggingFace model ID")
    model_type: str = Field(description="Model type (e.g., 'pali', 'modernvbert')")
    device: str = Field(default="cpu", description="Device to run the model on")
    torch_dtype: torch.dtype = Field(default=torch.bfloat16, description="Torch dtype for model weights")

    _model: Any = PrivateAttr(default=None)
    _processor: Any = PrivateAttr(default=None)

    # Class variable for supported model types
    SUPPORTED_MODEL_TYPES: ClassVar[list[str]] = list(COL_MODEL_REGISTRY.keys())

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._load_model()

    def _load_model(self) -> None:
        """Load the model and processor based on model_type."""
        model_class, processor_class = _load_col_model_classes(self.model_type)

        self._processor = processor_class.from_pretrained(self.model_name)
        self._model = model_class.from_pretrained(
            self.model_name,
            dtype=self.torch_dtype,
            trust_remote_code=True,
        )
        self._model.to(self.device)
        self._model.eval()

    def get_text_embedding(self, text: str) -> MultiVectorEmbedding:
        """Get multi-vector embedding for a text string."""
        text_inputs = self._processor.process_texts([text])
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            embeddings = self._model(**text_inputs)

        # Shape: (batch=1, num_tokens, hidden_dim) -> list[list[float]]
        return embeddings[0].cpu().tolist()

    async def aget_text_embedding(self, text: str) -> MultiVectorEmbedding:
        """Get text embedding asynchronously."""
        return await asyncio.to_thread(self.get_text_embedding, text)

    def get_query_embedding(self, query: str) -> MultiVectorEmbedding:
        """Get multi-vector embedding for a query string."""
        return self.get_text_embedding(query)

    async def aget_query_embedding(self, query: str) -> MultiVectorEmbedding:
        """Get query embedding asynchronously."""
        return await asyncio.to_thread(self.get_query_embedding, query)

    def get_image_embedding(self, img_file_path: ImageType) -> MultiVectorEmbedding:
        """Get multi-vector embedding for a single image."""
        image = _load_image(img_file_path)
        image_inputs = self._processor.process_images([image])
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        with torch.no_grad():
            embeddings = self._model(**image_inputs)

        # Shape: (batch=1, num_patches, hidden_dim) -> list[list[float]]
        return embeddings[0].cpu().tolist()

    async def aget_image_embedding(self, img_file_path: ImageType) -> MultiVectorEmbedding:
        """Get image embedding asynchronously."""
        return await asyncio.to_thread(self.get_image_embedding, img_file_path)

    def get_text_embeddings(self, texts: list[str]) -> list[MultiVectorEmbedding]:
        """Get multi-vector embeddings for multiple texts (GPU-optimized batch)."""
        if not texts:
            return []

        text_inputs = self._processor.process_texts(texts)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            embeddings = self._model(**text_inputs)

        # Shape: (batch, num_tokens, hidden_dim) -> list[list[list[float]]]
        return [emb.cpu().tolist() for emb in embeddings]

    def get_image_embeddings(self, img_file_paths: list[ImageType]) -> list[MultiVectorEmbedding]:
        """Get multi-vector embeddings for multiple images (GPU-optimized batch)."""
        if not img_file_paths:
            return []

        images = [_load_image(p) for p in img_file_paths]
        image_inputs = self._processor.process_images(images)
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        with torch.no_grad():
            embeddings = self._model(**image_inputs)

        # Shape: (batch, num_patches, hidden_dim) -> list[list[list[float]]]
        return [emb.cpu().tolist() for emb in embeddings]
