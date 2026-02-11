"""ColPali embeddings for multi-vector multi-modal retrieval."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field, PrivateAttr

if TYPE_CHECKING:
    from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
    from transformers import PreTrainedModel

from autorag_research.embeddings.base import (
    MultiVectorEmbedding,
    MultiVectorMultiModalEmbedding,
)
from autorag_research.types import ImageType
from autorag_research.util import load_image

# Model type registry for Col* models: maps model_type to (model_class, processor_class)
COL_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "modernvbert": ("ColModernVBert", "ColModernVBertProcessor"),
    "smolvlm": ("ColIdefics3", "ColIdefics3Processor"),
    "pali": ("ColPali", "ColPaliProcessor"),
    "qwen2": ("ColQwen2", "ColQwen2Processor"),
    "qwen2_5": ("ColQwen2_5", "ColQwen2_5_Processor"),
}


def _load_col_model_classes(model_type: str) -> tuple[PreTrainedModel, BaseVisualRetrieverProcessor]:
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
        >>> text_emb = embeddings.embed_text("Hello world")  # list[list[float]]
        >>> image_emb = embeddings.embed_image("image.png")  # list[list[float]]
    """

    model_name: str = Field(description="HuggingFace model ID")
    model_type: str = Field(description="Model type (e.g., 'pali', 'modernvbert')")
    device: str = Field(default="cpu", description="Device to run the model on")
    torch_dtype: Any = Field(
        default="bfloat16", description="Torch dtype for model weights. String name or torch.dtype."
    )

    _model: Any = PrivateAttr(default=None)
    _processor: Any = PrivateAttr(default=None)

    # Class variable for supported model types
    SUPPORTED_MODEL_TYPES: ClassVar[list[str]] = list(COL_MODEL_REGISTRY.keys())

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._load_model()

    def _load_model(self) -> None:
        """Load the model and processor based on model_type."""
        import torch

        model_class, processor_class = _load_col_model_classes(self.model_type)

        resolved_dtype = getattr(torch, self.torch_dtype) if isinstance(self.torch_dtype, str) else self.torch_dtype

        self._processor = processor_class.from_pretrained(self.model_name)  # ty: ignore
        self._model = model_class.from_pretrained(
            self.model_name,
            dtype=resolved_dtype,
            trust_remote_code=True,
        )
        self._model.to(self.device)
        self._model.eval()

    def embed_text(self, text: str) -> MultiVectorEmbedding:
        """Embed a single text/document.

        Args:
            text: The text to embed.

        Returns:
            Multi-vector embedding as list[list[float]].
        """
        import torch

        text_inputs = self._processor.process_texts([text])
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            embeddings = self._model(**text_inputs)

        # Shape: (batch=1, num_tokens, hidden_dim) -> list[list[float]]
        return embeddings[0].cpu().tolist()

    async def aembed_text(self, text: str) -> MultiVectorEmbedding:
        """Embed a single text/document asynchronously.

        Args:
            text: The text to embed.

        Returns:
            Multi-vector embedding as list[list[float]].
        """
        return await asyncio.to_thread(self.embed_text, text)

    def embed_query(self, query: str) -> MultiVectorEmbedding:
        """Embed a single query string.

        Args:
            query: The query to embed.

        Returns:
            Multi-vector embedding as list[list[float]].
        """
        return self.embed_text(query)

    async def aembed_query(self, query: str) -> MultiVectorEmbedding:
        """Embed a single query string asynchronously.

        Args:
            query: The query to embed.

        Returns:
            Multi-vector embedding as list[list[float]].
        """
        return await asyncio.to_thread(self.embed_query, query)

    def embed_image(self, img_file_path: ImageType) -> MultiVectorEmbedding:
        """Embed a single image.

        Args:
            img_file_path: Path to image file or bytes.

        Returns:
            Multi-vector embedding as list[list[float]].
        """
        import torch

        image = _load_image(img_file_path)
        image_inputs = self._processor.process_images([image])
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        with torch.no_grad():
            embeddings = self._model(**image_inputs)

        # Shape: (batch=1, num_patches, hidden_dim) -> list[list[float]]
        return embeddings[0].cpu().tolist()

    async def aembed_image(self, img_file_path: ImageType) -> MultiVectorEmbedding:
        """Embed a single image asynchronously.

        Args:
            img_file_path: Path to image file or bytes.

        Returns:
            Multi-vector embedding as list[list[float]].
        """
        return await asyncio.to_thread(self.embed_image, img_file_path)

    def embed_documents(self, texts: list[str]) -> list[MultiVectorEmbedding]:
        """Embed multiple documents (GPU-optimized batch).

        Args:
            texts: List of texts to embed.

        Returns:
            List of multi-vector embeddings.
        """
        import torch

        if not texts:
            return []

        text_inputs = self._processor.process_texts(texts)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            embeddings = self._model(**text_inputs)

        # Shape: (batch, num_tokens, hidden_dim) -> list[list[list[float]]]
        return [emb.cpu().tolist() for emb in embeddings]

    def embed_images(self, img_file_paths: list[ImageType]) -> list[MultiVectorEmbedding]:
        """Embed multiple images (GPU-optimized batch).

        Args:
            img_file_paths: List of paths to image files or bytes.

        Returns:
            List of multi-vector embeddings.
        """
        import torch

        if not img_file_paths:
            return []

        images = [load_image(p) for p in img_file_paths]
        image_inputs = self._processor.process_images(images)
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        with torch.no_grad():
            embeddings = self._model(**image_inputs)

        # Shape: (batch, num_patches, hidden_dim) -> list[list[list[float]]]
        return [emb.cpu().tolist() for emb in embeddings]
