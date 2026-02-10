"""VisRAG-Gen (Vision-based RAG Generation) Pipeline for AutoRAG-Research.

Implements vision-language model (VLM) based generation that processes retrieved
document images directly without text parsing. Preserves visual information like
layout, formatting, tables, and embedded images that traditional text-based RAG loses.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseGenerationPipelineConfig
from autorag_research.orm.service.generation_pipeline import GenerationResult
from autorag_research.pipelines.generation.base import BaseGenerationPipeline
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.util import image_chunk_to_pil_images, pil_image_to_data_uri

DEFAULT_VISRAG_PROMPT = """Based on the provided document images, answer the following question:

{query}

Answer:"""

logger = logging.getLogger("AutoRAG-Research")


@dataclass(kw_only=True)
class VisRAGGenerationPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for VisRAG-Gen pipeline.

    Attributes:
        name: Unique name for this pipeline instance.
        retrieval_pipeline_name: Name of the retrieval pipeline (must return image chunks).
        llm: LangChain VLM with multi-modal support (string or instance).
        image_processing_mode: Strategy for handling multiple images:
            - "multi_image": Pass all images to VLM in single call (best performance)
            - "concatenate": Merge images into single composite (single-image VLM fallback)
        concatenation_direction: Direction for image concatenation ("horizontal" or "vertical").
        prompt_template: Template for VLM prompt with {query} placeholder.
        temperature: Sampling temperature for generation (0.0 = deterministic).
        max_tokens: Maximum tokens to generate (None = no limit).
        top_k: Number of image chunks to retrieve per query.
        batch_size: Number of queries to process in each batch.
    """

    image_processing_mode: Literal["multi_image", "concatenate"] = "multi_image"
    concatenation_direction: Literal["horizontal", "vertical"] = "horizontal"
    prompt_template: str = field(default=DEFAULT_VISRAG_PROMPT)
    temperature: float = 0.0
    max_tokens: int | None = None

    def get_pipeline_class(self) -> type["VisRAGGenerationPipeline"]:
        """Return the VisRAGGenerationPipeline class."""
        return VisRAGGenerationPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for VisRAGGenerationPipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)

        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "image_processing_mode": self.image_processing_mode,
            "concatenation_direction": self.concatenation_direction,
            "prompt_template": self.prompt_template,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


class VisRAGGenerationPipeline(BaseGenerationPipeline):
    """Vision-based RAG generation pipeline using VLM with document images.

    This pipeline processes retrieved document images directly through a vision-language
    model (VLM) to generate answers. It preserves visual information that would be
    lost in traditional text-based RAG, including:
    - Document layout and formatting
    - Tables and charts
    - Embedded images and diagrams
    - Handwritten content

    Two image processing modes are supported:
    1. multi_image: Pass all images to VLM in a single call (best for multi-image VLMs)
    2. concatenate: Merge images into a single composite image (fallback for single-image VLMs)

    Example:
        ```python
        from langchain_openai import ChatOpenAI

        from autorag_research.orm.connection import DBConnection
        from autorag_research.pipelines.generation.visrag_gen import VisRAGGenerationPipeline
        from autorag_research.pipelines.retrieval.colpali import ColPaliRetrievalPipeline

        db = DBConnection.from_config()
        session_factory = db.get_session_factory()

        # Create retrieval pipeline for image chunks
        retrieval_pipeline = ColPaliRetrievalPipeline(
            session_factory=session_factory,
            name="colpali_retrieval",
            model_path="/path/to/model",
        )

        # Create VisRAG generation pipeline
        pipeline = VisRAGGenerationPipeline(
            session_factory=session_factory,
            name="visrag_gen_v1",
            llm=ChatOpenAI(model="gpt-4o"),
            retrieval_pipeline=retrieval_pipeline,
            image_processing_mode="multi_image",
        )

        # Run pipeline
        results = pipeline.run(top_k=5)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseChatModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        image_processing_mode: Literal["multi_image", "concatenate"] = "multi_image",
        concatenation_direction: Literal["horizontal", "vertical"] = "horizontal",
        prompt_template: str = DEFAULT_VISRAG_PROMPT,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        schema: Any | None = None,
    ):
        """Initialize VisRAG-Gen pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            llm: LangChain VLM with multi-modal support (e.g., GPT-4o, Qwen2-VL).
            retrieval_pipeline: Retrieval pipeline for fetching relevant image chunks.
            image_processing_mode: Strategy for handling multiple images.
            concatenation_direction: Direction for image concatenation (used in concatenate mode).
            prompt_template: Template for VLM prompt with {query} placeholder.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens to generate (None = no limit).
            schema: Schema namespace from create_schema(). If None, uses default schema.

        Raises:
            ValueError: If the LLM does not support multi-modal input.
        """
        # CRITICAL: Store params BEFORE calling super().__init__()
        # because _get_pipeline_config() is called during base initialization
        self._image_processing_mode = image_processing_mode
        self._concatenation_direction = concatenation_direction
        self._prompt_template = prompt_template
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Health check: verify LLM supports multi-modal input
        self._verify_multimodal_support(llm)

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _verify_multimodal_support(self, llm: BaseChatModel) -> None:
        """Verify the LLM supports multi-modal input by testing with a tiny image.

        Creates a 1x1 pixel image and attempts to invoke the LLM with it.
        Runs once during initialization as a health check.

        Args:
            llm: The LangChain chat model to verify.

        Raises:
            ValueError: If the LLM does not support multi-modal input.
        """
        # Create smallest possible image (1x1 pixel, white)
        tiny_image = Image.new("RGB", (1, 1), color=(255, 255, 255))
        data_uri = pil_image_to_data_uri(tiny_image)

        # Build multi-modal message
        content: list[str | dict] = [
            {"type": "text", "text": "Health check. Reply with 'ok'."},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]
        message = HumanMessage(content=content)

        try:
            llm.invoke([message])
        except Exception as e:
            msg = (
                f"LLM '{type(llm).__name__}' does not support multi-modal input. "
                f"VisRAG-Gen requires a Vision Language Model (VLM) such as GPT-4o, "
                f"Qwen2-VL, or Claude 3. Error: {e}"
            )
            raise ValueError(msg) from e

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return VisRAG-Gen pipeline configuration."""
        return {
            "type": "visrag_gen",
            "image_processing_mode": self._image_processing_mode,
            "concatenation_direction": self._concatenation_direction,
            "prompt_template": self._prompt_template,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "retrieval_pipeline_id": self._retrieval_pipeline.pipeline_id,
        }

    def _extract_generation_result(self, response: Any) -> GenerationResult:
        """Extract GenerationResult from LangChain VLM response.

        Args:
            response: LangChain response object.

        Returns:
            GenerationResult containing text and token usage.
        """
        text = response.content if hasattr(response, "content") else str(response)

        token_usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            token_usage = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        elif hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("token_usage", {})
            if usage:
                token_usage = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

        return GenerationResult(text=text, token_usage=token_usage)

    def _concatenate_images(
        self,
        images: list[Image.Image],
        direction: Literal["horizontal", "vertical"],
    ) -> Image.Image:
        """Concatenate multiple PIL Images into single composite image.

        Args:
            images: List of PIL Image objects.
            direction: Concatenation direction ("horizontal" or "vertical").

        Returns:
            Single concatenated PIL Image.

        Raises:
            ValueError: If no images provided.
        """
        if not images:
            msg = "No images to concatenate"
            raise ValueError(msg)

        if len(images) == 1:
            return images[0]

        if direction == "horizontal":
            total_width = sum(img.width for img in images)
            max_height = max(img.height for img in images)
            concatenated = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))
            x_offset = 0
            for img in images:
                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")
                concatenated.paste(img, (x_offset, 0))
                x_offset += img.width
        else:  # vertical
            max_width = max(img.width for img in images)
            total_height = sum(img.height for img in images)
            concatenated = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))
            y_offset = 0
            for img in images:
                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")
                concatenated.paste(img, (0, y_offset))
                y_offset += img.height

        return concatenated

    async def _generate_single_image(self, query: str, image: Image.Image) -> GenerationResult:
        """Generate answer using VLM with a single image.

        Args:
            query: The query text.
            image: Single PIL Image.

        Returns:
            GenerationResult from VLM.
        """
        data_uri = pil_image_to_data_uri(image)
        content: list[str | dict] = [
            {"type": "text", "text": self._prompt_template.format(query=query)},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]

        message = HumanMessage(content=content)
        response = await self._llm.ainvoke([message])
        return self._extract_generation_result(response)

    async def _generate_multi_image(self, query: str, images: list[Image.Image]) -> GenerationResult:
        """Generate answer using VLM with multiple images in single call.

        Best performance for VLMs supporting multi-image input (GPT-4o, Qwen2-VL, etc.).
        Enables cross-image reasoning and multi-hop QA.

        Args:
            query: The query text.
            images: List of PIL Images.

        Returns:
            GenerationResult from VLM.
        """
        content: list[str | dict] = [{"type": "text", "text": self._prompt_template.format(query=query)}]

        for img in images:
            data_uri = pil_image_to_data_uri(img)
            content.append({"type": "image_url", "image_url": {"url": data_uri}})

        message = HumanMessage(content=content)
        response = await self._llm.ainvoke([message])
        return self._extract_generation_result(response)

    async def _generate_concatenated(self, query: str, images: list[Image.Image]) -> GenerationResult:
        """Generate answer by concatenating images into single composite image.

        Fallback for VLMs accepting only one image. Performance degrades with many
        images due to resolution constraints.

        Args:
            query: The query text.
            images: List of PIL Images.

        Returns:
            GenerationResult from VLM.
        """
        concatenated = self._concatenate_images(images, self._concatenation_direction)
        return await self._generate_single_image(query, concatenated)

    async def _generate_text_only_fallback(self, query: str, metadata_key: str) -> GenerationResult:
        """Generate answer using text-only mode when no valid images are available.

        Args:
            query: The query text.
            metadata_key: Key to set in metadata indicating fallback reason.

        Returns:
            GenerationResult with metadata indicating fallback mode.
        """
        content: list[str | dict] = [{"type": "text", "text": self._prompt_template.format(query=query)}]
        message = HumanMessage(content=content)
        response = await self._llm.ainvoke([message])
        result = self._extract_generation_result(response)
        if result.metadata is None:
            result.metadata = {}
        result.metadata[metadata_key] = True
        return result

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate answer using VisRAG-Gen with retrieved document images.

        Algorithm Flow:
        1. Get query text from database
        2. Retrieve top-k document images using retrieval pipeline (async)
        3. Fetch image chunks (PIL Images) and retrieval scores from database
        4. Apply image processing based on mode:
           a. multi_image: Pass all images to VLM in single call
           b. concatenate: Merge images into single composite image
        5. Construct multi-modal LangChain message with images
        6. Call VLM to generate answer (async)
        7. Return GenerationResult with text, token usage, metadata

        Args:
            query_id: The query ID to answer.
            top_k: Number of image chunks to retrieve.

        Returns:
            GenerationResult containing generated text, token usage, and metadata.
        """
        # 1. Get query text from database
        query = self._service.get_query_text(query_id)

        # 2. Retrieve relevant image chunks using composed retrieval pipeline (async)
        retrieved = await self._retrieval_pipeline._retrieve_by_id(query_id, top_k)

        # Handle empty retrieval results (text-only fallback)
        if not retrieved:
            logger.warning("No images retrieved for query_id: %s", query_id)
            return await self._generate_text_only_fallback(query, "no_images_retrieved")

        # 3. Get image chunk contents via service layer
        image_chunk_ids = [r["doc_id"] for r in retrieved]
        retrieval_scores = [r["score"] for r in retrieved]
        image_contents = self._service.get_image_chunk_contents(image_chunk_ids)

        # 4. Convert to PIL Images (skip empty/invalid images)
        images = image_chunk_to_pil_images(image_contents)

        # Handle case where all images are invalid
        if not images:
            return await self._generate_text_only_fallback(query, "all_images_invalid")

        # 5. Apply image processing based on mode
        if self._image_processing_mode == "multi_image":
            result = await self._generate_multi_image(query, images)
        elif self._image_processing_mode == "concatenate":
            result = await self._generate_concatenated(query, images)
        else:
            msg = f"Unknown image_processing_mode: {self._image_processing_mode}"
            raise ValueError(msg)

        # 6. Add metadata about retrieval
        if result.metadata is None:
            result.metadata = {}
        result.metadata["retrieved_image_chunk_ids"] = image_chunk_ids
        result.metadata["retrieved_scores"] = retrieval_scores

        return result
