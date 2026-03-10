from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.language_models.fake import FakeListLLM

from autorag_research.pipelines.generation.autothinkrag import (
    AutoThinkRAGPipeline,
    AutoThinkRAGPipelineConfig,
)
from tests.autorag_research.pipelines.pipeline_test_utils import (
    cleanup_pipeline_results_factory,
    create_mock_retrieval_pipeline,
)


@pytest.fixture
def cleanup(session_factory):
    yield from cleanup_pipeline_results_factory(session_factory)


@pytest.fixture
def mock_retrieval():
    return create_mock_retrieval_pipeline(
        default_results=[
            {"doc_id": 1, "score": 0.95},
            {"doc_id": 2, "score": 0.85},
            {"doc_id": 3, "score": 0.75},
        ]
    )


class TestAutoThinkRAGPipeline:
    @pytest.mark.asyncio
    async def test_autothinkrag_simple_query(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(responses=["simple", "Simple answer"])

        pipeline = AutoThinkRAGPipeline(
            session_factory=session_factory,
            name="test_autothinkrag_simple",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=3)

        assert result.text == "Simple answer"
        assert result.metadata is not None
        assert result.metadata["complexity_tier"] == "simple"
        assert result.metadata["retrieved_chunk_ids"] == [1, 2, 3]
        assert result.metadata["retrieved_scores"] == [0.95, 0.85, 0.75]
        assert result.token_usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        assert mock_retrieval._retrieve_by_id.call_count == 1
        assert mock_retrieval.retrieve.call_count == 0

    @pytest.mark.asyncio
    async def test_autothinkrag_moderate_query(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(responses=["moderate", "Moderate answer"])
        vlm = FakeListLLM(responses=["Detected a chart showing quarterly growth."])

        pipeline = AutoThinkRAGPipeline(
            session_factory=session_factory,
            name="test_autothinkrag_moderate",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
            vlm=vlm,
        )
        cleanup.append(pipeline.pipeline_id)

        with (
            patch.object(
                pipeline._service,
                "get_image_chunk_contents",
                return_value=[(b"fake-image-bytes", "image/png")],
            ),
            patch(
                "autorag_research.pipelines.generation.autothinkrag.image_chunk_to_pil_images",
                return_value=["fake-image"],
            ),
            patch(
                "autorag_research.pipelines.generation.autothinkrag.pil_image_to_data_uri",
                return_value="data:image/png;base64,ZmFrZQ==",
            ),
        ):
            result = await pipeline._generate(query_id=1, top_k=3)

        assert result.text == "Moderate answer"
        assert result.metadata is not None
        assert result.metadata["complexity_tier"] == "moderate"
        assert result.metadata["visual_interpretation"] == "Detected a chart showing quarterly growth."
        assert result.metadata["retrieved_chunk_ids"] == [1, 2, 3]
        assert mock_retrieval._retrieve_by_id.call_count == 1
        assert mock_retrieval.retrieve.call_count == 0

    @pytest.mark.asyncio
    async def test_autothinkrag_complex_query(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(
            responses=[
                "complex",
                "Reasoning step 1",
                "Reasoning step 2",
                "Final complex answer",
            ]
        )
        mock_retrieval.retrieve = AsyncMock(
            side_effect=[
                [{"doc_id": 4, "score": 0.65}],
                [{"doc_id": 5, "score": 0.55}],
            ]
        )

        pipeline = AutoThinkRAGPipeline(
            session_factory=session_factory,
            name="test_autothinkrag_complex",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
            max_reasoning_steps=2,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=3)

        assert result.text == "Final complex answer"
        assert result.metadata is not None
        assert result.metadata["complexity_tier"] == "complex"
        assert result.metadata["reasoning_steps"] == ["Reasoning step 1", "Reasoning step 2"]
        assert result.metadata["retrieved_chunk_ids"] == [1, 2, 3, 4, 5]
        assert result.metadata["retrieved_scores"] == [0.95, 0.85, 0.75, 0.65, 0.55]
        assert mock_retrieval._retrieve_by_id.call_count == 1
        assert mock_retrieval.retrieve.call_count == 2

    def test_autothinkrag_config(self, mock_retrieval):
        llm = FakeListLLM(responses=["answer"])
        vlm = FakeListLLM(responses=["vision"])
        config = AutoThinkRAGPipelineConfig(
            name="autothinkrag_cfg",
            retrieval_pipeline_name="bm25",
            llm=llm,
            vlm=vlm,
            max_reasoning_steps=4,
            temperature=0.2,
            max_tokens=256,
        )
        config.inject_retrieval_pipeline(mock_retrieval)

        kwargs = config.get_pipeline_kwargs()

        assert config.get_pipeline_class() is AutoThinkRAGPipeline
        assert kwargs["llm"] is llm
        assert kwargs["vlm"] is vlm
        assert kwargs["retrieval_pipeline"] is mock_retrieval
        assert kwargs["max_reasoning_steps"] == 4
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 256
        assert "complexity_prompt_template" in kwargs
        assert "simple_prompt_template" in kwargs
        assert "complex_prompt_template" in kwargs
        assert "visual_interpretation_prompt_template" in kwargs

    def test_autothinkrag_pipeline_config_output(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(responses=["answer"])
        pipeline = AutoThinkRAGPipeline(
            session_factory=session_factory,
            name="test_autothinkrag_config_output",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
            max_reasoning_steps=5,
            temperature=0.1,
            max_tokens=512,
        )
        cleanup.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()

        assert config["type"] == "autothinkrag"
        assert config["retrieval_pipeline_id"] == mock_retrieval.pipeline_id
        assert config["max_reasoning_steps"] == 5
        assert config["temperature"] == 0.1
        assert config["max_tokens"] == 512
        assert config["complexity_tiers"] == ["simple", "moderate", "complex"]
        assert "complexity_prompt_template" in config
        assert "simple_prompt_template" in config
        assert "complex_prompt_template" in config
        assert "visual_interpretation_prompt_template" in config
