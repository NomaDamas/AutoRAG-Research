from unittest.mock import AsyncMock, MagicMock, patch

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
        # Simple path: single _retrieve_by_id(top_k=3), no retrieve()
        mock_retrieval._retrieve_by_id.assert_awaited_once_with(1, 3)
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
        # Moderate path: single _retrieve_by_id(top_k=3, no reranker so no over-fetch)
        mock_retrieval._retrieve_by_id.assert_awaited_once_with(1, 3)
        assert mock_retrieval.retrieve.call_count == 0

    @pytest.mark.asyncio
    async def test_autothinkrag_complex_query(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(
            responses=[
                "complex",
                "1. What is X?\n2. How does Y work?",  # decomposition
                "Reasoning step 1",
                "Reasoning step 2",
                "Final complex answer",
            ]
        )
        mock_retrieval.retrieve = AsyncMock(
            side_effect=[
                [{"doc_id": 4, "score": 0.65}],  # sub-query 1
                [{"doc_id": 5, "score": 0.55}],  # sub-query 2
                [{"doc_id": 6, "score": 0.45}],  # reasoning step 1 follow-up
                [{"doc_id": 7, "score": 0.35}],  # reasoning step 2 follow-up
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
        assert result.metadata["sub_queries"] == ["What is X?", "How does Y work?"]
        assert result.metadata["reasoning_steps"] == ["Reasoning step 1", "Reasoning step 2"]
        assert result.metadata["retrieved_chunk_ids"] == [1, 2, 3, 4, 5, 6, 7]
        # _retrieve_by_id for original query + retrieve for 2 sub-queries + 2 reasoning steps
        assert mock_retrieval._retrieve_by_id.call_count == 1
        assert mock_retrieval.retrieve.call_count == 4

    @pytest.mark.asyncio
    async def test_autothinkrag_moderate_with_reranker(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(responses=["moderate", "Reranked moderate answer"])

        mock_reranker = MagicMock()
        mock_reranker.model_name = "mock-reranker"
        mock_reranker.arerank = AsyncMock(
            return_value=[
                MagicMock(index=1, score=0.99),  # doc_id=2 ranked first
                MagicMock(index=0, score=0.80),  # doc_id=1 ranked second
            ]
        )

        pipeline = AutoThinkRAGPipeline(
            session_factory=session_factory,
            name="test_autothinkrag_moderate_rerank",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
            reranker=mock_reranker,
            fetch_k_multiplier=2,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=2)

        assert result.text == "Reranked moderate answer"
        assert result.metadata["complexity_tier"] == "moderate"
        # Moderate with reranker: _retrieve_by_id(top_k * multiplier = 4)
        mock_retrieval._retrieve_by_id.assert_awaited_once_with(1, 4)
        # Reranker was called
        mock_reranker.arerank.assert_awaited_once()
        # Reranked order: doc_id 2 first, then doc_id 1
        assert result.metadata["retrieved_chunk_ids"] == [2, 1]
        assert result.metadata["retrieved_scores"] == [0.99, 0.80]

    @pytest.mark.asyncio
    async def test_autothinkrag_simple_skips_reranker(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(responses=["simple", "Direct answer"])

        mock_reranker = MagicMock()
        mock_reranker.model_name = "mock-reranker"
        mock_reranker.arerank = AsyncMock()

        pipeline = AutoThinkRAGPipeline(
            session_factory=session_factory,
            name="test_autothinkrag_simple_no_rerank",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
            reranker=mock_reranker,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=3)

        assert result.text == "Direct answer"
        assert result.metadata["complexity_tier"] == "simple"
        # Simple path: reranker is NOT called, retrieval uses exact top_k
        mock_reranker.arerank.assert_not_awaited()
        mock_retrieval._retrieve_by_id.assert_awaited_once_with(1, 3)

    @pytest.mark.asyncio
    async def test_autothinkrag_decomposition_deduplicates(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(
            responses=[
                "complex",
                # LLM returns duplicates and the original query text
                "1. What is the query about?\n2. What is the query about?\n3. A unique sub-question?",
                "Step 1",
                "Final answer",
            ]
        )
        mock_retrieval.retrieve = AsyncMock(
            side_effect=[
                [{"doc_id": 4, "score": 0.65}],  # sub-query "What is the query about?"
                [{"doc_id": 5, "score": 0.55}],  # sub-query "A unique sub-question?"
                [{"doc_id": 6, "score": 0.45}],  # reasoning follow-up
            ]
        )

        pipeline = AutoThinkRAGPipeline(
            session_factory=session_factory,
            name="test_autothinkrag_dedup",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
            max_reasoning_steps=1,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=3)

        # Deduplicated: "What is the query about?" appears once, plus "A unique sub-question?"
        assert len(result.metadata["sub_queries"]) == 2
        # retrieve called: 2 sub-queries + 1 reasoning step = 3
        assert mock_retrieval.retrieve.call_count == 3

    def test_autothinkrag_config(self, mock_retrieval):
        llm = FakeListLLM(responses=["answer"])
        vlm = FakeListLLM(responses=["vision"])
        config = AutoThinkRAGPipelineConfig(
            name="autothinkrag_cfg",
            retrieval_pipeline_name="bm25",
            llm=llm,
            vlm=vlm,
            max_reasoning_steps=4,
            max_subquestions=5,
            fetch_k_multiplier=3,
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
        assert kwargs["max_subquestions"] == 5
        assert kwargs["fetch_k_multiplier"] == 3
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 256
        assert "complexity_prompt_template" in kwargs
        assert "simple_prompt_template" in kwargs
        assert "moderate_prompt_template" in kwargs
        assert "complex_prompt_template" in kwargs
        assert "decomposition_prompt_template" in kwargs
        assert "visual_interpretation_prompt_template" in kwargs

    def test_autothinkrag_pipeline_config_output(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(responses=["answer"])
        pipeline = AutoThinkRAGPipeline(
            session_factory=session_factory,
            name="test_autothinkrag_config_output",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
            max_reasoning_steps=5,
            max_subquestions=4,
            fetch_k_multiplier=3,
            temperature=0.1,
            max_tokens=512,
        )
        cleanup.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()

        assert config["type"] == "autothinkrag"
        assert config["retrieval_pipeline_id"] == mock_retrieval.pipeline_id
        assert config["max_reasoning_steps"] == 5
        assert config["max_subquestions"] == 4
        assert config["fetch_k_multiplier"] == 3
        assert config["temperature"] == 0.1
        assert config["max_tokens"] == 512
        assert config["complexity_tiers"] == ["simple", "moderate", "complex"]
        assert config["reranker"] is None
        assert "complexity_prompt_template" in config
        assert "simple_prompt_template" in config
        assert "moderate_prompt_template" in config
        assert "complex_prompt_template" in config
        assert "decomposition_prompt_template" in config
        assert "visual_interpretation_prompt_template" in config
