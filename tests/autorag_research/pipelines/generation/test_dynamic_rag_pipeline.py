"""Tests for DynamicRAG generation pipeline."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake import FakeListLLM

from autorag_research.pipelines.generation.dynamic_rag import DynamicRAGPipeline, DynamicRAGPipelineConfig
from autorag_research.rerankers.base import BaseReranker, RerankResult
from autorag_research.rerankers.dynamic_rag import DynamicRAGReranker
from tests.autorag_research.pipelines.pipeline_test_utils import create_mock_retrieval_pipeline


class FixedAsyncReranker(BaseReranker):
    """Async reranker with fixed output."""

    results: list[RerankResult]

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        return self.results[:top_k] if top_k is not None else self.results

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        return self.rerank(query, documents, top_k)


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.content = content
    response.usage_metadata = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    return response


def _build_unit_pipeline(llm: Any, retrieval_pipeline: Any, service: Any, **kwargs: Any) -> DynamicRAGPipeline:
    with patch("autorag_research.pipelines.generation.base.BaseGenerationPipeline.__init__", return_value=None):
        pipeline = DynamicRAGPipeline(
            session_factory=MagicMock(),
            name="dynamic_rag_unit",
            llm=llm,
            retrieval_pipeline=retrieval_pipeline,
            **kwargs,
        )
    pipeline._llm = llm
    pipeline._retrieval_pipeline = retrieval_pipeline
    pipeline._service = service
    pipeline.pipeline_id = 1
    return pipeline


class TestDynamicRAGPipelineConfig:
    """Tests for DynamicRAGPipelineConfig."""

    def test_get_pipeline_class(self):
        config = DynamicRAGPipelineConfig(
            name="dynamic_rag",
            llm=FakeListLLM(responses=["answer"]),
            retrieval_pipeline_name="bm25",
        )

        assert config.get_pipeline_class() == DynamicRAGPipeline

    def test_get_pipeline_kwargs_requires_injected_retrieval_pipeline(self):
        config = DynamicRAGPipelineConfig(
            name="dynamic_rag",
            llm=FakeListLLM(responses=["answer"]),
            retrieval_pipeline_name="bm25",
        )

        with pytest.raises(ValueError, match="not injected"):
            config.get_pipeline_kwargs()

    def test_get_pipeline_kwargs_after_injection(self):
        retrieval_pipeline = create_mock_retrieval_pipeline(pipeline_id=55)
        llm = FakeListLLM(responses=["answer"])
        reranker = FixedAsyncReranker(results=[])
        config = DynamicRAGPipelineConfig(
            name="dynamic_rag",
            llm=llm,
            retrieval_pipeline_name="bm25",
            reranker=reranker,
            candidate_top_k=30,
        )

        config.inject_retrieval_pipeline(retrieval_pipeline)
        kwargs = config.get_pipeline_kwargs()

        assert kwargs["llm"] is llm
        assert kwargs["retrieval_pipeline"] is retrieval_pipeline
        assert kwargs["reranker"] is reranker
        assert kwargs["candidate_top_k"] == 30


class TestDynamicRAGPipeline:
    """Tests for DynamicRAG generation behavior."""

    def test_initialization_rejects_invalid_candidate_top_k(self):
        with (
            patch("autorag_research.pipelines.generation.base.BaseGenerationPipeline.__init__", return_value=None),
            pytest.raises(ValueError, match="candidate_top_k must be >= 1"),
        ):
            DynamicRAGPipeline(
                session_factory=MagicMock(),
                name="invalid_dynamic_rag",
                llm=FakeListLLM(responses=["answer"]),
                retrieval_pipeline=create_mock_retrieval_pipeline(),
                candidate_top_k=0,
            )

    def test_pipeline_config_distinguishes_dynamic_rag_cut_policy(self):
        llm = FakeListLLM(responses=["answer"])
        retrieval_pipeline = create_mock_retrieval_pipeline(pipeline_id=77)
        service = MagicMock()
        conservative = _build_unit_pipeline(
            llm,
            retrieval_pipeline,
            service,
            reranker=DynamicRAGReranker(score_drop_threshold=0.10, min_top_k=1, max_top_k=8),
        )
        permissive = _build_unit_pipeline(
            llm,
            retrieval_pipeline,
            service,
            reranker=DynamicRAGReranker(score_drop_threshold=0.90, min_top_k=1, max_top_k=8),
        )

        conservative_config = conservative._get_pipeline_config()
        permissive_config = permissive._get_pipeline_config()

        assert conservative_config != permissive_config
        assert conservative_config["reranker"]["score_drop_threshold"] == 0.10
        assert permissive_config["reranker"]["score_drop_threshold"] == 0.90

    def test_pipeline_config_records_dynamic_rag_base_reranker_identity(self):
        llm = FakeListLLM(responses=["answer"])
        retrieval_pipeline = create_mock_retrieval_pipeline(pipeline_id=77)
        service = MagicMock()
        base_reranker = FixedAsyncReranker(model_name="fixed-scorer", batch_size=7, results=[])
        pipeline = _build_unit_pipeline(
            llm,
            retrieval_pipeline,
            service,
            reranker=DynamicRAGReranker(
                base_reranker=base_reranker,
                min_top_k=2,
                max_top_k=6,
                score_drop_threshold=0.30,
                min_score=0.40,
            ),
        )

        config = pipeline._get_pipeline_config()

        assert config["reranker"] == {
            "type": "DynamicRAGReranker",
            "model_name": "dynamic-rag",
            "batch_size": 64,
            "max_concurrency": 10,
            "base_reranker": {
                "type": "FixedAsyncReranker",
                "model_name": "fixed-scorer",
                "batch_size": 7,
                "max_concurrency": 10,
            },
            "min_top_k": 2,
            "max_top_k": 6,
            "score_drop_threshold": 0.30,
            "min_score": 0.40,
        }

    @pytest.mark.asyncio
    async def test_generate_retrieves_dynamic_reranks_and_answers(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=_mock_response("final answer"))
        retrieval_pipeline = create_mock_retrieval_pipeline(
            default_results=[
                {"doc_id": 1, "score": 0.7, "content": "first"},
                {"doc_id": 2, "score": 0.6, "content": "second"},
            ]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        reranker = FixedAsyncReranker(results=[RerankResult(index=1, text="second", score=0.95)])
        pipeline = _build_unit_pipeline(
            llm,
            retrieval_pipeline,
            service,
            reranker=reranker,
            candidate_top_k=5,
        )

        result = await pipeline._generate(1, top_k=2)

        retrieval_pipeline._retrieve_by_id.assert_awaited_once_with(1, 5)
        assert "second" in llm.ainvoke.await_args.args[0]
        assert result.text == "final answer"
        metadata = result.metadata
        assert metadata is not None
        assert metadata["candidate_chunk_ids"] == [1, 2]
        assert metadata["selected_chunk_ids"] == [2]
        assert metadata["selected_scores"] == [0.95]
        assert metadata["effective_top_k"] == 1

    @pytest.mark.asyncio
    async def test_generate_backfills_missing_contents(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=_mock_response("answer"))
        retrieval_pipeline = create_mock_retrieval_pipeline(default_results=[{"doc_id": 5, "score": 0.7}])
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        service.get_chunk_contents.return_value = ["Fetched content."]
        reranker = FixedAsyncReranker(results=[RerankResult(index=0, text="Fetched content.", score=0.8)])
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, reranker=reranker, candidate_top_k=1)

        result = await pipeline._generate(1, top_k=1)

        service.get_chunk_contents.assert_called_once_with([5])
        metadata = result.metadata
        assert metadata is not None
        assert metadata["selected_chunk_ids"] == [5]
