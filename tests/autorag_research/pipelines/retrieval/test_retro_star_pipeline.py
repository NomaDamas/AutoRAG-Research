"""Tests for the RETRO* retrieval pipeline."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake import FakeListLLM
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.pipelines.retrieval.retro_star import (
    DEFAULT_RETRO_STAR_PROMPT_TEMPLATE,
    DEFAULT_RETRO_STAR_RELEVANCE_DEFINITION,
    RetroStarPipelineConfig,
    RetroStarRetrievalPipeline,
    _integrate_retro_scores,
    _parse_retro_score,
)
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
    create_mock_retrieval_pipeline,
)


def _mock_llm_response(content: str) -> MagicMock:
    response = MagicMock()
    response.content = content
    response.__str__ = lambda _: content
    return response


@pytest.fixture
def cleanup_pipeline_results(session_factory: sessionmaker[Session]):
    """Cleanup fixture that deletes RETRO* pipeline results after test."""
    created_pipeline_ids: list[int] = []

    yield created_pipeline_ids

    session = session_factory()
    try:
        result_repo = ChunkRetrievedResultRepository(session)
        for pipeline_id in created_pipeline_ids:
            result_repo.delete_by_pipeline(pipeline_id)
        session.commit()
    finally:
        session.close()


class TestRetroStarHelpers:
    """Unit tests for RETRO* helper functions."""

    def test_parse_retro_score_extracts_integer(self):
        assert _parse_retro_score("analysis\n<score>84</score>") == 84

    def test_parse_retro_score_uses_final_score_tag(self):
        assert _parse_retro_score("draft <score>10</score> revised <score>90</score>") == 90

    @pytest.mark.parametrize("response_text", ["<score>124</score>", "<score>-8</score>"])
    def test_parse_retro_score_rejects_out_of_range_values(self, response_text: str):
        with pytest.raises(ValueError, match="RETRO\\* score must be an integer between 0 and 100"):
            _parse_retro_score(response_text)

    def test_parse_retro_score_rejects_missing_score_tag(self):
        with pytest.raises(ValueError, match="RETRO\\* response must contain"):
            _parse_retro_score("no score tag present")

    def test_integrate_retro_scores_returns_mean_by_default(self):
        assert _integrate_retro_scores([30, 60, 90]) == pytest.approx(60.0)

    def test_integrate_retro_scores_supports_explicit_weights(self):
        assert _integrate_retro_scores([20, 80], weights=[1.0, 3.0]) == pytest.approx(65.0)

    def test_integrate_retro_scores_rejects_negative_weights(self):
        with pytest.raises(ValueError, match="weights must not contain negative values"):
            _integrate_retro_scores([0, 100], weights=[-1.0, 2.0])


class TestRetroStarPipelineConfig:
    """Tests for RetroStarPipelineConfig."""

    def test_get_pipeline_class(self):
        llm = FakeListLLM(responses=["<score>80</score>"])
        config = RetroStarPipelineConfig(
            name="retro_star",
            llm=llm,
            retrieval_pipeline_name="bm25",
        )

        assert config.get_pipeline_class() == RetroStarRetrievalPipeline

    def test_default_prompt_template_and_relevance_definition(self):
        llm = FakeListLLM(responses=["<score>80</score>"])
        config = RetroStarPipelineConfig(
            name="retro_star",
            llm=llm,
            retrieval_pipeline_name="bm25",
        )

        assert config.prompt_template == DEFAULT_RETRO_STAR_PROMPT_TEMPLATE
        assert config.relevance_definition == DEFAULT_RETRO_STAR_RELEVANCE_DEFINITION

    def test_get_pipeline_kwargs_requires_injected_retrieval_pipeline(self):
        llm = FakeListLLM(responses=["<score>80</score>"])
        config = RetroStarPipelineConfig(
            name="retro_star",
            llm=llm,
            retrieval_pipeline_name="bm25",
        )

        with pytest.raises(ValueError, match="not injected"):
            config.get_pipeline_kwargs()

    def test_get_pipeline_kwargs_after_injection(self):
        llm = FakeListLLM(responses=["<score>80</score>"])
        wrapped_retrieval = create_mock_retrieval_pipeline(pipeline_id=77)
        config = RetroStarPipelineConfig(
            name="retro_star",
            llm=llm,
            retrieval_pipeline_name="bm25",
            candidate_top_k=25,
            num_samples=3,
            sample_weights=[0.2, 0.3, 0.5],
        )

        config.inject_retrieval_pipeline(wrapped_retrieval)
        kwargs = config.get_pipeline_kwargs()

        assert kwargs["llm"] is llm
        assert kwargs["retrieval_pipeline"] is wrapped_retrieval
        assert kwargs["candidate_top_k"] == 25
        assert kwargs["num_samples"] == 3
        assert kwargs["sample_weights"] == [0.2, 0.3, 0.5]

    @pytest.mark.api
    def test_string_llm_conversion(self):
        with patch("autorag_research.injection.load_llm") as mock_load:
            mock_llm = MagicMock()
            mock_load.return_value = mock_llm

            config = RetroStarPipelineConfig(
                name="retro_star",
                llm="mock",
                retrieval_pipeline_name="bm25",
            )

            mock_load.assert_called_once_with("mock")
            assert config.llm is mock_llm


class TestRetroStarRetrievalPipeline:
    """Tests for RetroStarRetrievalPipeline."""

    def test_creation_rejects_missing_query_or_doc_placeholder(self, session_factory):
        with pytest.raises(ValueError, match="must contain"):
            RetroStarRetrievalPipeline(
                session_factory=session_factory,
                name="retro_star_invalid_prompt",
                llm=FakeListLLM(responses=["<score>80</score>"]),
                retrieval_pipeline=create_mock_retrieval_pipeline(),
                prompt_template="Missing placeholders altogether",
            )

    def test_creation_rejects_negative_sample_weights(self, session_factory):
        with (
            patch("autorag_research.pipelines.retrieval.base.BaseRetrievalPipeline.__init__", return_value=None),
            pytest.raises(ValueError, match="sample_weights must not contain negative values"),
        ):
            RetroStarRetrievalPipeline(
                session_factory=session_factory,
                name="retro_star_invalid_weights",
                llm=FakeListLLM(responses=["<score>80</score>"]),
                retrieval_pipeline=create_mock_retrieval_pipeline(),
                num_samples=2,
                sample_weights=[-0.1, 1.1],
            )

    def test_pipeline_config(self, session_factory, cleanup_pipeline_results: list[int]):
        pipeline = RetroStarRetrievalPipeline(
            session_factory=session_factory,
            name="retro_star_config",
            llm=FakeListLLM(responses=["<score>80</score>"]),
            retrieval_pipeline=create_mock_retrieval_pipeline(pipeline_id=123),
            candidate_top_k=50,
            num_samples=4,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["type"] == "retro_star"
        assert config["candidate_top_k"] == 50
        assert config["num_samples"] == 4
        assert config["retrieval_pipeline_id"] == 123

    @pytest.mark.asyncio
    async def test_retrieve_by_text_reranks_wrapped_candidates(
        self, session_factory, cleanup_pipeline_results: list[int]
    ):
        wrapped_retrieval = create_mock_retrieval_pipeline(
            default_results=[
                {"doc_id": 1, "score": 0.99, "content": "first document"},
                {"doc_id": 2, "score": 0.40, "content": "second document"},
            ]
        )
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_llm_response("analysis\n<score>35</score>"),
                _mock_llm_response("analysis\n<score>92</score>"),
            ]
        )

        pipeline = RetroStarRetrievalPipeline(
            session_factory=session_factory,
            name="retro_star_by_text",
            llm=llm,
            retrieval_pipeline=wrapped_retrieval,
            candidate_top_k=2,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        results = await pipeline._retrieve_by_text("original query", top_k=1)

        wrapped_retrieval.retrieve.assert_awaited_once_with("original query", 2)
        assert results == [{"doc_id": 2, "score": 92.0, "content": "second document"}]

    @pytest.mark.asyncio
    async def test_retrieve_by_text_backfills_missing_candidate_content(
        self,
        session_factory,
        cleanup_pipeline_results: list[int],
    ):
        wrapped_retrieval = create_mock_retrieval_pipeline(
            default_results=[
                {"doc_id": 1, "score": 0.9},
                {"doc_id": 2, "score": 0.8},
            ]
        )
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_llm_response("reasoning\n<score>55</score>"),
                _mock_llm_response("reasoning\n<score>75</score>"),
            ]
        )

        session = session_factory()
        try:
            chunk_repo = ChunkRepository(session)
            chunk_contents = [chunk.contents for chunk in chunk_repo.get_by_ids([1, 2])]
        finally:
            session.close()

        pipeline = RetroStarRetrievalPipeline(
            session_factory=session_factory,
            name="retro_star_backfill_content",
            llm=llm,
            retrieval_pipeline=wrapped_retrieval,
            candidate_top_k=2,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        await pipeline._retrieve_by_text("query text", top_k=2)

        prompts = [call.args[0] for call in llm.ainvoke.await_args_list]
        assert any(chunk_contents[0] in prompt for prompt in prompts)
        assert any(chunk_contents[1] in prompt for prompt in prompts)

    @pytest.mark.asyncio
    async def test_retrieve_by_text_uses_multi_sample_score_integration(
        self,
        session_factory,
        cleanup_pipeline_results: list[int],
    ):
        wrapped_retrieval = create_mock_retrieval_pipeline(
            default_results=[{"doc_id": 1, "score": 0.9, "content": "doc"}]
        )
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_llm_response("first\n<score>20</score>"),
                _mock_llm_response("second\n<score>80</score>"),
            ]
        )

        pipeline = RetroStarRetrievalPipeline(
            session_factory=session_factory,
            name="retro_star_multi_sample",
            llm=llm,
            retrieval_pipeline=wrapped_retrieval,
            candidate_top_k=1,
            num_samples=2,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        results = await pipeline._retrieve_by_text("query text", top_k=1)

        assert results[0]["score"] == pytest.approx(50.0)
        assert llm.ainvoke.await_count == 2

    @pytest.mark.asyncio
    async def test_retrieve_by_text_raises_on_malformed_score_output(
        self,
        session_factory,
    ):
        wrapped_retrieval = create_mock_retrieval_pipeline(
            default_results=[{"doc_id": 1, "score": 0.9, "content": "doc"}]
        )
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=[_mock_llm_response("reasoning without score tags")])

        with patch("autorag_research.pipelines.retrieval.base.BaseRetrievalPipeline.__init__", return_value=None):
            pipeline = RetroStarRetrievalPipeline(
                session_factory=session_factory,
                name="retro_star_invalid_score_output",
                llm=llm,
                retrieval_pipeline=wrapped_retrieval,
            )

        with pytest.raises(ValueError, match="RETRO\\* response must contain"):
            await pipeline._retrieve_by_text("query text", top_k=1)

    @pytest.mark.asyncio
    async def test_retrieve_by_text_raises_on_out_of_range_score_output(
        self,
        session_factory,
    ):
        wrapped_retrieval = create_mock_retrieval_pipeline(
            default_results=[{"doc_id": 1, "score": 0.9, "content": "doc"}]
        )
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=[_mock_llm_response("analysis\n<score>124</score>")])

        with patch("autorag_research.pipelines.retrieval.base.BaseRetrievalPipeline.__init__", return_value=None):
            pipeline = RetroStarRetrievalPipeline(
                session_factory=session_factory,
                name="retro_star_out_of_range_score_output",
                llm=llm,
                retrieval_pipeline=wrapped_retrieval,
            )

        with pytest.raises(ValueError, match="RETRO\\* score must be an integer between 0 and 100"):
            await pipeline._retrieve_by_text("query text", top_k=1)

    @pytest.mark.asyncio
    async def test_retrieve_by_id_fetches_query_text_then_reranks(
        self,
        session_factory,
        cleanup_pipeline_results: list[int],
    ):
        wrapped_retrieval = create_mock_retrieval_pipeline(
            default_results=[{"doc_id": 3, "score": 0.88, "content": "doc"}]
        )
        pipeline = RetroStarRetrievalPipeline(
            session_factory=session_factory,
            name="retro_star_by_id",
            llm=FakeListLLM(responses=["analysis\n<score>88</score>"]),
            retrieval_pipeline=wrapped_retrieval,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        results = await pipeline._retrieve_by_id(1, top_k=2)

        wrapped_retrieval.retrieve.assert_awaited_once()
        assert results == [{"doc_id": 3, "score": 88.0, "content": "doc"}]

    def test_run_full_pipeline(self, session_factory, cleanup_pipeline_results: list[int]):
        session = session_factory()
        try:
            query_count = QueryRepository(session).count()
        finally:
            session.close()

        wrapped_retrieval = create_mock_retrieval_pipeline(
            default_results=[
                {"doc_id": 1, "score": 0.91, "content": "Content 1"},
                {"doc_id": 2, "score": 0.83, "content": "Content 2"},
            ]
        )
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[_mock_llm_response("<score>80</score>"), _mock_llm_response("<score>60</score>")]
            * max(query_count, 1)
        )

        pipeline = RetroStarRetrievalPipeline(
            session_factory=session_factory,
            name="retro_star_full_run",
            llm=llm,
            retrieval_pipeline=wrapped_retrieval,
            candidate_top_k=2,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline.run(top_k=2)

        verifier = PipelineTestVerifier(
            result,
            pipeline.pipeline_id,
            session_factory,
            PipelineTestConfig(
                pipeline_type="retrieval",
                expected_total_queries=query_count,
                expected_min_results=0,
                check_persistence=True,
            ),
        )
        verifier.verify_all()
