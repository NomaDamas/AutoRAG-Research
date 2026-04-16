"""Tests for PowerOfNoiseRetrievalPipeline."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.schema import Chunk
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.pipelines.retrieval.power_of_noise import (
    PowerOfNoiseRetrievalPipeline,
    PowerOfNoiseRetrievalPipelineConfig,
)
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
)


class _FakeLeafRetrievalPipeline(BaseRetrievalPipeline):
    def _get_pipeline_config(self) -> dict[str, Any]:
        return {"type": "fake_leaf"}

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        return []

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        return []


class _FakeWrapperRetrievalPipeline(BaseRetrievalPipeline):
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        inner_retrieval_pipeline: BaseRetrievalPipeline,
        schema: Any | None = None,
    ):
        self.inner_retrieval_pipeline = inner_retrieval_pipeline
        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        return {
            "type": "fake_wrapper",
            "inner_pipeline_name": self.inner_retrieval_pipeline.name,
        }

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        return []

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        return []


@dataclass(kw_only=True)
class _FakeLeafPipelineConfig(BaseRetrievalPipelineConfig):
    def get_pipeline_class(self) -> type[BaseRetrievalPipeline]:
        return _FakeLeafRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        return {}


@dataclass(kw_only=True)
class _FakeWrapperPipelineConfig(BaseRetrievalPipelineConfig):
    inner_retrieval_pipeline_name: str
    _inner_retrieval_pipeline: BaseRetrievalPipeline | None = None

    def get_pipeline_class(self) -> type[BaseRetrievalPipeline]:
        return _FakeWrapperRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        if self._inner_retrieval_pipeline is None:
            msg = f"Inner retrieval pipeline '{self.inner_retrieval_pipeline_name}' not injected"
            raise ValueError(msg)

        return {"inner_retrieval_pipeline": self._inner_retrieval_pipeline}

    def inject_retrieval_pipeline(self, pipeline: BaseRetrievalPipeline) -> None:
        self._inner_retrieval_pipeline = pipeline


@pytest.fixture
def cleanup_pipeline_results(session_factory: sessionmaker[Session]):
    """Delete created retrieval results and temporary chunks after each test."""
    created_pipeline_ids: list[int] = []
    created_chunk_ids: list[int] = []

    yield created_pipeline_ids, created_chunk_ids

    session = session_factory()
    try:
        result_repo = ChunkRetrievedResultRepository(session)
        for pipeline_id in created_pipeline_ids:
            result_repo.delete_by_pipeline(pipeline_id)
        if created_chunk_ids:
            session.execute(delete(Chunk).where(Chunk.id.in_(created_chunk_ids)))
        session.commit()
    finally:
        session.close()


@pytest.fixture
def base_pipeline_stub() -> SimpleNamespace:
    """Provide a stub retrieval pipeline for wrapper tests."""
    return SimpleNamespace(
        name="vector_search",
        _retrieve_by_id=AsyncMock(
            return_value=[
                {"doc_id": 2, "score": 0.9, "content": "Chunk 1-2"},
                {"doc_id": 3, "score": 0.8, "content": "Chunk 2-1"},
            ]
        ),
        _retrieve_by_text=AsyncMock(
            return_value=[
                {"doc_id": 2, "score": 0.9, "content": "Chunk 1-2"},
                {"doc_id": 3, "score": 0.8, "content": "Chunk 2-1"},
            ]
        ),
    )


class TestPowerOfNoisePipelineConfig:
    """Tests for PowerOfNoiseRetrievalPipelineConfig."""

    def test_config_get_pipeline_class(self):
        config = PowerOfNoiseRetrievalPipelineConfig(
            name="power_of_noise",
            base_retrieval_pipeline_name="vector_search",
        )

        assert config.get_pipeline_class() == PowerOfNoiseRetrievalPipeline

    def test_config_get_pipeline_kwargs(self):
        config = PowerOfNoiseRetrievalPipelineConfig(
            name="power_of_noise",
            base_retrieval_pipeline_name="vector_search",
            noise_count=2,
            noise_order="interleave",
            noise_mode="answer_aware_random",
            seed=13,
        )

        assert config.get_pipeline_kwargs() == {
            "base_retrieval_pipeline": "vector_search",
            "noise_count": 2,
            "noise_ratio": None,
            "noise_order": "interleave",
            "noise_mode": "answer_aware_random",
            "seed": 13,
        }


class TestPowerOfNoiseRetrievalPipeline:
    """Tests for PowerOfNoiseRetrievalPipeline."""

    def test_named_base_pipeline_loads_nested_retrieval_dependencies(
        self,
        monkeypatch: pytest.MonkeyPatch,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: tuple[list[int], list[int]],
    ):
        created_pipeline_ids, _ = cleanup_pipeline_results
        config_map = {
            "fake_wrapper_cfg": _FakeWrapperPipelineConfig(
                name="fake_wrapper",
                inner_retrieval_pipeline_name="fake_leaf",
            ),
            "fake_leaf_cfg": _FakeLeafPipelineConfig(name="fake_leaf"),
        }

        def fake_resolve_config(_self, config_groups: list[str], name: str) -> str:
            assert config_groups == ["pipelines", "retrieval"]
            return f"{name}_cfg"

        monkeypatch.setattr(
            "autorag_research.cli.config_resolver.ConfigResolver.resolve_config",
            fake_resolve_config,
        )
        monkeypatch.setattr(
            "autorag_research.pipelines.retrieval.loader.instantiate",
            lambda pipeline_cfg: config_map[pipeline_cfg],
        )

        pipeline = PowerOfNoiseRetrievalPipeline(
            session_factory=session_factory,
            name="test_power_of_noise_named_dependency_loader",
            base_retrieval_pipeline="fake_wrapper",
        )
        created_pipeline_ids.extend([
            pipeline.pipeline_id,
            pipeline._base_retrieval_pipeline.pipeline_id,
            pipeline._base_retrieval_pipeline.inner_retrieval_pipeline.pipeline_id,
        ])

        assert pipeline._base_retrieval_pipeline.name == "fake_wrapper"
        assert pipeline._base_retrieval_pipeline.inner_retrieval_pipeline.name == "fake_leaf"

    @pytest.mark.asyncio
    async def test_retrieved_first_appends_deterministic_noise(
        self,
        session_factory: sessionmaker[Session],
        base_pipeline_stub: SimpleNamespace,
        cleanup_pipeline_results: tuple[list[int], list[int]],
    ):
        created_pipeline_ids, _ = cleanup_pipeline_results
        pipeline = PowerOfNoiseRetrievalPipeline(
            session_factory=session_factory,
            name="test_power_of_noise_retrieved_first",
            base_retrieval_pipeline=base_pipeline_stub,
            noise_count=2,
            noise_order="retrieved_first",
            seed=7,
        )
        created_pipeline_ids.append(pipeline.pipeline_id)

        results_a = await pipeline._retrieve_by_text("ad hoc query", top_k=4)
        results_b = await pipeline._retrieve_by_text("ad hoc query", top_k=4)

        assert [result["doc_id"] for result in results_a[:2]] == [2, 3]
        assert results_a == results_b
        assert all(result["doc_id"] not in {2, 3} for result in results_a[2:])

    @pytest.mark.asyncio
    async def test_noise_first_places_noise_before_retrieved_docs(
        self,
        session_factory: sessionmaker[Session],
        base_pipeline_stub: SimpleNamespace,
        cleanup_pipeline_results: tuple[list[int], list[int]],
    ):
        created_pipeline_ids, _ = cleanup_pipeline_results
        pipeline = PowerOfNoiseRetrievalPipeline(
            session_factory=session_factory,
            name="test_power_of_noise_noise_first",
            base_retrieval_pipeline=base_pipeline_stub,
            noise_count=2,
            noise_order="noise_first",
            seed=7,
        )
        created_pipeline_ids.append(pipeline.pipeline_id)

        results = await pipeline._retrieve_by_text("ad hoc query", top_k=4)

        assert [result["doc_id"] for result in results[-2:]] == [2, 3]
        assert all(result["doc_id"] not in {2, 3} for result in results[:2])

    @pytest.mark.asyncio
    async def test_interleave_alternates_retrieved_and_noise(
        self,
        session_factory: sessionmaker[Session],
        base_pipeline_stub: SimpleNamespace,
        cleanup_pipeline_results: tuple[list[int], list[int]],
    ):
        created_pipeline_ids, _ = cleanup_pipeline_results
        pipeline = PowerOfNoiseRetrievalPipeline(
            session_factory=session_factory,
            name="test_power_of_noise_interleave",
            base_retrieval_pipeline=base_pipeline_stub,
            noise_count=2,
            noise_order="interleave",
            seed=7,
        )
        created_pipeline_ids.append(pipeline.pipeline_id)

        results = await pipeline._retrieve_by_text("ad hoc query", top_k=4)

        assert results[0]["doc_id"] == 2
        assert results[2]["doc_id"] == 3
        assert results[1]["doc_id"] not in {2, 3}
        assert results[3]["doc_id"] not in {2, 3}

    def test_pipeline_config_includes_noise_settings(
        self,
        session_factory: sessionmaker[Session],
        base_pipeline_stub: SimpleNamespace,
        cleanup_pipeline_results: tuple[list[int], list[int]],
    ):
        created_pipeline_ids, _ = cleanup_pipeline_results
        pipeline = PowerOfNoiseRetrievalPipeline(
            session_factory=session_factory,
            name="test_power_of_noise_config",
            base_retrieval_pipeline=base_pipeline_stub,
            noise_count=2,
            noise_ratio=0.5,
            noise_order="interleave",
            noise_mode="answer_aware_random",
            seed=11,
        )
        created_pipeline_ids.append(pipeline.pipeline_id)

        assert pipeline._get_pipeline_config() == {
            "type": "power_of_noise",
            "base_retrieval_pipeline": "vector_search",
            "noise_count": 2,
            "noise_ratio": 0.5,
            "noise_order": "interleave",
            "noise_mode": "answer_aware_random",
            "seed": 11,
        }

    def test_answer_aware_candidates_exclude_ground_truth_and_answer_chunks(
        self,
        session_factory: sessionmaker[Session],
        base_pipeline_stub: SimpleNamespace,
        cleanup_pipeline_results: tuple[list[int], list[int]],
    ):
        created_pipeline_ids, created_chunk_ids = cleanup_pipeline_results
        session = session_factory()
        try:
            answer_chunk = Chunk(contents="alpha appears in this distractor")
            safe_chunk = Chunk(contents="totally unrelated evidence")
            session.add_all([answer_chunk, safe_chunk])
            session.commit()
            created_chunk_ids.extend([answer_chunk.id, safe_chunk.id])
        finally:
            session.close()

        pipeline = PowerOfNoiseRetrievalPipeline(
            session_factory=session_factory,
            name="test_power_of_noise_answer_aware",
            base_retrieval_pipeline=base_pipeline_stub,
            noise_count=2,
            noise_mode="answer_aware_random",
            seed=5,
        )
        created_pipeline_ids.append(pipeline.pipeline_id)

        candidate_ids = pipeline._get_noise_candidate_ids(query_id=1, excluded_doc_ids={2})

        assert 1 not in candidate_ids  # retrieval GT chunk for query 1
        assert answer_chunk.id not in candidate_ids
        assert safe_chunk.id in candidate_ids
        assert 2 not in candidate_ids

    @pytest.mark.asyncio
    async def test_noise_count_is_fixed_even_when_base_underfills(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: tuple[list[int], list[int]],
    ):
        created_pipeline_ids, _ = cleanup_pipeline_results
        underfilled_base = SimpleNamespace(
            name="vector_search",
            _retrieve_by_id=AsyncMock(return_value=[{"doc_id": 2, "score": 0.9, "content": "Chunk 1-2"}]),
            _retrieve_by_text=AsyncMock(return_value=[{"doc_id": 2, "score": 0.9, "content": "Chunk 1-2"}]),
        )
        pipeline = PowerOfNoiseRetrievalPipeline(
            session_factory=session_factory,
            name="test_power_of_noise_fixed_budget",
            base_retrieval_pipeline=underfilled_base,
            noise_count=1,
            seed=17,
        )
        created_pipeline_ids.append(pipeline.pipeline_id)

        results = await pipeline._retrieve_by_text("ad hoc query", top_k=4)

        assert len(results) == 2
        assert [result["doc_id"] for result in results] == [2, 7]

    @pytest.mark.asyncio
    async def test_noise_ratio_controls_noise_budget(
        self,
        session_factory: sessionmaker[Session],
        base_pipeline_stub: SimpleNamespace,
        cleanup_pipeline_results: tuple[list[int], list[int]],
    ):
        created_pipeline_ids, _ = cleanup_pipeline_results
        pipeline = PowerOfNoiseRetrievalPipeline(
            session_factory=session_factory,
            name="test_power_of_noise_ratio_budget",
            base_retrieval_pipeline=base_pipeline_stub,
            noise_ratio=0.5,
            seed=19,
        )
        created_pipeline_ids.append(pipeline.pipeline_id)

        results = await pipeline._retrieve_by_text("ad hoc query", top_k=4)

        assert len(results) == 4
        assert [result["doc_id"] for result in results[:2]] == [2, 3]
        assert all(result["doc_id"] not in {2, 3} for result in results[2:])

    def test_run_persists_results(
        self,
        session_factory: sessionmaker[Session],
        base_pipeline_stub: SimpleNamespace,
        cleanup_pipeline_results: tuple[list[int], list[int]],
    ):
        from autorag_research.orm.repository.query import QueryRepository

        created_pipeline_ids, _ = cleanup_pipeline_results

        session = session_factory()
        try:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()
        finally:
            session.close()

        pipeline = PowerOfNoiseRetrievalPipeline(
            session_factory=session_factory,
            name="test_power_of_noise_run",
            base_retrieval_pipeline=base_pipeline_stub,
            noise_count=1,
            seed=3,
        )
        created_pipeline_ids.append(pipeline.pipeline_id)

        result = pipeline.run(top_k=3)

        verifier = PipelineTestVerifier(
            result,
            pipeline.pipeline_id,
            session_factory,
            PipelineTestConfig(
                pipeline_type="retrieval",
                expected_total_queries=query_count,
                expected_min_results=query_count * 3,
                check_persistence=True,
            ),
        )
        verifier.verify_all()
