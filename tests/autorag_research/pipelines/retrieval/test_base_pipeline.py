"""Non-DB tests for base retrieval pipeline unit behavior."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline, get_retrieval_pipeline_unit


class _RunDispatchPipeline(BaseRetrievalPipeline):
    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        return []

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        return []

    def _get_pipeline_config(self) -> dict[str, Any]:
        return {"type": "run_dispatch", "retrieval_unit": self.retrieval_unit}


def _pipeline_with_service(retrieval_unit: object) -> _RunDispatchPipeline:
    pipeline = _RunDispatchPipeline.__new__(_RunDispatchPipeline)
    pipeline.retrieval_unit = retrieval_unit
    pipeline.pipeline_id = 42
    pipeline._service = MagicMock()
    pipeline._service.run_pipeline.return_value = {"pipeline_id": 42, "total_queries": 1, "total_results": 1}
    pipeline._service.run_image_pipeline.return_value = {"pipeline_id": 42, "total_queries": 1, "total_results": 1}
    return pipeline


def test_run_dispatches_image_unit_to_image_persistence():
    pipeline = _pipeline_with_service("image_chunk")

    result = pipeline.run(top_k=3, query_limit=2)

    assert result == {"pipeline_id": 42, "total_queries": 1, "total_results": 1}
    pipeline._service.run_image_pipeline.assert_called_once_with(
        retrieval_func=pipeline._retrieve_by_id,
        pipeline_id=42,
        top_k=3,
        batch_size=128,
        max_concurrency=16,
        max_retries=3,
        retry_delay=1.0,
        query_limit=2,
    )
    pipeline._service.run_pipeline.assert_not_called()


def test_run_rejects_mixed_unit_until_mixed_persistence_exists():
    pipeline = _pipeline_with_service("mixed")

    with pytest.raises(ValueError, match="Mixed retrieval_unit persistence is not supported"):
        pipeline.run()

    pipeline._service.run_pipeline.assert_not_called()
    pipeline._service.run_image_pipeline.assert_not_called()


def test_get_retrieval_pipeline_unit_rejects_unknown_typed_value():
    pipeline = MagicMock()
    pipeline.retrieval_unit = "chunks"
    pipeline._get_pipeline_config.return_value = {"retrieval_unit": "chunk"}

    with pytest.raises(ValueError, match="Invalid retrieval_unit 'chunks'"):
        get_retrieval_pipeline_unit(pipeline)


def test_run_rejects_invalid_explicit_unit_before_persistence_dispatch():
    pipeline = _pipeline_with_service("image_chunks")

    with pytest.raises(ValueError, match="Invalid retrieval_unit 'image_chunks'"):
        pipeline.run()

    pipeline._service.run_pipeline.assert_not_called()
    pipeline._service.run_image_pipeline.assert_not_called()


@pytest.mark.parametrize("retrieval_unit", [False, 123, ["image_chunk"], {"unit": "image_chunk"}])
def test_run_rejects_malformed_typed_unit_before_config_fallback(retrieval_unit: object):
    pipeline = _pipeline_with_service(retrieval_unit)
    pipeline._get_pipeline_config = MagicMock(return_value={"type": "run_dispatch", "retrieval_unit": "image_chunk"})

    with pytest.raises(ValueError, match="Invalid retrieval_unit"):
        pipeline.run()

    pipeline._service.run_pipeline.assert_not_called()
    pipeline._service.run_image_pipeline.assert_not_called()
