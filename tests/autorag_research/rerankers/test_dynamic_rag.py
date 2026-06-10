"""Tests for DynamicRAG reranker."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from autorag_research.rerankers.dynamic_rag import DynamicRAGReranker


def _llm_response(content: str) -> MagicMock:
    response = MagicMock()
    response.content = content
    return response


def test_dynamic_rag_reranker_parses_ordered_subset_with_synthetic_scores():
    llm = MagicMock()
    llm.invoke.return_value = _llm_response("3, 1")
    reranker = DynamicRAGReranker(llm=llm)

    results = reranker.rerank("query", ["a", "b", "c"], top_k=3)

    assert [(result.index, result.text, result.score) for result in results] == [(2, "c", 2.0), (0, "a", 1.0)]
    prompt = llm.invoke.call_args.args[0]
    assert "Query:\nquery" in prompt
    assert "[1] a" in prompt
    assert "[3] c" in prompt


def test_dynamic_rag_reranker_none_allows_zero_documents():
    llm = MagicMock()
    llm.invoke.return_value = _llm_response("None")
    reranker = DynamicRAGReranker(llm=llm)

    assert reranker.rerank("query", ["a", "b"], top_k=2) == []


def test_dynamic_rag_reranker_none_prefixed_response_means_zero_documents_despite_digits():
    llm = MagicMock()
    llm.invoke.return_value = _llm_response("None of the documents 1, 2 are required.")
    reranker = DynamicRAGReranker(llm=llm)

    assert reranker.rerank("query", ["a", "b"], top_k=2) == []


def test_dynamic_rag_reranker_ignores_duplicates_and_out_of_range_ids():
    llm = MagicMock()
    llm.invoke.return_value = _llm_response("[Doc 2]\nDoc 99, 2, document 1")
    reranker = DynamicRAGReranker(llm=llm)

    results = reranker.rerank("query", ["a", "b", "c"], top_k=3)

    assert [result.index for result in results] == [1, 0]
    assert [result.score for result in results] == [2.0, 1.0]


def test_dynamic_rag_reranker_top_k_is_safety_cap():
    llm = MagicMock()
    llm.invoke.return_value = _llm_response("3, 1, 2")
    reranker = DynamicRAGReranker(llm=llm)

    results = reranker.rerank("query", ["a", "b", "c"], top_k=2)

    assert [result.index for result in results] == [2, 0]
    assert [result.score for result in results] == [2.0, 1.0]


def test_dynamic_rag_reranker_requires_llm_at_rerank_time():
    reranker = DynamicRAGReranker()

    with pytest.raises(ValueError, match="requires llm"):
        reranker.rerank("query", ["a"], top_k=1)


def test_dynamic_rag_reranker_unparseable_retries_once_then_recovers(caplog: pytest.LogCaptureFixture):
    llm = MagicMock()
    llm.invoke.side_effect = [_llm_response("select the helpful evidence"), _llm_response("2, 1")]
    reranker = DynamicRAGReranker(llm=llm)

    with caplog.at_level("WARNING", logger="AutoRAG-Research"):
        results = reranker.rerank("query", ["a", "b", "c"], top_k=2)

    assert llm.invoke.call_count == 2
    assert [result.index for result in results] == [1, 0]
    assert "retrying once" in caplog.text


def test_dynamic_rag_reranker_unparseable_twice_raises():
    llm = MagicMock()
    llm.invoke.side_effect = [
        _llm_response("select the helpful evidence"),
        _llm_response("still no identifiers here"),
    ]
    reranker = DynamicRAGReranker(llm=llm)

    with pytest.raises(ValueError, match="not an ordered document-ID list or None"):
        reranker.rerank("query", ["a", "b", "c"], top_k=2)


@pytest.mark.asyncio
async def test_dynamic_rag_reranker_async_uses_ainvoke():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=_llm_response("2, 1"))
    reranker = DynamicRAGReranker(llm=llm)

    results = await reranker.arerank("query", ["a", "b"], top_k=2)

    llm.ainvoke.assert_awaited_once()
    assert [result.index for result in results] == [1, 0]


def test_dynamic_rag_default_config_loads_llm_reranker_without_removed_fields():
    config = yaml.safe_load(Path("configs/reranker/dynamic_rag.yaml").read_text())

    assert "base_reranker" not in config
    assert "min_top_k" not in config
    assert "max_top_k" not in config
    assert "score_drop_threshold" not in config
    assert "min_score" not in config
    reranker = DynamicRAGReranker(**{key: value for key, value in config.items() if key != "_target_"})
    assert reranker.model_name == "dynamic-rag"
    assert reranker.llm is None
