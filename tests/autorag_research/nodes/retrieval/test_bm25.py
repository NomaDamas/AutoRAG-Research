import shutil
from pathlib import Path

import pytest

from autorag_research.nodes.retrieval.bm25 import BM25Module

TEST_INDEX_PATH = Path(__file__).parent.parent.parent.parent / "resources" / "bm25_test_index"

pytestmark = pytest.mark.skipif(not shutil.which("java"), reason="Java not available")


@pytest.fixture
def bm25_prebuilt():
    return BM25Module(index_name="beir-v1.0.0-scifact.flat")


@pytest.fixture
def bm25_private():
    return BM25Module(index_path=str(TEST_INDEX_PATH))


def test_bm25_prebuilt_index(bm25_prebuilt):
    results = bm25_prebuilt.run(["scientific evidence"], top_k=3)

    assert len(results) == 1
    assert len(results[0]) == 3
    assert all("doc_id" in r and "score" in r and "content" in r for r in results[0])
    assert results[0][0]["score"] >= results[0][1]["score"] >= results[0][2]["score"]


def test_bm25_prebuilt_multiple_queries(bm25_prebuilt):
    results = bm25_prebuilt.run(["vaccine effectiveness", "gene therapy"], top_k=2)

    assert len(results) == 2
    assert all(len(r) == 2 for r in results)


def test_bm25_prebuilt_empty_queries(bm25_prebuilt):
    results = bm25_prebuilt.run([], top_k=5)

    assert results == []


def test_bm25_private_index(bm25_private):
    results = bm25_private.run(["dog joat jaehwan"], top_k=3)

    assert len(results) == 1
    assert len(results[0]) == 3
    assert all("doc_id" in r and "score" in r and "content" in r for r in results[0])
    assert results[0][0]["score"] >= results[0][1]["score"] >= results[0][2]["score"]


def test_bm25_private_multiple_queries(bm25_private):
    results = bm25_private.run(["Will SSG buy dog joat jaehwan?", "Will KIA buy dog joat jaehwan?"], top_k=2)

    assert len(results) == 2
    assert all(len(r) == 2 for r in results)


def test_bm25_private_empty_queries(bm25_private):
    results = bm25_private.run([], top_k=5)

    # Pyserini returns an empty list for empty queries
    assert results == []
