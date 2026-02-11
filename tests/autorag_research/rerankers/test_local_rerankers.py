"""Tests for local/GPU-based reranker implementations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autorag_research.rerankers.base import BaseReranker, RerankResult
from autorag_research.rerankers.local_base import LocalReranker

# ===== LocalReranker Base Class Tests =====


class TestLocalRerankerImport:
    """Tests for LocalReranker import and hierarchy."""

    def test_import_from_local_base(self) -> None:
        """LocalReranker can be imported from local_base module."""
        from autorag_research.rerankers.local_base import LocalReranker

        assert LocalReranker is not None

    def test_import_from_init(self) -> None:
        """LocalReranker can be imported from rerankers package."""
        from autorag_research.rerankers import LocalReranker

        assert LocalReranker is not None

    def test_inherits_from_base_reranker(self) -> None:
        """LocalReranker inherits from BaseReranker."""
        assert issubclass(LocalReranker, BaseReranker)

    def test_default_device_is_none(self) -> None:
        """Default device field is None (auto-detect)."""
        assert LocalReranker.model_fields["device"].default is None

    def test_default_max_length(self) -> None:
        """Default max_length is 512."""
        assert LocalReranker.model_fields["max_length"].default == 512


class _MockLocalReranker(LocalReranker):
    """Concrete LocalReranker subclass for testing."""

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        return []

    def model_post_init(self, __context) -> None:
        pass


class TestLocalRerankerInitDevice:
    """Tests for LocalReranker._init_device method."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_init_device_auto_cpu(self, mock_cuda: MagicMock) -> None:
        """Auto-detects CPU when CUDA is not available."""
        reranker = _MockLocalReranker()
        result = reranker._init_device()
        assert result == "cpu"
        assert reranker._device == "cpu"

    @patch("torch.cuda.is_available", return_value=True)
    def test_init_device_auto_cuda(self, mock_cuda: MagicMock) -> None:
        """Auto-detects CUDA when available."""
        reranker = _MockLocalReranker()
        result = reranker._init_device()
        assert result == "cuda"
        assert reranker._device == "cuda"

    def test_init_device_explicit(self) -> None:
        """Uses explicit device when provided."""
        reranker = _MockLocalReranker(device="mps")
        result = reranker._init_device()
        assert result == "mps"
        assert reranker._device == "mps"


class TestLocalRerankerHierarchy:
    """Tests for local reranker inheritance hierarchy."""

    def test_colbert_is_local_reranker(self) -> None:
        """ColBERTReranker inherits from LocalReranker."""
        from autorag_research.rerankers.colbert import ColBERTReranker

        assert issubclass(ColBERTReranker, LocalReranker)

    def test_flag_embedding_is_local_reranker(self) -> None:
        """FlagEmbeddingReranker inherits from LocalReranker."""
        from autorag_research.rerankers.flag_embedding import FlagEmbeddingReranker

        assert issubclass(FlagEmbeddingReranker, LocalReranker)

    def test_flag_embedding_llm_is_local_reranker(self) -> None:
        """FlagEmbeddingLLMReranker inherits from LocalReranker."""
        from autorag_research.rerankers.flag_embedding_llm import FlagEmbeddingLLMReranker

        assert issubclass(FlagEmbeddingLLMReranker, LocalReranker)

    def test_flashrank_is_local_reranker(self) -> None:
        """FlashRankReranker inherits from LocalReranker."""
        from autorag_research.rerankers.flashrank import FlashRankReranker

        assert issubclass(FlashRankReranker, LocalReranker)

    def test_koreranker_is_local_reranker(self) -> None:
        """KoRerankerReranker inherits from LocalReranker."""
        from autorag_research.rerankers.koreranker import KoRerankerReranker

        assert issubclass(KoRerankerReranker, LocalReranker)

    def test_monot5_is_local_reranker(self) -> None:
        """MonoT5Reranker inherits from LocalReranker."""
        from autorag_research.rerankers.monot5 import MonoT5Reranker

        assert issubclass(MonoT5Reranker, LocalReranker)

    def test_openvino_is_local_reranker(self) -> None:
        """OpenVINOReranker inherits from LocalReranker."""
        from autorag_research.rerankers.openvino import OpenVINOReranker

        assert issubclass(OpenVINOReranker, LocalReranker)

    def test_sentence_transformer_is_local_reranker(self) -> None:
        """SentenceTransformerReranker inherits from LocalReranker."""
        from autorag_research.rerankers.sentence_transformer import SentenceTransformerReranker

        assert issubclass(SentenceTransformerReranker, LocalReranker)

    def test_tart_is_local_reranker(self) -> None:
        """TARTReranker inherits from LocalReranker."""
        from autorag_research.rerankers.tart import TARTReranker

        assert issubclass(TARTReranker, LocalReranker)


# ===== SentenceTransformerReranker Tests =====


class TestSentenceTransformerRerankerInit:
    """Tests for SentenceTransformerReranker initialization."""

    def test_import_error_when_missing_dependency(self) -> None:
        """Raises ImportError when sentence-transformers is not installed."""
        with (
            patch.dict("sys.modules", {"sentence_transformers": None}),
            pytest.raises(ImportError, match="sentence-transformers"),
        ):
            from autorag_research.rerankers.sentence_transformer import SentenceTransformerReranker

            SentenceTransformerReranker()

    @patch("autorag_research.rerankers.sentence_transformer.CrossEncoder", create=True)
    def test_default_model_name(self, mock_ce: MagicMock) -> None:
        """Default model_name is cross-encoder/ms-marco-MiniLM-L-2-v2."""
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            from autorag_research.rerankers.sentence_transformer import SentenceTransformerReranker

            reranker = SentenceTransformerReranker.__new__(SentenceTransformerReranker)
            object.__setattr__(reranker, "__dict__", {})
            object.__setattr__(reranker, "__pydantic_fields_set__", set())
            assert (
                SentenceTransformerReranker.model_fields["model_name"].default == "cross-encoder/ms-marco-MiniLM-L-2-v2"
            )


class TestSentenceTransformerRerankerRerank:
    """Tests for SentenceTransformerReranker.rerank method."""

    def _create_reranker(self, mock_scores: list[float]) -> object:
        """Create a SentenceTransformerReranker with mocked model."""
        from autorag_research.rerankers.sentence_transformer import SentenceTransformerReranker

        reranker = SentenceTransformerReranker.__new__(SentenceTransformerReranker)
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_scores
        object.__setattr__(reranker, "_model", mock_model)
        object.__setattr__(reranker, "model_name", "test-model")
        object.__setattr__(reranker, "batch_size", 64)
        object.__setattr__(reranker, "max_concurrency", 10)
        object.__setattr__(reranker, "max_length", 512)
        object.__setattr__(reranker, "device", None)
        return reranker

    def test_empty_documents(self) -> None:
        """Returns empty list for empty documents."""
        reranker = self._create_reranker([])
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_returns_correct_format(self) -> None:
        """Returns list of RerankResult sorted by score descending."""
        reranker = self._create_reranker([0.1, 0.9, 0.5])
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"])

        assert len(results) == 3
        assert all(isinstance(r, RerankResult) for r in results)
        assert results[0].score >= results[1].score >= results[2].score
        assert results[0].index == 1  # doc2 with score 0.9

    def test_rerank_respects_top_k(self) -> None:
        """Returns only top_k results."""
        reranker = self._create_reranker([0.1, 0.9, 0.5])
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"], top_k=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_arerank_returns_correct_format(self) -> None:
        """Async rerank returns correct results."""
        reranker = self._create_reranker([0.3, 0.7])
        results = await reranker.arerank("query", ["doc1", "doc2"])

        assert len(results) == 2
        assert results[0].index == 1


# ===== FlagEmbeddingReranker Tests =====


class TestFlagEmbeddingRerankerInit:
    """Tests for FlagEmbeddingReranker initialization."""

    def test_import_error_when_missing_dependency(self) -> None:
        """Raises ImportError when FlagEmbedding is not installed."""
        with patch.dict("sys.modules", {"FlagEmbedding": None}), pytest.raises(ImportError, match="FlagEmbedding"):
            from autorag_research.rerankers.flag_embedding import FlagEmbeddingReranker

            FlagEmbeddingReranker()

    def test_default_model_name(self) -> None:
        """Default model_name is BAAI/bge-reranker-large."""
        from autorag_research.rerankers.flag_embedding import FlagEmbeddingReranker

        assert FlagEmbeddingReranker.model_fields["model_name"].default == "BAAI/bge-reranker-large"


class TestFlagEmbeddingRerankerRerank:
    """Tests for FlagEmbeddingReranker.rerank method."""

    def _create_reranker(self, mock_scores: list[float] | float) -> object:
        """Create a FlagEmbeddingReranker with mocked model."""
        from autorag_research.rerankers.flag_embedding import FlagEmbeddingReranker

        reranker = FlagEmbeddingReranker.__new__(FlagEmbeddingReranker)
        mock_model = MagicMock()
        mock_model.compute_score.return_value = mock_scores
        object.__setattr__(reranker, "_model", mock_model)
        object.__setattr__(reranker, "model_name", "test-model")
        object.__setattr__(reranker, "batch_size", 64)
        object.__setattr__(reranker, "max_concurrency", 10)
        object.__setattr__(reranker, "use_fp16", False)
        return reranker

    def test_empty_documents(self) -> None:
        """Returns empty list for empty documents."""
        reranker = self._create_reranker([])
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_single_document(self) -> None:
        """Handles single document (compute_score returns float)."""
        reranker = self._create_reranker(0.95)
        results = reranker.rerank("query", ["doc1"])

        assert len(results) == 1
        assert results[0].score == 0.95

    def test_rerank_multiple_documents(self) -> None:
        """Returns sorted results for multiple documents."""
        reranker = self._create_reranker([0.2, 0.8, 0.5])
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"])

        assert len(results) == 3
        assert results[0].index == 1
        assert results[0].score == 0.8

    def test_rerank_respects_top_k(self) -> None:
        """Returns only top_k results."""
        reranker = self._create_reranker([0.2, 0.8, 0.5])
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"], top_k=1)
        assert len(results) == 1
        assert results[0].index == 1

    @pytest.mark.asyncio
    async def test_arerank_works(self) -> None:
        """Async rerank returns correct results."""
        reranker = self._create_reranker([0.3, 0.7])
        results = await reranker.arerank("query", ["doc1", "doc2"])
        assert len(results) == 2


# ===== FlagEmbeddingLLMReranker Tests =====


class TestFlagEmbeddingLLMRerankerInit:
    """Tests for FlagEmbeddingLLMReranker initialization."""

    def test_import_error_when_missing_dependency(self) -> None:
        """Raises ImportError when FlagEmbedding is not installed."""
        with patch.dict("sys.modules", {"FlagEmbedding": None}), pytest.raises(ImportError, match="FlagEmbedding"):
            from autorag_research.rerankers.flag_embedding_llm import FlagEmbeddingLLMReranker

            FlagEmbeddingLLMReranker()

    def test_default_model_name(self) -> None:
        """Default model_name is BAAI/bge-reranker-v2-gemma."""
        from autorag_research.rerankers.flag_embedding_llm import FlagEmbeddingLLMReranker

        assert FlagEmbeddingLLMReranker.model_fields["model_name"].default == "BAAI/bge-reranker-v2-gemma"


class TestFlagEmbeddingLLMRerankerRerank:
    """Tests for FlagEmbeddingLLMReranker.rerank method."""

    def _create_reranker(self, mock_scores: list[float] | float) -> object:
        """Create a FlagEmbeddingLLMReranker with mocked model."""
        from autorag_research.rerankers.flag_embedding_llm import FlagEmbeddingLLMReranker

        reranker = FlagEmbeddingLLMReranker.__new__(FlagEmbeddingLLMReranker)
        mock_model = MagicMock()
        mock_model.compute_score.return_value = mock_scores
        object.__setattr__(reranker, "_model", mock_model)
        object.__setattr__(reranker, "model_name", "test-model")
        object.__setattr__(reranker, "batch_size", 64)
        object.__setattr__(reranker, "max_concurrency", 10)
        object.__setattr__(reranker, "use_fp16", False)
        return reranker

    def test_empty_documents(self) -> None:
        """Returns empty list for empty documents."""
        reranker = self._create_reranker([])
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_multiple_documents(self) -> None:
        """Returns sorted results for multiple documents."""
        reranker = self._create_reranker([0.1, 0.9, 0.4])
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"])

        assert len(results) == 3
        assert results[0].index == 1

    @pytest.mark.asyncio
    async def test_arerank_works(self) -> None:
        """Async rerank returns correct results."""
        reranker = self._create_reranker([0.5, 0.3])
        results = await reranker.arerank("query", ["doc1", "doc2"])
        assert len(results) == 2


# ===== FlashRankReranker Tests =====


class TestFlashRankRerankerInit:
    """Tests for FlashRankReranker initialization."""

    def test_import_error_when_missing_dependency(self) -> None:
        """Raises ImportError when flashrank is not installed."""
        with patch.dict("sys.modules", {"flashrank": None}), pytest.raises(ImportError, match="flashrank"):
            from autorag_research.rerankers.flashrank import FlashRankReranker

            FlashRankReranker()

    def test_default_model_name(self) -> None:
        """Default model_name is ms-marco-MiniLM-L-12-v2."""
        from autorag_research.rerankers.flashrank import FlashRankReranker

        assert FlashRankReranker.model_fields["model_name"].default == "ms-marco-MiniLM-L-12-v2"


class TestFlashRankRerankerRerank:
    """Tests for FlashRankReranker.rerank method."""

    def _create_reranker(self, mock_response: list[dict]) -> object:
        """Create a FlashRankReranker with mocked ranker."""
        from autorag_research.rerankers.flashrank import FlashRankReranker

        reranker = FlashRankReranker.__new__(FlashRankReranker)
        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = mock_response
        object.__setattr__(reranker, "_ranker", mock_ranker)
        object.__setattr__(reranker, "model_name", "test-model")
        object.__setattr__(reranker, "batch_size", 64)
        object.__setattr__(reranker, "max_concurrency", 10)
        object.__setattr__(reranker, "max_length", 512)
        return reranker

    def test_empty_documents(self) -> None:
        """Returns empty list for empty documents."""
        reranker = self._create_reranker([])
        results = reranker.rerank("query", [])
        assert results == []

    @patch("autorag_research.rerankers.flashrank.RerankRequest", create=True)
    def test_rerank_returns_correct_format(self, mock_rr: MagicMock) -> None:
        """Returns sorted RerankResult list."""
        mock_response = [
            {"id": 1, "text": "doc2", "score": 0.9},
            {"id": 0, "text": "doc1", "score": 0.3},
        ]
        reranker = self._create_reranker(mock_response)

        with patch.dict("sys.modules", {"flashrank": MagicMock()}):
            results = reranker.rerank("query", ["doc1", "doc2"])

        assert len(results) == 2
        assert results[0].score >= results[1].score

    @patch("autorag_research.rerankers.flashrank.RerankRequest", create=True)
    def test_rerank_respects_top_k(self, mock_rr: MagicMock) -> None:
        """Returns only top_k results."""
        mock_response = [
            {"id": 1, "text": "doc2", "score": 0.9},
            {"id": 2, "text": "doc3", "score": 0.5},
            {"id": 0, "text": "doc1", "score": 0.1},
        ]
        reranker = self._create_reranker(mock_response)

        with patch.dict("sys.modules", {"flashrank": MagicMock()}):
            results = reranker.rerank("query", ["doc1", "doc2", "doc3"], top_k=1)

        assert len(results) == 1

    @pytest.mark.asyncio
    @patch("autorag_research.rerankers.flashrank.RerankRequest", create=True)
    async def test_arerank_works(self, mock_rr: MagicMock) -> None:
        """Async rerank returns correct results."""
        mock_response = [
            {"id": 0, "text": "doc1", "score": 0.8},
            {"id": 1, "text": "doc2", "score": 0.2},
        ]
        reranker = self._create_reranker(mock_response)

        with patch.dict("sys.modules", {"flashrank": MagicMock()}):
            results = await reranker.arerank("query", ["doc1", "doc2"])

        assert len(results) == 2


# ===== ColBERTReranker Tests =====


class TestColBERTRerankerInit:
    """Tests for ColBERTReranker initialization."""

    def test_import_error_when_missing_dependency(self) -> None:
        """Raises ImportError when torch/transformers is not installed."""
        with patch.dict("sys.modules", {"torch": None}), pytest.raises(ImportError, match="torch and transformers"):
            from autorag_research.rerankers.colbert import ColBERTReranker

            ColBERTReranker()

    def test_default_model_name(self) -> None:
        """Default model_name is colbert-ir/colbertv2.0."""
        from autorag_research.rerankers.colbert import ColBERTReranker

        assert ColBERTReranker.model_fields["model_name"].default == "colbert-ir/colbertv2.0"


class TestColBERTRerankerRerank:
    """Tests for ColBERTReranker.rerank method."""

    def _create_reranker(self) -> object:
        """Create a ColBERTReranker with mocked model."""
        from autorag_research.rerankers.colbert import ColBERTReranker

        reranker = ColBERTReranker.__new__(ColBERTReranker)
        object.__setattr__(reranker, "_model", MagicMock())
        object.__setattr__(reranker, "_tokenizer", MagicMock())
        object.__setattr__(reranker, "_device", "cpu")
        object.__setattr__(reranker, "model_name", "test-model")
        object.__setattr__(reranker, "batch_size", 64)
        object.__setattr__(reranker, "max_concurrency", 10)
        object.__setattr__(reranker, "max_length", 512)
        object.__setattr__(reranker, "device", None)
        return reranker

    def _create_reranker_with_mock_encode(self, scores: list[float]) -> tuple[object, list]:
        """Create a ColBERTReranker with mocked _encode and _maxsim_score.

        Args:
            scores: Pre-defined MaxSim scores for each document.

        Returns:
            Tuple of (reranker, encode_call_args) where encode_call_args
            accumulates the texts passed to each _encode call.
        """
        import torch

        from autorag_research.rerankers.colbert import ColBERTReranker

        reranker = ColBERTReranker.__new__(ColBERTReranker)

        encode_call_args: list[list[str]] = []
        dim = 32

        def mock_encode(texts: list[str]):
            encode_call_args.append(list(texts))
            n = len(texts)
            return torch.randn(n, 10, dim), torch.ones(n, 10, dtype=torch.long)

        score_idx = [0]

        def mock_maxsim(q_emb, q_mask, d_emb, d_mask):
            s = scores[score_idx[0]]
            score_idx[0] += 1
            return s

        object.__setattr__(reranker, "_encode", mock_encode)
        object.__setattr__(reranker, "_maxsim_score", mock_maxsim)
        object.__setattr__(reranker, "_model", MagicMock())
        object.__setattr__(reranker, "_tokenizer", MagicMock())
        object.__setattr__(reranker, "_device", "cpu")
        object.__setattr__(reranker, "model_name", "test-model")
        object.__setattr__(reranker, "batch_size", 64)
        object.__setattr__(reranker, "max_concurrency", 10)
        object.__setattr__(reranker, "max_length", 512)
        object.__setattr__(reranker, "device", None)
        return reranker, encode_call_args

    def test_empty_documents(self) -> None:
        """Returns empty list for empty documents."""
        reranker = self._create_reranker()
        results = reranker.rerank("query", [])
        assert results == []

    def test_batch_encode_called_twice(self) -> None:
        """Documents are batch-encoded with exactly two _encode calls."""
        reranker, encode_call_args = self._create_reranker_with_mock_encode([0.8, 0.5, 0.3])
        reranker.rerank("query", ["doc1", "doc2", "doc3"])

        assert len(encode_call_args) == 2
        assert encode_call_args[0] == ["query"]
        assert encode_call_args[1] == ["doc1", "doc2", "doc3"]

    def test_batch_returns_correct_format(self) -> None:
        """Returns sorted RerankResult list from batch encoding."""
        reranker, _ = self._create_reranker_with_mock_encode([0.3, 0.9, 0.6])
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"])

        assert len(results) == 3
        assert all(isinstance(r, RerankResult) for r in results)
        assert results[0].score >= results[1].score >= results[2].score

    def test_batch_preserves_original_indices(self) -> None:
        """Results preserve original document indices after batch scoring and sorting."""
        reranker, _ = self._create_reranker_with_mock_encode([0.2, 0.9, 0.5])
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"])

        assert results[0].index == 1
        assert results[0].text == "doc2"
        assert results[-1].index == 0
        assert results[-1].text == "doc1"

    def test_batch_respects_top_k(self) -> None:
        """Returns only top_k results from batch encoding."""
        reranker, _ = self._create_reranker_with_mock_encode([0.2, 0.9, 0.5])
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"], top_k=2)
        assert len(results) == 2
        assert results[0].index == 1


# ===== MonoT5Reranker Tests =====


class TestMonoT5RerankerInit:
    """Tests for MonoT5Reranker initialization."""

    def test_import_error_when_missing_dependency(self) -> None:
        """Raises ImportError when torch/transformers is not installed."""
        with patch.dict("sys.modules", {"torch": None}), pytest.raises(ImportError, match="torch and transformers"):
            from autorag_research.rerankers.monot5 import MonoT5Reranker

            MonoT5Reranker()

    def test_default_model_name(self) -> None:
        """Default model_name is castorini/monot5-3b-msmarco-10k."""
        from autorag_research.rerankers.monot5 import MonoT5Reranker

        assert MonoT5Reranker.model_fields["model_name"].default == "castorini/monot5-3b-msmarco-10k"


class TestMonoT5RerankerRerank:
    """Tests for MonoT5Reranker.rerank method."""

    def _create_reranker(self, num_docs: int, true_logits: list[float] | None = None) -> object:
        """Create a MonoT5Reranker with mocked model for batch inference.

        Args:
            num_docs: Number of documents (determines batch size for mock logits).
            true_logits: Per-document logit values for the "true" token.
                         Defaults to 5.0 for all documents.
        """
        import torch

        from autorag_research.rerankers.monot5 import MonoT5Reranker

        reranker = MonoT5Reranker.__new__(MonoT5Reranker)

        if true_logits is None:
            true_logits = [5.0] * num_docs

        mock_model = MagicMock()
        mock_scores = MagicMock()
        batch_logits = torch.zeros(num_docs, 32128)
        for i, tl in enumerate(true_logits):
            batch_logits[i, 1176] = tl
            batch_logits[i, 6136] = 1.0
        mock_scores.scores = [batch_logits]
        mock_model.generate.return_value = mock_scores

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text, add_special_tokens: [1176] if text == "true" else [6136]
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        object.__setattr__(reranker, "_model", mock_model)
        object.__setattr__(reranker, "_tokenizer", mock_tokenizer)
        object.__setattr__(reranker, "_device", "cpu")
        object.__setattr__(reranker, "_true_token_id", 1176)
        object.__setattr__(reranker, "_false_token_id", 6136)
        object.__setattr__(reranker, "model_name", "test-model")
        object.__setattr__(reranker, "batch_size", 64)
        object.__setattr__(reranker, "max_concurrency", 10)
        object.__setattr__(reranker, "max_length", 512)
        object.__setattr__(reranker, "device", None)
        return reranker

    def test_empty_documents(self) -> None:
        """Returns empty list for empty documents."""
        reranker = self._create_reranker(1)
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_returns_correct_format(self) -> None:
        """Returns list of RerankResult."""
        reranker = self._create_reranker(2)
        results = reranker.rerank("query", ["doc1", "doc2"])

        assert len(results) == 2
        assert all(isinstance(r, RerankResult) for r in results)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_rerank_respects_top_k(self) -> None:
        """Returns only top_k results."""
        reranker = self._create_reranker(3)
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"], top_k=1)
        assert len(results) == 1

    def test_batch_generate_called_once(self) -> None:
        """Model generate is called exactly once for batch inference."""
        reranker = self._create_reranker(3)
        reranker.rerank("query", ["doc1", "doc2", "doc3"])
        assert reranker._model.generate.call_count == 1

    def test_batch_tokenizer_receives_all_prompts(self) -> None:
        """Tokenizer receives all prompts at once."""
        reranker = self._create_reranker(2)
        reranker.rerank("test query", ["doc1", "doc2"])

        call_args = reranker._tokenizer.call_args
        prompts = call_args[0][0]
        assert len(prompts) == 2
        assert prompts[0] == "Query: test query Document: doc1 Relevant:"
        assert prompts[1] == "Query: test query Document: doc2 Relevant:"

    def test_batch_preserves_original_indices(self) -> None:
        """Results preserve original document indices after batch scoring and sorting."""
        # doc1: low true logit, doc2: high true logit, doc3: medium
        reranker = self._create_reranker(3, true_logits=[1.0, 10.0, 5.0])
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"])

        assert results[0].index == 1
        assert results[0].text == "doc2"
        assert results[-1].index == 0
        assert results[-1].text == "doc1"


# ===== TARTReranker Tests =====


class TestTARTRerankerInit:
    """Tests for TARTReranker initialization."""

    def test_import_error_when_missing_dependency(self) -> None:
        """Raises ImportError when torch/transformers is not installed."""
        with patch.dict("sys.modules", {"torch": None}), pytest.raises(ImportError, match="torch and transformers"):
            from autorag_research.rerankers.tart import TARTReranker

            TARTReranker()

    def test_default_model_name(self) -> None:
        """Default model_name is facebook/tart-full-flan-t5-xl."""
        from autorag_research.rerankers.tart import TARTReranker

        assert TARTReranker.model_fields["model_name"].default == "facebook/tart-full-flan-t5-xl"

    def test_default_instruction(self) -> None:
        """Default instruction is for question answering."""
        from autorag_research.rerankers.tart import TARTReranker

        assert TARTReranker.model_fields["instruction"].default == "Find passage to answer given question"


class TestTARTRerankerRerank:
    """Tests for TARTReranker.rerank method."""

    def _create_reranker(self, mock_logits_values: list[list[float]]) -> object:
        """Create a TARTReranker with mocked model for batch inference."""
        import torch

        from autorag_research.rerankers.tart import TARTReranker

        reranker = TARTReranker.__new__(TARTReranker)

        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.logits = torch.tensor(mock_logits_values)
        mock_model.return_value = mock_result

        mock_tokenizer = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        object.__setattr__(reranker, "_model", mock_model)
        object.__setattr__(reranker, "_tokenizer", mock_tokenizer)
        object.__setattr__(reranker, "_device", "cpu")
        object.__setattr__(reranker, "model_name", "test-model")
        object.__setattr__(reranker, "batch_size", 64)
        object.__setattr__(reranker, "max_concurrency", 10)
        object.__setattr__(reranker, "max_length", 512)
        object.__setattr__(reranker, "device", None)
        object.__setattr__(reranker, "instruction", "Find passage to answer given question")
        return reranker

    def test_empty_documents(self) -> None:
        """Returns empty list for empty documents."""
        reranker = self._create_reranker([[0.5, 0.5]])
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_returns_correct_format(self) -> None:
        """Returns list of RerankResult sorted by score."""
        reranker = self._create_reranker([[0.1, 0.9], [0.8, 0.2]])
        results = reranker.rerank("query", ["doc1", "doc2"])

        assert len(results) == 2
        assert all(isinstance(r, RerankResult) for r in results)
        assert results[0].score >= results[1].score

    def test_rerank_respects_top_k(self) -> None:
        """Returns only top_k results."""
        reranker = self._create_reranker([[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]])
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"], top_k=2)
        assert len(results) == 2

    def test_batch_model_called_once(self) -> None:
        """Model is called exactly once for batch inference."""
        reranker = self._create_reranker([[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]])
        reranker.rerank("query", ["doc1", "doc2", "doc3"])
        assert reranker._model.call_count == 1

    def test_batch_tokenizer_receives_pairs(self) -> None:
        """Tokenizer receives all query-document pairs at once."""
        reranker = self._create_reranker([[0.1, 0.9], [0.8, 0.2]])
        reranker.rerank("test query", ["doc1", "doc2"])

        call_args = reranker._tokenizer.call_args
        pairs = call_args[0][0]
        assert len(pairs) == 2
        assert all(p[0] == "Find passage to answer given question [SEP] test query" for p in pairs)
        assert pairs[0][1] == "doc1"
        assert pairs[1][1] == "doc2"

    def test_batch_preserves_original_indices(self) -> None:
        """Results preserve original document indices after batch scoring and sorting."""
        # doc1: low positive class, doc2: high positive class, doc3: medium
        reranker = self._create_reranker([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"])

        assert results[0].index == 1  # doc2 (highest positive class)
        assert results[0].text == "doc2"
        assert results[-1].index == 0  # doc1 (lowest positive class)
        assert results[-1].text == "doc1"


# ===== KoRerankerReranker Tests =====


class TestKoRerankerRerankerInit:
    """Tests for KoRerankerReranker initialization."""

    def test_import_error_when_missing_dependency(self) -> None:
        """Raises ImportError when torch/transformers is not installed."""
        with patch.dict("sys.modules", {"torch": None}), pytest.raises(ImportError, match="torch and transformers"):
            from autorag_research.rerankers.koreranker import KoRerankerReranker

            KoRerankerReranker()

    def test_default_model_name(self) -> None:
        """Default model_name is Dongjin-kr/ko-reranker."""
        from autorag_research.rerankers.koreranker import KoRerankerReranker

        assert KoRerankerReranker.model_fields["model_name"].default == "Dongjin-kr/ko-reranker"


class TestKoRerankerRerankerRerank:
    """Tests for KoRerankerReranker.rerank method."""

    def _create_reranker(self, mock_logits: list[float]) -> object:
        """Create a KoRerankerReranker with mocked model."""
        import torch

        from autorag_research.rerankers.koreranker import KoRerankerReranker

        reranker = KoRerankerReranker.__new__(KoRerankerReranker)

        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.logits = torch.tensor(mock_logits).unsqueeze(-1)
        mock_model.return_value = mock_result

        mock_tokenizer = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        object.__setattr__(reranker, "_model", mock_model)
        object.__setattr__(reranker, "_tokenizer", mock_tokenizer)
        object.__setattr__(reranker, "_device", "cpu")
        object.__setattr__(reranker, "model_name", "test-model")
        object.__setattr__(reranker, "batch_size", 64)
        object.__setattr__(reranker, "max_concurrency", 10)
        object.__setattr__(reranker, "max_length", 512)
        object.__setattr__(reranker, "device", None)
        return reranker

    def test_empty_documents(self) -> None:
        """Returns empty list for empty documents."""
        reranker = self._create_reranker([])
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_returns_correct_format(self) -> None:
        """Returns list of RerankResult with normalized scores."""
        reranker = self._create_reranker([1.0, 3.0, 2.0])
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"])

        assert len(results) == 3
        assert all(isinstance(r, RerankResult) for r in results)
        assert results[0].index == 1  # doc2 with highest logit
        # Scores should sum to approximately 1 (exp normalized)
        total = sum(r.score for r in results)
        assert abs(total - 1.0) < 1e-6

    def test_rerank_respects_top_k(self) -> None:
        """Returns only top_k results."""
        reranker = self._create_reranker([1.0, 3.0, 2.0])
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"], top_k=1)
        assert len(results) == 1

    def test_exp_normalize(self) -> None:
        """exp_normalize produces valid probability distribution."""
        from autorag_research.rerankers.koreranker import KoRerankerReranker

        scores = [1.0, 2.0, 3.0]
        normalized = KoRerankerReranker._exp_normalize(scores)

        assert len(normalized) == 3
        assert all(0 <= s <= 1 for s in normalized)
        assert abs(sum(normalized) - 1.0) < 1e-6
        # Higher input should produce higher output
        assert normalized[2] > normalized[1] > normalized[0]

    @pytest.mark.asyncio
    async def test_arerank_works(self) -> None:
        """Async rerank returns correct results."""
        reranker = self._create_reranker([1.0, 3.0])
        results = await reranker.arerank("query", ["doc1", "doc2"])
        assert len(results) == 2


# ===== OpenVINOReranker Tests =====


class TestOpenVINORerankerInit:
    """Tests for OpenVINOReranker initialization."""

    def test_import_error_when_missing_dependency(self) -> None:
        """Raises ImportError when optimum-intel is not installed."""
        with (
            patch.dict("sys.modules", {"optimum": None, "optimum.intel": None, "optimum.intel.openvino": None}),
            pytest.raises(ImportError, match="optimum-intel"),
        ):
            from autorag_research.rerankers.openvino import OpenVINOReranker

            OpenVINOReranker()

    def test_default_model_name(self) -> None:
        """Default model_name is BAAI/bge-reranker-large."""
        from autorag_research.rerankers.openvino import OpenVINOReranker

        assert OpenVINOReranker.model_fields["model_name"].default == "BAAI/bge-reranker-large"


class TestOpenVINORerankerRerank:
    """Tests for OpenVINOReranker.rerank method."""

    def _create_reranker(self, mock_logits: list[float]) -> object:
        """Create an OpenVINOReranker with mocked model."""
        import torch

        from autorag_research.rerankers.openvino import OpenVINOReranker

        reranker = OpenVINOReranker.__new__(OpenVINOReranker)

        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.logits = torch.tensor(mock_logits).unsqueeze(-1)
        mock_model.return_value = mock_result

        mock_tokenizer = MagicMock()
        mock_inputs = MagicMock()
        mock_tokenizer.return_value = mock_inputs

        object.__setattr__(reranker, "_model", mock_model)
        object.__setattr__(reranker, "_tokenizer", mock_tokenizer)
        object.__setattr__(reranker, "model_name", "test-model")
        object.__setattr__(reranker, "batch_size", 64)
        object.__setattr__(reranker, "max_concurrency", 10)
        object.__setattr__(reranker, "max_length", 512)
        return reranker

    def test_empty_documents(self) -> None:
        """Returns empty list for empty documents."""
        reranker = self._create_reranker([])
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_returns_correct_format(self) -> None:
        """Returns list of RerankResult with sigmoid scores."""
        reranker = self._create_reranker([-2.0, 2.0, 0.0])
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"])

        assert len(results) == 3
        assert all(isinstance(r, RerankResult) for r in results)
        assert results[0].index == 1  # doc2 with highest logit
        # Sigmoid should produce values between 0 and 1
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_rerank_respects_top_k(self) -> None:
        """Returns only top_k results."""
        reranker = self._create_reranker([-1.0, 1.0, 0.0])
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"], top_k=2)
        assert len(results) == 2

    def test_sigmoid(self) -> None:
        """Sigmoid function works correctly."""
        from autorag_research.rerankers.openvino import OpenVINOReranker

        assert abs(OpenVINOReranker._sigmoid(0.0) - 0.5) < 1e-6
        assert OpenVINOReranker._sigmoid(100.0) > 0.99
        assert OpenVINOReranker._sigmoid(-100.0) < 0.01

    @pytest.mark.asyncio
    async def test_arerank_works(self) -> None:
        """Async rerank returns correct results."""
        reranker = self._create_reranker([1.0, -1.0])
        results = await reranker.arerank("query", ["doc1", "doc2"])
        assert len(results) == 2
