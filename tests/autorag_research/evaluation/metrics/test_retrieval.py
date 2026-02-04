import numpy as np
import pytest

from autorag_research.evaluation.metrics.retrieval import (
    retrieval_f1,
    retrieval_map,
    retrieval_mrr,
    retrieval_ndcg,
    retrieval_precision,
    retrieval_recall,
)
from autorag_research.schema import MetricInput

retrieval_gt = [
    [["test-1", "test-2"], ["test-3"]],
    [["test-4", "test-5"], ["test-6", "test-7"], ["test-8"]],
    [["test-9", "test-10"]],
    [["test-11"], ["test-12"], ["test-13"]],
    [["test-14"]],
    [[]],
    [[""]],
    [["test-15"]],
]

pred = [
    ["test-1", "pred-1", "test-2", "pred-3"],  # recall: 0.5, precision: 0.5, f1: 0.5
    ["test-6", "pred-5", "pred-6", "pred-7"],  # recall: 1/3, precision: 0.25, f1: 2/7
    ["test-9", "pred-0", "pred-8", "pred-9"],  # recall: 1.0, precision: 0.25, f1: 2/5
    [
        "test-13",
        "test-12",
        "pred-10",
        "pred-11",
    ],  # recall: 2/3, precision: 0.5, f1: 4/7
    ["test-14", "pred-12"],  # recall: 1.0, precision: 0.5, f1: 2/3
    ["pred-13"],  # retrieval_gt is empty so not counted
    ["pred-14"],  # retrieval_gt is empty so not counted
    ["pred-15", "pred-16", "test-15"],  # recall:1, precision: 1/3, f1: 0.5
]
metric_inputs = [
    MetricInput(retrieval_gt=ret_gt, retrieved_ids=pr) for ret_gt, pr in zip(retrieval_gt, pred, strict=True)
]


def test_retrieval_f1():
    solution = [0.5, 2 / 7, 2 / 5, 4 / 7, 2 / 3, None, None, 0.5]
    result = retrieval_f1(metric_inputs=metric_inputs)
    for gt, res in zip(solution, result, strict=True):
        assert gt == pytest.approx(res, rel=1e-4)


def test_numpy_retrieval_metric():
    retrieval_gt_np = [[np.array(["test-1", "test-4"])], np.array([["test-2"]])]
    pred_np = np.array([["test-2", "test-3", "test-1"], ["test-5", "test-6", "test-8"]])
    solution = [1.0, 0.0]
    metric_inputs_np = [
        MetricInput(retrieval_gt=ret_gt_np, retrieved_ids=pr_np)
        for ret_gt_np, pr_np in zip(retrieval_gt_np, pred_np, strict=True)
    ]
    result = retrieval_recall(metric_inputs=metric_inputs_np)
    for gt, res in zip(solution, result, strict=True):
        assert gt == pytest.approx(res, rel=1e-4)


def test_retrieval_recall():
    solution = [0.5, 1 / 3, 1, 2 / 3, 1, None, None, 1]
    result = retrieval_recall(metric_inputs=metric_inputs)
    for gt, res in zip(solution, result, strict=True):
        assert gt == pytest.approx(res, rel=1e-4)


def test_retrieval_precision():
    solution = [0.5, 0.25, 0.25, 0.5, 0.5, None, None, 1 / 3]
    result = retrieval_precision(metric_inputs=metric_inputs)
    for gt, res in zip(solution, result, strict=True):
        assert gt == pytest.approx(res, rel=1e-4)


def test_retrieval_ndcg():
    solution = [
        0.7039180890341347,
        0.3903800499921017,
        0.6131471927654584,
        0.7653606369886217,
        1,
        None,
        None,
        0.5,
    ]
    result = retrieval_ndcg(metric_inputs=metric_inputs)
    for gt, res in zip(solution, result, strict=True):
        assert gt == pytest.approx(res, rel=1e-4)


def test_retrieval_mrr():
    solution = [1 / 2, 1 / 3, 1, 1 / 2, 1, None, None, 1 / 3]
    result = retrieval_mrr(metric_inputs=metric_inputs)
    for gt, res in zip(solution, result, strict=True):
        assert gt == pytest.approx(res, rel=1e-4)


def test_retrieval_map():
    solution = [5 / 12, 1 / 3, 1, 1 / 2, 1, None, None, 1 / 3]
    result = retrieval_map(metric_inputs=metric_inputs)
    for gt, res in zip(solution, result, strict=True):
        assert gt == pytest.approx(res, rel=1e-4)


class TestRetrievalNdcgGradedRelevance:
    """Test NDCG with graded relevance scores."""

    def test_graded_ndcg_highly_relevant_first(self):
        """Highly relevant document ranked first should score higher."""
        # Ground truth: doc_a (score=2), doc_b (score=1)
        # Retrieved: [doc_a, doc_b] - perfect ranking
        metric_input = MetricInput(
            retrieval_gt=[["doc_a", "doc_b"]],
            retrieved_ids=["doc_a", "doc_b"],
            relevance_scores={"doc_a": 2, "doc_b": 1},
        )
        result = retrieval_ndcg.__wrapped__(metric_input)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_graded_ndcg_highly_relevant_last(self):
        """Highly relevant document ranked last should score lower."""
        # Ground truth: doc_a (score=2), doc_b (score=1)
        # Retrieved: [doc_b, doc_a] - suboptimal ranking
        metric_input = MetricInput(
            retrieval_gt=[["doc_a", "doc_b"]],
            retrieved_ids=["doc_b", "doc_a"],
            relevance_scores={"doc_a": 2, "doc_b": 1},
        )
        result = retrieval_ndcg.__wrapped__(metric_input)
        # DCG: (2^1 - 1)/log2(2) + (2^2 - 1)/log2(3) = 1/1 + 3/1.585 = 1 + 1.893 = 2.893
        # IDCG: (2^2 - 1)/log2(2) + (2^1 - 1)/log2(3) = 3/1 + 1/1.585 = 3 + 0.631 = 3.631
        # NDCG: 2.893 / 3.631 ≈ 0.797
        assert result < 1.0
        assert result == pytest.approx(0.7969, rel=1e-3)

    def test_graded_ndcg_with_irrelevant_retrieved(self):
        """Retrieved documents not in ground truth should have score 0."""
        # Ground truth: doc_a (score=2)
        # Retrieved: [irrelevant, doc_a]
        metric_input = MetricInput(
            retrieval_gt=[["doc_a"]],
            retrieved_ids=["irrelevant", "doc_a"],
            relevance_scores={"doc_a": 2},
        )
        result = retrieval_ndcg.__wrapped__(metric_input)
        # DCG: (2^0 - 1)/log2(2) + (2^2 - 1)/log2(3) = 0 + 3/1.585 = 1.893
        # IDCG: (2^2 - 1)/log2(2) + (2^0 - 1)/log2(3) = 3/1 + 0 = 3
        # NDCG: 1.893 / 3 ≈ 0.631
        assert result == pytest.approx(0.631, rel=1e-2)

    def test_graded_ndcg_no_relevance_scores_backward_compat(self):
        """Without relevance_scores, should use binary relevance (backward compatible)."""
        metric_input = MetricInput(
            retrieval_gt=[["doc_a", "doc_b"]],
            retrieved_ids=["doc_a", "doc_b"],
            relevance_scores=None,  # No graded relevance
        )
        result = retrieval_ndcg.__wrapped__(metric_input)
        # With binary relevance, all relevant docs have score=1, so perfect ranking = 1.0
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_graded_ndcg_three_levels(self):
        """Test with three relevance levels: 0, 1, 2."""
        # Ground truth: highly=2, somewhat=1, not_relevant=0 (but not_relevant not in GT)
        # Retrieved: [highly, somewhat, irrelevant]
        metric_input = MetricInput(
            retrieval_gt=[["highly", "somewhat"]],
            retrieved_ids=["highly", "somewhat", "irrelevant"],
            relevance_scores={"highly": 2, "somewhat": 1},
        )
        result = retrieval_ndcg.__wrapped__(metric_input)
        # DCG: (2^2-1)/log2(2) + (2^1-1)/log2(3) + (2^0-1)/log2(4) = 3 + 0.631 + 0 = 3.631
        # IDCG: (2^2-1)/log2(2) + (2^1-1)/log2(3) + (2^0-1)/log2(4) = 3 + 0.631 + 0 = 3.631
        # NDCG: 1.0 (perfect ranking)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_graded_ndcg_score_zero_items(self):
        """Test with score=0 items (marked as not relevant but in GT)."""
        metric_input = MetricInput(
            retrieval_gt=[["doc_a", "doc_b"]],
            retrieved_ids=["doc_a", "doc_b"],
            relevance_scores={"doc_a": 2, "doc_b": 0},
        )
        result = retrieval_ndcg.__wrapped__(metric_input)
        # DCG: (2^2-1)/log2(2) + (2^0-1)/log2(3) = 3 + 0 = 3
        # IDCG: (2^2-1)/log2(2) + (2^0-1)/log2(3) = 3 + 0 = 3
        # NDCG: 1.0 (doc_b has score 0, so order doesn't matter for it)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_graded_ndcg_empty_retrieval(self):
        """Empty retrieval should return 0."""
        metric_input = MetricInput(
            retrieval_gt=[["doc_a"]],
            retrieved_ids=[],
            relevance_scores={"doc_a": 2},
        )
        result = retrieval_ndcg.__wrapped__(metric_input)
        assert result == 0.0

    def test_graded_ndcg_multiple_groups(self):
        """Test with multiple ground truth groups (AND/OR structure)."""
        metric_input = MetricInput(
            retrieval_gt=[["doc_a", "doc_b"], ["doc_c"]],  # (a OR b) AND c
            retrieved_ids=["doc_a", "doc_c", "doc_b"],
            relevance_scores={"doc_a": 2, "doc_b": 1, "doc_c": 2},
        )
        result = retrieval_ndcg.__wrapped__(metric_input)
        # All docs retrieved, but order matters for NDCG
        assert result > 0.0
        assert result <= 1.0
