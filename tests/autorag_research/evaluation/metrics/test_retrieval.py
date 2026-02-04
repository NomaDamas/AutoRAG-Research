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
    """Test NDCG with multi-hop AND-OR semantics.

    New semantics: [[A,B], [C]] means (A OR B) AND C
    - An item contributes to DCG only when it satisfies a NEW group
    - Redundant items from already-satisfied groups don't add value
    - IDCG = best item from each group at ideal positions

    Case 0: GT=[[test-1,test-2], [test-3]], pred=[test-1, pred-1, test-2, pred-3]
            test-1 satisfies G0, test-3 never retrieved (G1 unsatisfied)
            DCG=1, IDCG=1+1/log2(3)=1.631, NDCG=0.613

    Case 1: GT=[[test-4,test-5], [test-6,test-7], [test-8]], pred=[test-6, ...]
            Only test-6 (G1) retrieved. G0,G2 unsatisfied.
            DCG=1, IDCG=1+0.631+0.5=2.131, NDCG=0.469

    Case 2: GT=[[test-9,test-10]], pred=[test-9, ...]
            Single group, test-9 at rank 0. DCG=IDCG=1, NDCG=1.0

    Case 3: GT=[[test-11], [test-12], [test-13]], pred=[test-13, test-12, ...]
            test-13(G2) at rank 0, test-12(G1) at rank 1. G0 unsatisfied.
            DCG=1+0.631=1.631, IDCG=2.131, NDCG=0.765

    Case 4: GT=[[test-14]], pred=[test-14, ...]
            Perfect single-group retrieval. NDCG=1.0

    Cases 5,6: Empty GT -> None

    Case 7: GT=[[test-15]], pred=[pred-15, pred-16, test-15]
            test-15 at rank 2. DCG=1/log2(4)=0.5, IDCG=1, NDCG=0.5
    """
    solution = [
        0.6131471927654584,  # Case 0: only G0 satisfied
        0.4693015838914927,  # Case 1: only G1 satisfied (1 of 3 groups)
        1.0,  # Case 2: single group, perfect
        0.7653606369886217,  # Case 3: 2 of 3 groups satisfied
        1,  # Case 4: single group, perfect
        None,  # Case 5: empty GT
        None,  # Case 6: empty GT
        0.5,  # Case 7: single group, rank 2
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
        """Highly relevant document ranked last should score lower.

        With multi-hop semantics: items in the same group are alternatives (OR).
        Once the group is satisfied, subsequent items are redundant.

        GT: [["doc_a", "doc_b"]] - single group with alternatives
        Retrieved: [doc_b, doc_a] - got lower-scored doc_b first
        Relevance: doc_a=2 (highly), doc_b=1 (somewhat)

        - doc_b (i=0): satisfies G0 (score=1), DCG += 1/1 = 1
        - doc_a (i=1): G0 already satisfied, redundant!

        DCG = 1
        IDCG = best from group is score=2 -> 3/1 = 3
        NDCG = 1/3 ≈ 0.333

        This reflects that we "wasted" the opportunity by satisfying
        the group with a lower-quality item first.
        """
        metric_input = MetricInput(
            retrieval_gt=[["doc_a", "doc_b"]],
            retrieved_ids=["doc_b", "doc_a"],
            relevance_scores={"doc_a": 2, "doc_b": 1},
        )
        result = retrieval_ndcg.__wrapped__(metric_input)
        assert result < 1.0
        assert result == pytest.approx(1 / 3, rel=1e-4)

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
        """Test with multiple ground truth groups (AND/OR structure).

        GT: [[doc_a, doc_b], [doc_c]] = (a OR b) AND c
        Retrieved: [doc_a, doc_c, doc_b]
        Relevance: a=2, b=1, c=2

        With new semantics:
        - doc_a (i=0): satisfies G0 (best item score=2), DCG += 3/1 = 3
        - doc_c (i=1): satisfies G1 (score=2), DCG += 3/1.585 = 1.893
        - doc_b (i=2): G0 already satisfied, no contribution (redundant)

        DCG = 4.893
        IDCG = best from each group [2, 2] -> 3/1 + 3/1.585 = 4.893
        NDCG = 1.0 (perfect - we got best items from each group)
        """
        metric_input = MetricInput(
            retrieval_gt=[["doc_a", "doc_b"], ["doc_c"]],  # (a OR b) AND c
            retrieved_ids=["doc_a", "doc_c", "doc_b"],
            relevance_scores={"doc_a": 2, "doc_b": 1, "doc_c": 2},
        )
        result = retrieval_ndcg.__wrapped__(metric_input)
        # Perfect retrieval: best items from each group at optimal positions
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_graded_ndcg_multiple_groups_suboptimal(self):
        """Test suboptimal retrieval with AND/OR structure.

        GT: [[doc_a, doc_b], [doc_c]] = (a OR b) AND c
        Retrieved: [doc_b, doc_c]  (got lower-scored doc_b instead of doc_a)
        Relevance: a=2, b=1, c=2

        - doc_b (i=0): satisfies G0 (but score=1, not best), DCG += 1/1 = 1
        - doc_c (i=1): satisfies G1 (score=2), DCG += 3/1.585 = 1.893

        DCG = 2.893
        IDCG = best from each group [2, 2] -> 3/1 + 3/1.585 = 4.893
        NDCG = 2.893 / 4.893 ≈ 0.591
        """
        metric_input = MetricInput(
            retrieval_gt=[["doc_a", "doc_b"], ["doc_c"]],
            retrieved_ids=["doc_b", "doc_c"],
            relevance_scores={"doc_a": 2, "doc_b": 1, "doc_c": 2},
        )
        result = retrieval_ndcg.__wrapped__(metric_input)
        assert result == pytest.approx(0.5913, rel=1e-3)

    def test_graded_ndcg_partial_group_satisfaction(self):
        """Test when not all groups are satisfied.

        GT: [[doc_a], [doc_b], [doc_c]] = a AND b AND c (3-hop)
        Retrieved: [doc_a, doc_b]  (missing doc_c!)
        Relevance: all score=1

        - doc_a (i=0): satisfies G0, DCG += 1/1 = 1
        - doc_b (i=1): satisfies G1, DCG += 1/1.585 = 0.631

        DCG = 1.631
        IDCG = [1, 1, 1] -> 1 + 0.631 + 0.5 = 2.131
        NDCG = 1.631 / 2.131 ≈ 0.765
        """
        metric_input = MetricInput(
            retrieval_gt=[["doc_a"], ["doc_b"], ["doc_c"]],
            retrieved_ids=["doc_a", "doc_b"],
            relevance_scores={"doc_a": 1, "doc_b": 1, "doc_c": 1},
        )
        result = retrieval_ndcg.__wrapped__(metric_input)
        # Only 2 of 3 groups satisfied
        assert result == pytest.approx(0.7654, rel=1e-3)
