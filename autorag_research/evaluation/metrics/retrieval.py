import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

from autorag_research.config import BaseRetrievalMetricConfig
from autorag_research.evaluation.metrics.util import metric
from autorag_research.schema import MetricInput


@metric(fields_to_check=["retrieval_gt", "retrieved_ids"])
def retrieval_f1(metric_input: MetricInput) -> float:
    """Compute f1 score for retrieval.

    Args:
        metric_input: The MetricInput schema for AutoRAG metric.

    Returns:
        The f1 score.
    """
    recall_score = retrieval_recall.__wrapped__(metric_input)
    precision_score = retrieval_precision.__wrapped__(metric_input)
    if recall_score + precision_score == 0:
        return 0
    else:
        return 2 * (recall_score * precision_score) / (recall_score + precision_score)


@metric(fields_to_check=["retrieval_gt", "retrieved_ids"])
def retrieval_recall(metric_input: MetricInput) -> float:
    """Compute recall score for retrieval.

    Args:
        metric_input: The MetricInput schema for AutoRAG metric.

    Returns:
        The recall score.
    """
    gt, pred = metric_input.retrieval_gt, metric_input.retrieved_ids
    if pred is None or gt is None:
        return 0.0

    gt_sets = [frozenset(g) for g in gt]
    pred_set = set(pred)
    hits = sum(any(pred_id in gt_set for pred_id in pred_set) for gt_set in gt_sets)
    recall = hits / len(gt) if len(gt) > 0 else 0.0
    return recall


@metric(fields_to_check=["retrieval_gt", "retrieved_ids"])
def retrieval_precision(metric_input: MetricInput) -> float:
    """Compute precision score for retrieval.

    Args:
        metric_input: The MetricInput schema for AutoRAG metric.

    Returns:
        The precision score.
    """
    gt, pred = metric_input.retrieval_gt, metric_input.retrieved_ids
    if pred is None or gt is None:
        return 0.0

    gt_sets = [frozenset(g) for g in gt]
    pred_set = set(pred)
    hits = sum(any(pred_id in gt_set for gt_set in gt_sets) for pred_id in pred_set)
    precision = hits / len(pred) if len(pred) > 0 else 0.0
    return precision


@metric(fields_to_check=["retrieval_gt", "retrieved_ids"])
def retrieval_ndcg(metric_input: MetricInput) -> float:  # noqa: C901
    """Compute NDCG for multi-hop retrieval with AND-OR group semantics.

    Ground truth structure: [[A, B], [C]] means (A OR B) AND C
    - Each inner list is an OR group (any item satisfies the group)
    - Outer list is AND (all groups must be satisfied for complete retrieval)

    A retrieved item contributes to DCG only when it's the FIRST to satisfy
    a previously unsatisfied group. Subsequent items from the same group
    don't add value (they're redundant for answering the query).

    Supports graded relevance when `metric_input.relevance_scores` is provided.
    Falls back to binary relevance (score=1) when relevance_scores is None.

    Args:
        metric_input: The MetricInput schema for AutoRAG metric.
            - retrieval_gt: 2D list of ground truth IDs (AND/OR structure)
            - retrieved_ids: list of retrieved IDs
            - relevance_scores: optional dict mapping doc_id -> graded relevance score
              (e.g., 0=not relevant, 1=somewhat relevant, 2=highly relevant)

    Returns:
        The NDCG score.

    Examples:
        GT: [[A, B], [C]] (need A-or-B AND C)
        Retrieved: [A, C] -> Perfect (both groups satisfied at top positions)
        Retrieved: [A, B] -> Partial (group 1 not satisfied, B is redundant)
        Retrieved: [C, A] -> Good but suboptimal ordering
    """
    gt, pred = metric_input.retrieval_gt, metric_input.retrieved_ids
    if pred is None or gt is None:
        return 0.0

    # Filter out empty groups
    valid_groups = [group for group in gt if group and group != [""]]
    if not valid_groups:
        return 0.0

    # Build item -> group indices mapping (item can belong to multiple groups)
    item_to_groups: dict[str, list[int]] = {}
    for group_idx, group in enumerate(valid_groups):
        for item in group:
            if item:  # Skip empty strings
                if item not in item_to_groups:
                    item_to_groups[item] = []
                item_to_groups[item].append(group_idx)

    # Get relevance scores (default to 1 for backward compatibility)
    gt_flat = set(itertools.chain.from_iterable(valid_groups))
    relevance_map = metric_input.relevance_scores or dict.fromkeys(gt_flat, 1)

    # DCG: Track satisfied groups, item contributes only for first group satisfaction
    satisfied_groups: set[int] = set()
    dcg = 0.0
    for i, doc_id in enumerate(pred):
        if doc_id in item_to_groups:
            new_groups = [g for g in item_to_groups[doc_id] if g not in satisfied_groups]
            if new_groups:
                # Item contributes for satisfying new group(s)
                satisfied_groups.update(new_groups)
                dcg += (2 ** relevance_map.get(doc_id, 0) - 1) / math.log2(i + 2)

    # IDCG: Best item from each group, sorted by score descending
    best_per_group = []
    for group in valid_groups:
        best_score = max((relevance_map.get(item, 0) for item in group if item), default=0)
        best_per_group.append(best_score)

    ideal_scores = sorted(best_per_group, reverse=True)
    idcg = sum((2**score - 1) / math.log2(i + 2) for i, score in enumerate(ideal_scores))

    return dcg / idcg if idcg > 0 else 0.0


@metric(fields_to_check=["retrieval_gt", "retrieved_ids"])
def retrieval_mrr(metric_input: MetricInput) -> float:
    """Compute MRR (Mean Reciprocal Rank) score for retrieval.

    Reciprocal Rank (RR) is the reciprocal of the rank of the first relevant item.
    Mean of RR in whole queries is MRR.

    Args:
        metric_input: The MetricInput schema for AutoRAG metric.

    Returns:
        The MRR score.
    """
    gt, pred = metric_input.retrieval_gt, metric_input.retrieved_ids
    if pred is None or gt is None:
        return 0.0

    # Flatten the ground truth list of lists into a single set of relevant documents
    gt_sets = [frozenset(g) for g in gt]

    rr_list = []
    for gt_set in gt_sets:
        for i, pred_id in enumerate(pred):
            if pred_id in gt_set:
                rr_list.append(1.0 / (i + 1))
                break
    return sum(rr_list) / len(gt_sets) if rr_list else 0.0


@metric(fields_to_check=["retrieval_gt", "retrieved_ids"])
def retrieval_map(metric_input: MetricInput) -> float:
    """Compute MAP (Mean Average Precision) score for retrieval.

    Mean Average Precision (MAP) is the mean of Average Precision (AP) for all queries.

    Args:
        metric_input: The MetricInput schema for AutoRAG metric.

    Returns:
        The MAP score.
    """
    gt, pred = metric_input.retrieval_gt, metric_input.retrieved_ids
    if pred is None or gt is None:
        return 0.0

    gt_sets = [frozenset(g) for g in gt]

    ap_list = []

    for gt_set in gt_sets:
        pred_hits = [1 if pred_id in gt_set else 0 for pred_id in pred]
        precision_list = [sum(pred_hits[: i + 1]) / (i + 1) for i, hit in enumerate(pred_hits) if hit == 1]
        ap_list.append(sum(precision_list) / len(precision_list) if precision_list else 0.0)

    return sum(ap_list) / len(gt_sets) if ap_list else 0.0


# Metric Configurations
@dataclass
class RecallConfig(BaseRetrievalMetricConfig):
    """Configuration for retrieval recall metric."""

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return retrieval_recall


@dataclass
class PrecisionConfig(BaseRetrievalMetricConfig):
    """Configuration for retrieval precision metric."""

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return retrieval_precision


@dataclass
class F1Config(BaseRetrievalMetricConfig):
    """Configuration for retrieval F1 metric."""

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return retrieval_f1


@dataclass
class NDCGConfig(BaseRetrievalMetricConfig):
    """Configuration for retrieval NDCG metric."""

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return retrieval_ndcg


@dataclass
class MRRConfig(BaseRetrievalMetricConfig):
    """Configuration for retrieval MRR metric."""

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return retrieval_mrr


@dataclass
class MAPConfig(BaseRetrievalMetricConfig):
    """Configuration for retrieval MAP metric."""

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return retrieval_map
