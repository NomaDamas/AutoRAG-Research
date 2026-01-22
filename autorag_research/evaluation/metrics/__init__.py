"""Metrics module for AutoRAG-Research evaluation.

This module provides evaluation metrics for generation and retrieval tasks.
"""

from autorag_research.evaluation.metrics.generation import (
    BertScoreConfig,
    BleuConfig,
    MeteorConfig,
    RougeConfig,
    SemScoreConfig,
    bert_score,
    bleu,
    huggingface_evaluate,
    meteor,
    rouge,
    sem_score,
)
from autorag_research.evaluation.metrics.retrieval import (
    F1Config,
    MAPConfig,
    MRRConfig,
    NDCGConfig,
    PrecisionConfig,
    RecallConfig,
    retrieval_f1,
    retrieval_map,
    retrieval_mrr,
    retrieval_ndcg,
    retrieval_precision,
    retrieval_recall,
)
from autorag_research.evaluation.metrics.util import (
    calculate_cosine_similarity,
    calculate_inner_product,
    calculate_l2_distance,
    metric,
    metric_loop,
)

__all__ = [
    "BertScoreConfig",
    "BleuConfig",
    "F1Config",
    "MAPConfig",
    "MRRConfig",
    "MeteorConfig",
    "NDCGConfig",
    "PrecisionConfig",
    "RecallConfig",
    "RougeConfig",
    "SemScoreConfig",
    "bert_score",
    "bleu",
    "calculate_cosine_similarity",
    "calculate_inner_product",
    "calculate_l2_distance",
    "huggingface_evaluate",
    "meteor",
    "metric",
    "metric_loop",
    "retrieval_f1",
    "retrieval_map",
    "retrieval_mrr",
    "retrieval_ndcg",
    "retrieval_precision",
    "retrieval_recall",
    "rouge",
    "sem_score",
]
