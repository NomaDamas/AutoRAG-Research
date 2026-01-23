"""Metric configurations - available metrics for listing."""

# Available metrics for listing (metadata only, actual config in YAML)
AVAILABLE_METRICS = {
    "recall": "Recall - Retrieval recall",
    "precision": "Precision - Retrieval precision",
    "ndcg": "NDCG - Normalized Discounted Cumulative Gain",
    "mrr": "MRR - Mean Reciprocal Rank",
    "f1": "F1 Score - Token-level F1 for generation",
    "rouge": "ROUGE-L - Longest common subsequence",
}
