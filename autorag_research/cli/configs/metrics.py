"""Metric configurations - available metrics for listing."""

# Available metrics for listing (metadata only, actual config in YAML)
AVAILABLE_METRICS = {
    "recall": "Recall@10 - Retrieval recall at k=10",
    "recall_5": "Recall@5 - Retrieval recall at k=5",
    "precision": "Precision@10 - Retrieval precision at k=10",
    "ndcg": "NDCG@10 - Normalized Discounted Cumulative Gain",
    "mrr": "MRR@10 - Mean Reciprocal Rank",
    "f1": "F1 Score - Token-level F1 for generation",
    "rouge": "ROUGE-L - Longest common subsequence",
}
