"""Pipeline configurations - available pipelines for listing."""

# Available pipelines for listing (metadata only, actual config in YAML)
AVAILABLE_PIPELINES = {
    "bm25_baseline": "BM25 retrieval with Pyserini",
    "vector_search": "Dense vector similarity search",
    "hybrid_search": "Hybrid BM25 + Vector search",
    "naive_rag": "Simple retrieve-then-generate RAG",
}
