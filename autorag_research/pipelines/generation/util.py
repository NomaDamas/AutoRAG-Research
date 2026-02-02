def aggregate_token_usage(results: list[dict]) -> tuple[int, int, int, int]:
    """Aggregate token usage from generation results.

    Args:
        results: List of generation result dicts with token_usage and execution_time.

    Returns:
        Tuple of (prompt_tokens, completion_tokens, embedding_tokens, execution_time_ms).
    """
    prompt_tokens = 0
    completion_tokens = 0
    embedding_tokens = 0
    execution_time_ms = 0

    for result in results:
        if result["token_usage"]:
            prompt_tokens += result["token_usage"].get("prompt_tokens", 0)
            completion_tokens += result["token_usage"].get("completion_tokens", 0)
            embedding_tokens += result["token_usage"].get("embedding_tokens", 0)
        execution_time_ms += result["execution_time"]

    return prompt_tokens, completion_tokens, embedding_tokens, execution_time_ms
