def make_id(*parts: str | int) -> str:
    """Generate ID by joining parts with underscore."""
    return "_".join(str(p) for p in parts)
