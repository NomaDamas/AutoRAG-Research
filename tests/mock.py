from random import random
from typing import Any


def mock_get_text_embedding_batch(
    self,
    texts: list[str],
    show_progress: bool = False,
    **kwargs: Any,
) -> list[list[float]]:
    return [[random() for _ in range(1536)] for _ in range(len(texts))]
