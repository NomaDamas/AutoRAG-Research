"""UniEval scorer helpers."""

import logging
from functools import lru_cache
from typing import Protocol

logger = logging.getLogger("AutoRAG-Research")


class UniEvalScorer(Protocol):
    """Protocol for UniEval scorers used by generation metrics."""

    def score(self, inputs: list[str], batch_size: int = 8) -> list[float]:
        """Return one score per prompt."""


class HuggingFaceUniEvalScorer:
    """Thin wrapper around the official UniEval yes/no scoring logic."""

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 1024,
        device: str = "cpu",
        cache_dir: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised via higher-level patch tests
            msg = (
                "UniEval requires torch, transformers, and sentencepiece-compatible tokenizer support. "
                "Install the project extras with `uv sync --all-extras --all-groups`."
            )
            raise ImportError(msg) from exc

        self._torch = torch
        self.device = device
        self.max_length = max_length
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            config=self.config,
            cache_dir=cache_dir,
        )
        self.model.eval()
        self.model.to(device)

        self.pos_id = self.tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
        self.neg_id = self.tokenizer("No", add_special_tokens=False)["input_ids"][0]

    def score(self, inputs: list[str], batch_size: int = 8) -> list[float]:
        """Compute official-style Yes/(Yes+No) scores for Bool-QA prompts."""
        if not inputs:
            return []

        target_tokens = ["No"] * len(inputs)
        scores: list[float] = []

        for start in range(0, len(inputs), batch_size):
            batch_inputs = inputs[start : start + batch_size]
            batch_targets = target_tokens[start : start + batch_size]
            with self._torch.no_grad():
                encoded_src = self.tokenizer(
                    batch_inputs,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                encoded_tgt = self.tokenizer(
                    batch_targets,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                decoder_tokens = encoded_tgt["input_ids"].to(self.device)[:, 0].unsqueeze(-1)

                output = self.model(
                    input_ids=encoded_src["input_ids"].to(self.device),
                    attention_mask=encoded_src["attention_mask"].to(self.device),
                    labels=decoder_tokens,
                )
                logits = output.logits.reshape(-1, self.model.config.vocab_size)
                probabilities = self._torch.softmax(logits, dim=1)
                yes_scores = probabilities[:, self.pos_id]
                no_scores = probabilities[:, self.neg_id]
                batch_scores = yes_scores / (yes_scores + no_scores)
                scores.extend(float(score) for score in batch_scores.tolist())

        return scores


@lru_cache(maxsize=8)
def get_unieval_scorer(
    model_name_or_path: str = "MingZhong/unieval-sum",
    max_length: int = 1024,
    device: str = "cpu",
    cache_dir: str | None = None,
) -> UniEvalScorer:
    """Return a cached UniEval scorer instance."""
    logger.debug(
        "Loading UniEval scorer model_name_or_path=%s max_length=%s device=%s cache_dir=%s",
        model_name_or_path,
        max_length,
        device,
        cache_dir,
    )
    return HuggingFaceUniEvalScorer(
        model_name_or_path=model_name_or_path,
        max_length=max_length,
        device=device,
        cache_dir=cache_dir,
    )
