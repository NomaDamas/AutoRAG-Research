import importlib
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseLanguageModel

import evaluate
import nltk
import numpy as np
import pandas as pd
from rouge_score import tokenizers
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics.bleu import BLEU

from autorag_research.config import BaseGenerationMetricConfig
from autorag_research.evaluation.metrics.util import calculate_cosine_similarity, metric_loop
from autorag_research.exceptions import EmbeddingError
from autorag_research.injection import with_embedding, with_llm
from autorag_research.schema import MetricInput
from autorag_research.util import convert_inputs_to_list, truncate_texts, unpack_and_run

logger = logging.getLogger("AutoRAG-Research")

RAGAS_RESPONSE_RELEVANCE_INSTRUCTION = """Generate a question for the given answer and Identify if answer is noncommittal. Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers"""

DEFAULT_RESPONSE_RELEVANCY_PROMPT = """Generate a question for the given answer and identify if the answer is noncommittal.

Use this exact instruction:
{instruction}

Return a JSON object:
{{
  "question": "<generated question>",
  "noncommittal": 0 or 1
}}

Example input:
Albert Einstein was born in Germany.
Example output:
{{"question":"Where was Albert Einstein born?","noncommittal":0}}

Example input:
I don't know about the  groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022.
Example output:
{{"question":"What was the groundbreaking feature of the smartphone invented in 2023?","noncommittal":1}}

Input:
{response}
"""

_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
DEFAULT_BARTSCORE_CHECKPOINT = "facebook/bart-large-cnn"
DEFAULT_BARTSCORE_BATCH_SIZE = 4
DEFAULT_BARTSCORE_DEVICE = "auto"
DEFAULT_BARTSCORE_MAX_LENGTH = 1024
BARTSCORE_DEPENDENCY_MESSAGE = (
    "BARTScore requires the optional `torch` and `transformers` dependencies. "
    "Install them with `pip install 'autorag-research[gpu]'` or run "
    "`uv sync --all-extras --all-groups` in a development checkout."
)


def _extract_llm_text(response: Any) -> str:
    """Extract text from LLM response object."""
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)


def _parse_noncommittal(value: Any) -> int:
    """Convert noncommittal value to 0/1."""
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes"})
    return int(bool(value))


def _parse_response_relevancy_output(text: str) -> tuple[str, int]:
    """Parse question/noncommittal JSON from LLM output."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate).strip()
        candidate = re.sub(r"\s*```$", "", candidate).strip()

    payload: dict[str, Any] = {}
    for raw in [candidate]:
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                payload = loaded
                break
        except json.JSONDecodeError:
            continue

    if not payload:
        match = _JSON_BLOCK_PATTERN.search(candidate)
        if match:
            try:
                loaded = json.loads(match.group(0))
                if isinstance(loaded, dict):
                    payload = loaded
            except json.JSONDecodeError:
                pass

    question = str(payload.get("question", "")).strip()
    noncommittal = _parse_noncommittal(payload.get("noncommittal", 0))
    return question, noncommittal


def _calculate_response_relevancy_score(
    query: str,
    generated_questions: list[str],
    noncommittal_flags: list[int],
    embedding_model: "Embeddings",
) -> float:
    """RAGAS response relevancy core logic."""
    if all(question == "" for question in generated_questions):
        logger.warning("Invalid JSON response. Expected dictionary with key 'question'")
        return float("nan")

    query_vector = np.asarray(embedding_model.embed_query(query)).reshape(1, -1)
    generated_vectors = np.asarray(embedding_model.embed_documents(generated_questions)).reshape(
        len(generated_questions), -1
    )
    norm = np.linalg.norm(generated_vectors, axis=1) * np.linalg.norm(query_vector, axis=1)
    cosine_sim = np.dot(generated_vectors, query_vector.T).reshape(-1) / norm
    all_noncommittal = np.all(noncommittal_flags)
    return float(cosine_sim.mean() * int(not all_noncommittal))


@convert_inputs_to_list
def huggingface_evaluate(instance: Any, key: str, metric_inputs: list[MetricInput], **kwargs: Any) -> list[float]:
    """Compute huggingface evaluate metric.

    Args:
        instance: The instance of huggingface evaluates metric.
        key: The key to retrieve result score from huggingface evaluate result.
        metric_inputs: A list of MetricInput schema.
        **kwargs: The additional arguments for metric function.

    Returns:
        The list of scores.
    """

    def compute_score(gt: list[str], pred: str) -> float:
        return max([instance.compute(predictions=[pred], references=[x], **kwargs)[key] for x in gt])

    result = [compute_score(x.generation_gt, x.generated_texts) for x in metric_inputs]  # ty: ignore
    return result


def _resolve_bartscore_device(device: str) -> str:
    """Resolve a concrete torch device string for BARTScore."""
    if device != DEFAULT_BARTSCORE_DEVICE:
        return device

    torch = _import_bartscore_torch()

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _import_bartscore_torch() -> Any:
    """Import torch with a metric-specific dependency error message."""
    try:
        return importlib.import_module("torch")
    except ImportError as e:
        raise ImportError(BARTSCORE_DEPENDENCY_MESSAGE) from e


def _import_bartscore_runtime() -> tuple[Any, Any, Any]:
    """Import BARTScore runtime dependencies with a guided error message."""
    torch = _import_bartscore_torch()
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as e:
        raise ImportError(BARTSCORE_DEPENDENCY_MESSAGE) from e
    return torch, transformers.BartForConditionalGeneration, transformers.BartTokenizer


class _BartScoreBackend:
    """Thin wrapper around a pretrained BART conditional generation model."""

    def __init__(self, checkpoint: str, device: str, max_length: int) -> None:
        torch, BartForConditionalGeneration, BartTokenizer = _import_bartscore_runtime()

        self._torch = torch
        self._device = device
        self._max_length = max_length
        self._tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self._model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self._model.eval()
        self._model.to(device)
        self._loss_fct = torch.nn.NLLLoss(reduction="none", ignore_index=self._model.config.pad_token_id)
        self._log_softmax = torch.nn.LogSoftmax(dim=1)

    def score(self, src_texts: list[str], tgt_texts: list[str], batch_size: int) -> list[float]:
        """Compute average token log-likelihood scores for paired source/target texts."""
        scores: list[float] = []

        for batch_start in range(0, len(src_texts), batch_size):
            batch_src = src_texts[batch_start : batch_start + batch_size]
            batch_tgt = tgt_texts[batch_start : batch_start + batch_size]
            with self._torch.no_grad():
                encoded_src = self._tokenizer(
                    batch_src,
                    max_length=self._max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                encoded_tgt = self._tokenizer(
                    batch_tgt,
                    max_length=self._max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                src_tokens = encoded_src["input_ids"].to(self._device)
                src_mask = encoded_src["attention_mask"].to(self._device)
                tgt_tokens = encoded_tgt["input_ids"].to(self._device)
                tgt_mask = encoded_tgt["attention_mask"].to(self._device)
                tgt_len = tgt_mask.sum(dim=1).clamp_min(1)

                outputs = self._model(input_ids=src_tokens, attention_mask=src_mask, labels=tgt_tokens)
                logits = outputs.logits.view(-1, self._model.config.vocab_size)
                loss = self._loss_fct(self._log_softmax(logits), tgt_tokens.view(-1))
                loss = loss.view(tgt_tokens.shape[0], -1)
                loss = loss.sum(dim=1) / tgt_len
                scores.extend((-loss).tolist())

        return scores


@lru_cache(maxsize=4)
def _get_bartscore_backend(checkpoint: str, device: str, max_length: int) -> _BartScoreBackend:
    """Cache BARTScore model loading by checkpoint/device/length."""
    resolved_device = _resolve_bartscore_device(device)
    return _BartScoreBackend(checkpoint=checkpoint, device=resolved_device, max_length=max_length)


def _score_bartscore_pairs(
    src_texts: list[str],
    tgt_texts: list[str],
    checkpoint: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> list[float]:
    """Score aligned source/target text pairs with the cached BART backend."""
    backend = _get_bartscore_backend(checkpoint=checkpoint, device=device, max_length=max_length)
    return backend.score(src_texts=src_texts, tgt_texts=tgt_texts, batch_size=batch_size)


def _join_retrieved_contents(retrieved_contents: list[str]) -> str:
    """Combine retrieved passages into the single conditioning context used by BARTScore."""
    return "\n\n".join(content.strip() for content in retrieved_contents)


def _score_generation_references_with_bartscore(
    metric_inputs: list[MetricInput],
    *,
    source_from_reference: bool,
    checkpoint: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> list[float]:
    """Compute best-reference BARTScore in either reference→answer or answer→reference direction."""
    owner_indices: list[int] = []
    src_texts: list[str] = []
    tgt_texts: list[str] = []

    for index, metric_input in enumerate(metric_inputs):
        generated_text = cast(str, metric_input.generated_texts)
        references = cast(list[str], metric_input.generation_gt)
        for reference in references:
            owner_indices.append(index)
            if source_from_reference:
                src_texts.append(reference)
                tgt_texts.append(generated_text)
            else:
                src_texts.append(generated_text)
                tgt_texts.append(reference)

    scores = _score_bartscore_pairs(
        src_texts=src_texts,
        tgt_texts=tgt_texts,
        checkpoint=checkpoint,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    grouped_scores: list[list[float]] = [[] for _ in metric_inputs]
    for owner_index, score in zip(owner_indices, scores, strict=True):
        grouped_scores[owner_index].append(score)
    return [max(example_scores) for example_scores in grouped_scores]


@metric_loop(fields_to_check=["generation_gt", "generated_texts"])
def bleu(
    metric_inputs: list[MetricInput],
    tokenize: str | None = None,
    smooth_method: str = "exp",
    smooth_value: float | None = None,
    max_ngram_order: int = 4,
    trg_lang: str = "",
    effective_order: bool = True,
    **kwargs: Any,
) -> list[float]:
    """Computes the BLEU metric given pred and ground-truth.

    Args:
        metric_inputs: A list of MetricInput schema (Required Field -> "generation_gt", "generated_texts").
        tokenize: The tokenizer to use. If None, defaults to language-specific tokenizers with '13a' as the
            fallback default. check https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/bleu.py
        smooth_method: The smoothing method to use ('floor', 'add-k', 'exp' or 'none').
        smooth_value: The smoothing value for `floor` and `add-k` methods. `None` falls back to default value.
        max_ngram_order: If given, it overrides the maximum n-gram order (default: 4) when computing precisions.
        trg_lang: An optional language code to raise potential tokenizer warnings.
        effective_order: If `True`, stop including n-gram orders for which precision is 0. This should be
            `True`, if sentence-level BLEU will be computed.
        **kwargs: Additional arguments.

    Returns:
        A list of BLEU scores.
    """
    bleu_instance = BLEU(
        tokenize=tokenize,
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        max_ngram_order=max_ngram_order,
        trg_lang=trg_lang,
        effective_order=effective_order,
        **kwargs,
    )

    result = [bleu_instance.sentence_score(x.generated_texts, x.generation_gt).score for x in metric_inputs]  # ty: ignore
    return result


@metric_loop(fields_to_check=["generation_gt", "generated_texts"])
def meteor(
    metric_inputs: list[MetricInput],
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> list[float]:
    """Compute meteor score for generation.

    Args:
        metric_inputs: A list of MetricInput schema (Required Field -> "generation_gt", "generated_texts").
        alpha: Parameter for controlling relative weights of precision and recall. Default is 0.9.
        beta: Parameter for controlling shape of penalty as a function of as a function of fragmentation.
            Default is 3.0.
        gamma: Relative weight assigned to fragmentation penalty. Default is 0.5.

    Returns:
        A list of computed metric scores.
    """
    nltk.download("punkt", quiet=True)
    meteor_instance = evaluate.load("meteor")
    result = huggingface_evaluate(
        meteor_instance,
        "meteor",
        metric_inputs,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    del meteor_instance
    return result


@metric_loop(fields_to_check=["generation_gt", "generated_texts"])
def rouge(
    metric_inputs: list[MetricInput],
    rouge_type: str | None = "rougeL",
    use_stemmer: bool = False,
    split_summaries: bool = False,
) -> list[float]:
    """Compute rouge score for generation.

    Args:
        metric_inputs: A list of MetricInput schema (Required Field -> "generation_gt", "generated_texts").
        rouge_type: A rouge type to use for evaluation. Default is 'RougeL'.
            Choose between rouge1, rouge2, rougeL, and rougeLSum.
            - rouge1: unigram (1-gram) based scoring.
            - rouge2: bigram (2-gram) based scoring.
            - rougeL: Longest Common Subsequence based scoring.
            - rougeLSum: splits text using "\n"
        use_stemmer: Bool indicating whether Porter stemmer should be used to
            strip word suffixes to improve matching. This arg is used in the
            DefaultTokenizer, but other tokenizers might or might not choose to
            use this. Default is False.
        split_summaries: Whether to add newlines between sentences for rougeLsum. Default is False.

    Returns:
        A list of computed metric scores.
    """
    rouge_instance = RougeScorer(
        rouge_types=[rouge_type],
        use_stemmer=use_stemmer,
        split_summaries=split_summaries,
        tokenizer=tokenizers.DefaultTokenizer(use_stemmer),
    )

    result = [
        rouge_instance.score_multi(metric_input.generation_gt, metric_input.generated_texts)[rouge_type].fmeasure
        for metric_input in metric_inputs
    ]
    del rouge_instance
    return result


@metric_loop(fields_to_check=["generation_gt", "generated_texts"])
@with_embedding()
def sem_score(
    metric_inputs: list[MetricInput],
    embedding_model: "Embeddings | str",
    truncate_length: int = 4096,
) -> list[float]:
    """Compute sem score between generation gt and pred with cosine similarity.

    Args:
        metric_inputs: A list of MetricInput schema (Required Field -> "generation_gt", "generated_texts").
        embedding_model: Embedding model to use for compute cosine similarity.
            Can be an Embeddings instance or a string config name (e.g., "openai-large").
        truncate_length: Maximum length of texts to embedding. Default is 4096.

    Returns:
        A list of computed metric scores.
    """
    from langchain_core.embeddings import Embeddings

    if not isinstance(embedding_model, Embeddings):
        raise EmbeddingError

    generations = [metric_input.generated_texts for metric_input in metric_inputs]
    generation_gt = [metric_input.generation_gt for metric_input in metric_inputs]

    # Truncate texts to fit embedding model limit (Use tiktoken)
    generations = truncate_texts(generations, max_tokens=truncate_length)  # ty: ignore
    generation_gt = [truncate_texts(gen_gt, max_tokens=truncate_length) for gen_gt in generation_gt]  # ty: ignore

    embedded_pred: list[list[float]] = embedding_model.embed_documents(generations)
    embedded_gt: list[list[float]] = unpack_and_run(
        generation_gt,
        embedding_model.embed_documents,
    )

    result = []
    for gt, pred in zip(embedded_gt, embedded_pred, strict=True):
        similarity_scores: list[float] = [calculate_cosine_similarity(x, pred) for x in gt]
        result.append(max(similarity_scores))

    return result


@metric_loop(fields_to_check=["generation_gt", "generated_texts"])
def bert_score(
    metric_inputs: list[MetricInput],
    lang: str = "en",
    batch: int = 128,
    n_threads: int | None = os.cpu_count(),
) -> list[float]:
    """Compute BERTScore metric for generation.

    Args:
        metric_inputs: A list of MetricInput schema (Required Field -> "generation_gt", "generated_texts").
        lang: Language code for the text. Default is "en".
        batch: Batch size for processing. Default is 128.
        n_threads: Number of threads to use. Default is the number of CPU cores.

    Returns:
        A list of BERTScore F1 scores.
    """
    generations = [metric_input.generated_texts for metric_input in metric_inputs]
    generation_gt = [metric_input.generation_gt for metric_input in metric_inputs]
    evaluator = evaluate.load("bertscore")

    df = pd.DataFrame({
        "reference": generation_gt,
        "prediction": generations,
        "lang": lang,
    })

    df = df.explode("reference", ignore_index=False)
    result = evaluator.compute(  # ty: ignore
        predictions=df["prediction"].tolist(),
        references=df["reference"].tolist(),
        lang=lang,
        nthreads=n_threads,
        batch_size=batch,
    )
    df["bert_score"] = result["f1"]  # ty: ignore

    del evaluator

    return df.groupby(level=0)["bert_score"].max().tolist()


@metric_loop(fields_to_check=["retrieved_contents", "generated_texts"])
def bart_score_faithfulness(
    metric_inputs: list[MetricInput],
    checkpoint: str = DEFAULT_BARTSCORE_CHECKPOINT,
    batch_size: int = DEFAULT_BARTSCORE_BATCH_SIZE,
    max_length: int = DEFAULT_BARTSCORE_MAX_LENGTH,
    device: str = DEFAULT_BARTSCORE_DEVICE,
) -> list[float]:
    """Compute BARTScore faithfulness using retrieved context → generated answer."""
    src_texts = [
        _join_retrieved_contents(cast(list[str], metric_input.retrieved_contents)) for metric_input in metric_inputs
    ]
    tgt_texts = [cast(str, metric_input.generated_texts) for metric_input in metric_inputs]
    return _score_bartscore_pairs(
        src_texts=src_texts,
        tgt_texts=tgt_texts,
        checkpoint=checkpoint,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )


@metric_loop(fields_to_check=["generation_gt", "generated_texts"])
def bart_score_precision(
    metric_inputs: list[MetricInput],
    checkpoint: str = DEFAULT_BARTSCORE_CHECKPOINT,
    batch_size: int = DEFAULT_BARTSCORE_BATCH_SIZE,
    max_length: int = DEFAULT_BARTSCORE_MAX_LENGTH,
    device: str = DEFAULT_BARTSCORE_DEVICE,
) -> list[float]:
    """Compute BARTScore precision using reference → generated answer."""
    return _score_generation_references_with_bartscore(
        metric_inputs,
        source_from_reference=True,
        checkpoint=checkpoint,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )


@metric_loop(fields_to_check=["generation_gt", "generated_texts"])
def bart_score_recall(
    metric_inputs: list[MetricInput],
    checkpoint: str = DEFAULT_BARTSCORE_CHECKPOINT,
    batch_size: int = DEFAULT_BARTSCORE_BATCH_SIZE,
    max_length: int = DEFAULT_BARTSCORE_MAX_LENGTH,
    device: str = DEFAULT_BARTSCORE_DEVICE,
) -> list[float]:
    """Compute BARTScore recall using generated answer → reference."""
    return _score_generation_references_with_bartscore(
        metric_inputs,
        source_from_reference=False,
        checkpoint=checkpoint,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )


@metric_loop(fields_to_check=["generation_gt", "generated_texts"])
def bart_score_f1(
    metric_inputs: list[MetricInput],
    checkpoint: str = DEFAULT_BARTSCORE_CHECKPOINT,
    batch_size: int = DEFAULT_BARTSCORE_BATCH_SIZE,
    max_length: int = DEFAULT_BARTSCORE_MAX_LENGTH,
    device: str = DEFAULT_BARTSCORE_DEVICE,
) -> list[float]:
    """Compute the arithmetic mean of BARTScore precision and recall."""
    precision_scores = bart_score_precision(
        metric_inputs,
        checkpoint=checkpoint,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    recall_scores = bart_score_recall(
        metric_inputs,
        checkpoint=checkpoint,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    return [
        (precision_score + recall_score) / 2
        for precision_score, recall_score in zip(precision_scores, recall_scores, strict=True)
    ]


@metric_loop(fields_to_check=["query", "generated_texts"])
@with_embedding()
@with_llm()
def response_relevancy(
    metric_inputs: list[MetricInput],
    llm: "BaseLanguageModel | str",
    embedding_model: "Embeddings | str",
    strictness: int = 3,
    prompt_template: str = DEFAULT_RESPONSE_RELEVANCY_PROMPT,
) -> list[float]:
    """RAGAS-style response relevancy metric without ragas dependency."""
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseLanguageModel

    if strictness < 1:
        msg = "strictness must be >= 1"
        raise ValueError(msg)

    if not isinstance(llm, BaseLanguageModel):
        msg = "llm must be a BaseLanguageModel instance after with_llm injection"
        raise TypeError(msg)

    if not isinstance(embedding_model, Embeddings):
        raise EmbeddingError

    scores = []
    for metric_input in metric_inputs:
        query = metric_input.query
        if query is None:
            msg = "query is required for response_relevancy"
            raise ValueError(msg)

        prompt = prompt_template.format(
            instruction=RAGAS_RESPONSE_RELEVANCE_INSTRUCTION,
            response=metric_input.generated_texts,
        )
        questions: list[str] = []
        noncommittal_flags: list[int] = []
        for _ in range(strictness):
            response = llm.invoke(prompt)
            question, noncommittal = _parse_response_relevancy_output(_extract_llm_text(response))
            questions.append(question)
            noncommittal_flags.append(noncommittal)

        scores.append(
            _calculate_response_relevancy_score(
                query=query,
                generated_questions=questions,
                noncommittal_flags=noncommittal_flags,
                embedding_model=embedding_model,
            )
        )
    return scores


# Metric Configurations
@dataclass
class BleuConfig(BaseGenerationMetricConfig):
    """Configuration for BLEU metric.

    Attributes:
        tokenize: The tokenizer to use. If None, defaults to language-specific tokenizers.
        smooth_method: The smoothing method ('floor', 'add-k', 'exp' or 'none').
        smooth_value: The smoothing value for 'floor' and 'add-k' methods.
        max_ngram_order: Maximum n-gram order when computing precisions.
        effective_order: Stop including n-gram orders for which precision is 0.
    """

    tokenize: str | None = None
    smooth_method: str = "exp"
    smooth_value: float | None = None
    max_ngram_order: int = 4
    effective_order: bool = True

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return bleu

    def get_metric_kwargs(self) -> dict[str, Any]:
        """Return kwargs for the metric function."""
        return {
            "tokenize": self.tokenize,
            "smooth_method": self.smooth_method,
            "smooth_value": self.smooth_value,
            "max_ngram_order": self.max_ngram_order,
            "effective_order": self.effective_order,
        }


@dataclass
class MeteorConfig(BaseGenerationMetricConfig):
    """Configuration for METEOR metric.

    Attributes:
        alpha: Parameter for controlling relative weights of precision and recall.
        beta: Parameter for controlling shape of penalty as a function of fragmentation.
        gamma: Relative weight assigned to fragmentation penalty.
    """

    alpha: float = 0.9
    beta: float = 3.0
    gamma: float = 0.5

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return meteor

    def get_metric_kwargs(self) -> dict[str, Any]:
        """Return kwargs for the metric function."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }


@dataclass
class RougeConfig(BaseGenerationMetricConfig):
    """Configuration for ROUGE metric.

    Attributes:
        rouge_type: Rouge type to use ('rouge1', 'rouge2', 'rougeL', 'rougeLSum').
        use_stemmer: Whether to use Porter stemmer for word suffix stripping.
        split_summaries: Whether to add newlines between sentences for rougeLsum.
    """

    rouge_type: str = "rougeL"
    use_stemmer: bool = False
    split_summaries: bool = False

    def get_metric_name(self) -> str:
        """Return the metric name."""
        return f"rouge_{self.rouge_type}"

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return rouge

    def get_metric_kwargs(self) -> dict[str, Any]:
        """Return kwargs for the metric function."""
        return {
            "rouge_type": self.rouge_type,
            "use_stemmer": self.use_stemmer,
            "split_summaries": self.split_summaries,
        }


@dataclass
class SemScoreConfig(BaseGenerationMetricConfig):
    """Configuration for SemScore (semantic similarity) metric.

    Attributes:
        embedding_model: Embedding model config name (e.g., "openai-large") or Embeddings instance.
        truncate_length: Maximum length of texts to embed.
    """

    truncate_length: int = 4096
    embedding_model: "Embeddings | str" = "openai-large"

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return sem_score

    def get_metric_kwargs(self) -> dict[str, Any]:
        """Return kwargs for the metric function."""
        return {
            "embedding_model": self.embedding_model,
            "truncate_length": self.truncate_length,
        }


@dataclass
class ResponseRelevancyConfig(BaseGenerationMetricConfig):
    """Configuration for RAGAS-style response relevancy metric."""

    strictness: int = 3
    llm: "BaseLanguageModel | str" = "openai-gpt5-mini"
    embedding_model: "Embeddings | str" = "openai-large"
    prompt_template: str = DEFAULT_RESPONSE_RELEVANCY_PROMPT

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return response_relevancy

    def get_metric_kwargs(self) -> dict[str, Any]:
        """Return kwargs for the metric function."""
        return {
            "strictness": self.strictness,
            "llm": self.llm,
            "embedding_model": self.embedding_model,
            "prompt_template": self.prompt_template,
        }


@dataclass
class BertScoreConfig(BaseGenerationMetricConfig):
    """Configuration for BERTScore metric.

    Attributes:
        lang: Language code for the text.
        batch: Batch size for processing.
        n_threads: Number of threads to use.
    """

    lang: str = "en"
    batch: int = 128
    n_threads: int | None = None

    def __post_init__(self) -> None:
        """Set default n_threads if not provided."""
        if self.n_threads is None:
            self.n_threads = os.cpu_count()

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return bert_score

    def get_metric_kwargs(self) -> dict[str, Any]:
        """Return kwargs for the metric function."""
        return {
            "lang": self.lang,
            "batch": self.batch,
            "n_threads": self.n_threads,
        }


@dataclass
class _BaseBartScoreConfig(BaseGenerationMetricConfig):
    """Shared configuration for BARTScore variants."""

    checkpoint: str = DEFAULT_BARTSCORE_CHECKPOINT
    batch_size: int = DEFAULT_BARTSCORE_BATCH_SIZE
    device: str = DEFAULT_BARTSCORE_DEVICE
    max_length: int = DEFAULT_BARTSCORE_MAX_LENGTH

    def get_metric_kwargs(self) -> dict[str, Any]:
        """Return kwargs shared across BARTScore variants."""
        return {
            "checkpoint": self.checkpoint,
            "batch_size": self.batch_size,
            "device": self.device,
            "max_length": self.max_length,
        }


@dataclass
class BartScoreFaithfulnessConfig(_BaseBartScoreConfig):
    """Configuration for BARTScore faithfulness (context → answer)."""

    def get_metric_name(self) -> str:
        """Return the metric name."""
        return "bart_score_faithfulness"

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return bart_score_faithfulness


@dataclass
class BartScorePrecisionConfig(_BaseBartScoreConfig):
    """Configuration for BARTScore precision (reference → answer)."""

    def get_metric_name(self) -> str:
        """Return the metric name."""
        return "bart_score_precision"

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return bart_score_precision


@dataclass
class BartScoreRecallConfig(_BaseBartScoreConfig):
    """Configuration for BARTScore recall (answer → reference)."""

    def get_metric_name(self) -> str:
        """Return the metric name."""
        return "bart_score_recall"

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return bart_score_recall


@dataclass
class BartScoreF1Config(_BaseBartScoreConfig):
    """Configuration for BARTScore F1 (mean of precision and recall)."""

    def get_metric_name(self) -> str:
        """Return the metric name."""
        return "bart_score_f1"

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return bart_score_f1
