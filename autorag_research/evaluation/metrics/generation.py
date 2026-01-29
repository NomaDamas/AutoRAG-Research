import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import evaluate
import nltk
import pandas as pd
from llama_index.core.embeddings import BaseEmbedding
from rouge_score import tokenizers
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics.bleu import BLEU

from autorag_research.config import BaseGenerationMetricConfig
from autorag_research.evaluation.metrics.util import calculate_cosine_similarity, metric_loop
from autorag_research.exceptions import EmbeddingError
from autorag_research.injection import with_embedding
from autorag_research.schema import MetricInput
from autorag_research.util import convert_inputs_to_list, truncate_texts, unpack_and_run


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
    embedding_model: BaseEmbedding | str,
    truncate_length: int = 4096,
) -> list[float]:
    """Compute sem score between generation gt and pred with cosine similarity.

    Args:
        metric_inputs: A list of MetricInput schema (Required Field -> "generation_gt", "generated_texts").
        embedding_model: Embedding model to use for compute cosine similarity.
            Can be a BaseEmbedding instance or a string config name (e.g., "openai-large").
        truncate_length: Maximum length of texts to embedding. Default is 4096.

    Returns:
        A list of computed metric scores.
    """
    if not isinstance(embedding_model, BaseEmbedding):
        raise EmbeddingError

    generations = [metric_input.generated_texts for metric_input in metric_inputs]
    generation_gt = [metric_input.generation_gt for metric_input in metric_inputs]

    # Truncate texts to fit embedding model limit (Use tiktoken)
    generations = truncate_texts(generations, max_tokens=truncate_length)  # ty: ignore
    generation_gt = [truncate_texts(gen_gt, max_tokens=truncate_length) for gen_gt in generation_gt]  # ty: ignore

    embedded_pred: list[list[float]] = embedding_model.get_text_embedding_batch(generations, show_progress=True)
    embedded_gt: list[list[float]] = unpack_and_run(
        generation_gt,
        embedding_model.get_text_embedding_batch,
        show_progress=True,
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
        embedding_model: Embedding model config name (e.g., "openai-large") or BaseEmbedding instance.
        truncate_length: Maximum length of texts to embed.
    """

    truncate_length: int = 4096
    embedding_model: BaseEmbedding | str = "openai-large"

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
