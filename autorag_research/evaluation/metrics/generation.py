import json
import logging
import os
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
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
from autorag_research.evaluation.metrics.unieval import UniEvalScorer, get_unieval_scorer
from autorag_research.evaluation.metrics.util import calculate_cosine_similarity, metric_loop
from autorag_research.exceptions import EmbeddingError
from autorag_research.injection import with_embedding, with_llm
from autorag_research.schema import MetricInput
from autorag_research.util import convert_inputs_to_list, normalize_string, truncate_texts, unpack_and_run

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
UNIEVAL_MODEL_NAME = "MingZhong/unieval-sum"
UNIEVAL_DIMENSIONS = ("coherence", "consistency", "fluency", "relevance")


def _get_normalized_tokens(text: str) -> list[str]:
    """Normalize text with SQuAD rules and split into tokens."""
    return normalize_string(text).split()


def _score_generation_against_references(
    prediction: str,
    references: list[str],
    scorer: Callable[[str, str], float],
) -> float:
    """Return the best score for a prediction across all references."""
    return max(scorer(prediction, reference) for reference in references)


def _compute_generation_reference_scores(
    metric_inputs: list[MetricInput],
    scorer: Callable[[str, str], float],
) -> list[float]:
    """Compute best-reference scores for generation metric inputs."""
    scores = []
    for metric_input in metric_inputs:
        generated_text = cast(str, metric_input.generated_texts)
        generation_gt = cast(list[str], metric_input.generation_gt)
        scores.append(_score_generation_against_references(generated_text, generation_gt, scorer))
    return scores


def _exact_match_score(prediction: str, reference: str) -> float:
    """Return binary exact match after SQuAD-style normalization."""
    return float(normalize_string(prediction) == normalize_string(reference))


def _token_f1_score(prediction: str, reference: str) -> float:
    """Compute SQuAD-style token F1 for one prediction/reference pair."""
    prediction_tokens = _get_normalized_tokens(prediction)
    reference_tokens = _get_normalized_tokens(reference)

    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return float((2 * precision * recall) / (precision + recall))


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


def _validate_unieval_dimension(dimension: str) -> str:
    """Validate and normalize UniEval dimension name."""
    normalized = dimension.strip().lower()
    if normalized not in UNIEVAL_DIMENSIONS:
        msg = f"Unsupported UniEval dimension: {dimension}"
        raise ValueError(msg)
    return normalized


def _split_unieval_sentences(text: str) -> list[str]:
    """Split text into sentences with a regex fallback when punkt data is unavailable."""
    try:
        sentences = [sentence.strip() for sentence in nltk.sent_tokenize(text) if sentence.strip()]
    except LookupError:
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    return sentences or [text.strip()]


def _build_unieval_prompt(
    *,
    dimension: str,
    generated_text: str,
    document: str | None = None,
    reference: str | None = None,
) -> str:
    """Build the Bool-QA prompt for a UniEval dimension.

    These strings intentionally mirror UniEval's published `add_question`
    summarization prompts, including the `</s>` separators.
    """
    if dimension == "fluency":
        return f"question: Is this a fluent paragraph? </s> paragraph: {generated_text}"
    if dimension == "coherence":
        return (
            "question: Is this a coherent summary to the document? "
            f"</s> summary: {generated_text} </s> document: {document}"
        )
    if dimension == "consistency":
        return (
            "question: Is this claim consistent with the document? "
            f"</s> claim: {generated_text} </s> document: {document}"
        )
    return (
        "question: Is this summary relevant to the reference? "
        f"</s> summary: {generated_text} </s> reference: {reference}"
    )


def _build_unieval_document(retrieved_contents: list[str]) -> str:
    """Join retrieved passages into the document field used by UniEval summarization prompts."""
    return " ".join(content.strip() for content in retrieved_contents)


def _prepare_unieval_references(references: list[str]) -> list[str]:
    """Normalize available references for UniEval relevance scoring."""
    return [reference.strip() for reference in references if reference.strip()]


def _prepare_unieval_prompts(
    metric_input: MetricInput,
    dimension: str,
    sentence_level_dimensions: set[str],
) -> list[str]:
    """Build UniEval prompts for a single metric input."""
    generated_text = cast(str, metric_input.generated_texts).strip()
    if dimension == "relevance":
        return [
            _build_unieval_prompt(
                dimension=dimension,
                generated_text=generated_text,
                reference=reference,
            )
            for reference in _prepare_unieval_references(cast(list[str], metric_input.generation_gt))
        ]

    document = None
    if dimension in {"coherence", "consistency"}:
        document = _build_unieval_document(cast(list[str], metric_input.retrieved_contents))

    text_units = (
        _split_unieval_sentences(generated_text) if dimension in sentence_level_dimensions else [generated_text]
    )
    return [
        _build_unieval_prompt(
            dimension=dimension,
            generated_text=text_unit,
            document=document,
        )
        for text_unit in text_units
    ]


def _aggregate_unieval_scores(dimension: str, scores: list[float]) -> float:
    """Aggregate prompt-level UniEval scores back to one sample-level score."""
    if dimension in {"fluency", "consistency"}:
        return float(sum(scores) / len(scores))
    if dimension == "relevance":
        return max(scores)
    return scores[0]


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

    def compute_score(prediction: str, reference: str) -> float:
        return instance.compute(predictions=[prediction], references=[reference], **kwargs)[key]

    return _compute_generation_reference_scores(metric_inputs, compute_score)


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
def exact_match(metric_inputs: list[MetricInput]) -> list[float]:
    """Compute SQuAD-style exact match for generation."""
    return _compute_generation_reference_scores(metric_inputs, _exact_match_score)


@metric_loop(fields_to_check=["generation_gt", "generated_texts"])
def token_f1(metric_inputs: list[MetricInput]) -> list[float]:
    """Compute SQuAD-style token F1 for generation."""
    return _compute_generation_reference_scores(metric_inputs, _token_f1_score)


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


@convert_inputs_to_list
def unieval(
    metric_inputs: list[MetricInput],
    dimension: str,
    model_name_or_path: str = UNIEVAL_MODEL_NAME,
    batch_size: int = 8,
    max_length: int = 1024,
    device: str = "cpu",
    cache_dir: str | None = None,
    scorer: UniEvalScorer | None = None,
) -> list[float | None]:
    """Compute one UniEval dimension using the shared summarization checkpoint.

    The implementation follows the official UniEval summarization Bool-QA contract:
    - fluency: generated text only, averaged per sentence
    - coherence: retrieved source context + generated text, scored per sample
    - consistency: retrieved source context + generated text, averaged per sentence
    - relevance: reference + generated text, scored per sample

    Consistency intentionally skips samples with missing retrieved context instead of
    silently falling back to hidden ground truth, keeping the metric faithful to the
    actual evidence available to the generation pipeline. Relevance keeps the official
    single-reference prompt contract, but evaluates every available reference and keeps
    the best score so multi-reference inputs stay order-independent.
    """
    normalized_dimension = _validate_unieval_dimension(dimension)
    field_map = {
        "fluency": ["generated_texts"],
        "coherence": ["generated_texts", "retrieved_contents"],
        "consistency": ["generated_texts", "retrieved_contents"],
        "relevance": ["generated_texts", "generation_gt"],
    }
    sentence_level_dimensions = {"fluency", "consistency"}

    prepared_inputs: list[str] = []
    prompt_counts: list[tuple[int, int]] = []
    results: list[float | None] = [None] * len(metric_inputs)
    for index, metric_input in enumerate(metric_inputs):
        if not metric_input.is_fields_notnone(field_map[normalized_dimension]):
            continue

        prompts = _prepare_unieval_prompts(metric_input, normalized_dimension, sentence_level_dimensions)
        if not prompts:
            continue

        prompt_count = len(prompts)
        prepared_inputs.extend(prompts)
        prompt_counts.append((index, prompt_count))

    if not prepared_inputs:
        return results

    effective_scorer = scorer or get_unieval_scorer(
        model_name_or_path=model_name_or_path,
        max_length=max_length,
        device=device,
        cache_dir=cache_dir,
    )

    prompt_scores = effective_scorer.score(prepared_inputs, batch_size=batch_size)
    if len(prompt_scores) != len(prepared_inputs):
        msg = (
            f"UniEval scorer returned {len(prompt_scores)} scores for {len(prepared_inputs)} prompts "
            f"for dimension '{normalized_dimension}'"
        )
        raise ValueError(msg)
    prompt_offset = 0
    for index, count in prompt_counts:
        current_scores = prompt_scores[prompt_offset : prompt_offset + count]
        results[index] = _aggregate_unieval_scores(normalized_dimension, current_scores)
        prompt_offset += count

    return results


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
class ExactMatchConfig(BaseGenerationMetricConfig):
    """Configuration for SQuAD-style exact match metric."""

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return exact_match

    def get_metric_kwargs(self) -> dict[str, Any]:
        """Return kwargs for the metric function."""
        return {}


@dataclass
class TokenF1Config(BaseGenerationMetricConfig):
    """Configuration for SQuAD-style token F1 metric."""

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return token_f1

    def get_metric_kwargs(self) -> dict[str, Any]:
        """Return kwargs for the metric function."""
        return {}


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
class UniEvalConfig(BaseGenerationMetricConfig):
    """Configuration for one UniEval dimension."""

    dimension: str = "consistency"
    model_name_or_path: str = UNIEVAL_MODEL_NAME
    batch_size: int = 8
    max_length: int = 1024
    device: str = "cpu"
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        """Validate UniEval configuration."""
        self.dimension = _validate_unieval_dimension(self.dimension)

    def get_metric_name(self) -> str:
        """Return the metric name."""
        return f"unieval_{self.dimension}"

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return unieval

    def get_metric_kwargs(self) -> dict[str, Any]:
        """Return kwargs for the metric function."""
        return {
            "dimension": self.dimension,
            "model_name_or_path": self.model_name_or_path,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "device": self.device,
            "cache_dir": self.cache_dir,
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
