import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseLanguageModel

import evaluate
import nltk
import pandas as pd
from rouge_score import tokenizers
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics.bleu import BLEU

from autorag_research.config import BaseGenerationMetricConfig
from autorag_research.evaluation.metrics.util import calculate_cosine_similarity, metric_loop
from autorag_research.exceptions import EmbeddingError
from autorag_research.injection import with_embedding, with_llm
from autorag_research.schema import MetricInput
from autorag_research.util import convert_inputs_to_list, normalize_string, truncate_texts, unpack_and_run

REFUSAL_JUDGE_SYSTEM_PROMPT = "You are an intelligent evaluator."

REFUSAL_JUDGE_USER_PROMPT = """Instruction: You are an intelligent evaluator. Given a pair of Question and Answer, your task is to determine if the Answer is classified as REFUSED or NOT REFUSED. Please follow these steps to produce your judgement:
1. REFUSED means that the Answer contains any sentiment of being unable to provide a response. This is regardless of whether the Answer then contains information to answer the question.
2. NOT REFUSED means the absence of any sentiment of being unable to provide a response. The Answer attempts to address the Question.
3. Carefully read and understand the Question and Answer.
4. Before determining your final judgement of REFUSED or NOT REFUSED, state your reasoning and evidence (by extracting minimal subspan of Answer accurately) first.
5. For each Answer, determine if it is REFUSED or NOT REFUSED.
6. At the end of your response, return the following:
- Reasoning: Why do you think this should be classified as REFUSED or NOT REFUSED.
- Span: minimal subspan of Answer accurately extracted that supports reasoning
- Judgement: whether it should be classified as REFUSED or NOT REFUSED.
Question: {question}
Answer: {answer}
Return the following:
Reasoning:
Span:
Judgement:
"""


def _safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two values and return 0 when denominator is 0."""
    return numerator / denominator if denominator > 0 else 0.0


def _safe_f1(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall with zero-division guard."""
    return _safe_divide(2.0 * precision * recall, precision + recall)


def _extract_response_text(response: Any) -> str:
    """Extract text from LangChain response object."""
    return response.content if hasattr(response, "content") else str(response)


def _normalize_non_empty(text: str | None) -> str:
    """Normalize text and return empty string for None/blank."""
    if text is None:
        return ""
    normalized = normalize_string(text)
    return normalized.strip()


def _is_refusal_by_phrase(answer: str, rejection_flag: str, rejection_threshold: int) -> bool:
    """Fallback phrase-based refusal detector for constrained environments."""
    normalized_answer = _normalize_non_empty(answer)
    normalized_flag = _normalize_non_empty(rejection_flag)
    if not normalized_answer or not normalized_flag:
        return False
    if normalized_flag in normalized_answer:
        return True

    overlap = len(set(normalized_answer.split()) & set(normalized_flag.split()))
    ratio = _safe_divide(overlap * 100.0, len(set(normalized_flag.split())))
    return ratio >= rejection_threshold


def _parse_refusal_judgement(text: str) -> bool | None:
    """Parse REFUSED / NOT REFUSED from evaluator output."""
    upper_text = text.upper()
    matches = re.findall(r"JUDGEMENT\s*:\s*(REFUSED|NOT REFUSED)", upper_text)
    if matches:
        return matches[-1] == "REFUSED"
    if "NOT REFUSED" in upper_text:
        return False
    if "REFUSED" in upper_text:
        return True
    return None


def _is_answerable(metric_input: MetricInput) -> bool:
    """Determine answerability from generation ground truth."""
    return bool(metric_input.generation_gt)


def _judge_refusal(
    metric_input: MetricInput,
    judge_mode: Literal["llm", "phrase"],
    judge_llm: "BaseLanguageModel",
    rejection_flag: str,
    rejection_threshold: int,
) -> bool:
    """Judge whether a model answer is a refusal."""
    generated_text = metric_input.generated_texts or ""
    if generated_text.strip() == "":
        return True

    if judge_mode == "llm":
        from langchain_core.messages import HumanMessage, SystemMessage

        query_text = metric_input.query or ""
        prompt = REFUSAL_JUDGE_USER_PROMPT.format(question=query_text, answer=generated_text)
        response = judge_llm.invoke([SystemMessage(content=REFUSAL_JUDGE_SYSTEM_PROMPT), HumanMessage(content=prompt)])
        parsed = _parse_refusal_judgement(_extract_response_text(response))
        if parsed is not None:
            return parsed

    return _is_refusal_by_phrase(generated_text, rejection_flag, rejection_threshold)


def _claim_in_text(claim: str, text: str) -> bool:
    """Check whether a normalized claim is present in a text."""
    normalized_claim = _normalize_non_empty(claim)
    normalized_text = _normalize_non_empty(text)
    if not normalized_claim or not normalized_text:
        return False
    return normalized_claim in normalized_text


def _calibrated_gold_claims(metric_input: MetricInput, use_retrieval_calibration: bool) -> list[str]:
    """Build calibrated claim set AG ∩ AD from generation GT and retrieval contexts."""
    claims = metric_input.generation_gt or []
    normalized_claims = []
    seen: set[str] = set()
    for claim in claims:
        normalized_claim = _normalize_non_empty(claim)
        if normalized_claim and normalized_claim not in seen:
            seen.add(normalized_claim)
            normalized_claims.append(normalized_claim)

    if not use_retrieval_calibration:
        return normalized_claims

    retrieval_groups = metric_input.retrieval_gt_contents or []
    retrieval_contexts = [ctx for group in retrieval_groups for ctx in group if ctx]
    if not retrieval_contexts:
        return normalized_claims

    calibrated_claims: list[str] = []
    for claim in normalized_claims:
        if any(_claim_in_text(claim, ctx) for ctx in retrieval_contexts):
            calibrated_claims.append(claim)
    return calibrated_claims


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


@convert_inputs_to_list
@with_llm(param_name="judge_llm")
def grounded_refusal_f1(
    metric_inputs: list[MetricInput],
    judge_llm: "BaseLanguageModel | str" = "openai-gpt5-mini",
    judge_mode: Literal["llm", "phrase"] = "llm",
    rejection_flag: str = "I apologize, but I couldn't find an answer to your question in the search results.",
    rejection_threshold: int = 85,
) -> list[float]:
    """Compute grounded refusal F1 (F1GR) from dataset-level precision/recall.

    Paper-aligned formula:
    F1GR = 0.5 * (F1ref + F1ans)
    where F1ref measures correct refusals for unanswerable questions and
    F1ans measures correct non-refusal answers for answerable questions.

    Returns:
        Dataset-level F1GR repeated for each input for DB compatibility.
    """
    if not metric_inputs:
        return []

    answerable_flags: list[bool] = []
    refusal_flags: list[bool] = []

    for metric_input in metric_inputs:
        answerable_flags.append(_is_answerable(metric_input))
        refusal_flags.append(
            _judge_refusal(
                metric_input=metric_input,
                judge_mode=judge_mode,
                judge_llm=judge_llm,  # ty: ignore[arg-type]
                rejection_flag=rejection_flag,
                rejection_threshold=rejection_threshold,
            )
        )

    ag = sum(answerable_flags)
    ar = sum(not refused for refused in refusal_flags)
    not_ag = len(answerable_flags) - ag
    not_ar = len(refusal_flags) - ar

    correct_refused_unanswerable = sum(
        (not answerable) and refused for answerable, refused in zip(answerable_flags, refusal_flags, strict=True)
    )
    pref = _safe_divide(correct_refused_unanswerable, not_ar)
    rref = _safe_divide(correct_refused_unanswerable, not_ag)
    f1_ref = _safe_f1(pref, rref)

    correct_answered_answerable = sum(
        answerable and (not refused) for answerable, refused in zip(answerable_flags, refusal_flags, strict=True)
    )
    pans = _safe_divide(correct_answered_answerable, ar)
    rans = _safe_divide(correct_answered_answerable, ag)
    f1_ans = _safe_f1(pans, rans)

    f1_gr = 0.5 * (f1_ref + f1_ans)
    return [f1_gr] * len(metric_inputs)


@convert_inputs_to_list
@with_llm(param_name="judge_llm")
def answer_correctness_f1(
    metric_inputs: list[MetricInput],
    judge_llm: "BaseLanguageModel | str" = "openai-gpt5-mini",
    judge_mode: Literal["llm", "phrase"] = "llm",
    rejection_flag: str = "I apologize, but I couldn't find an answer to your question in the search results.",
    rejection_threshold: int = 85,
    use_retrieval_calibration: bool = True,
) -> list[float]:
    """Compute calibrated answer correctness F1 (F1AC) at dataset-level.

    Follows paper definitions:
    - AC_q = |AG ∩ AD ∩ AR| / |AG ∩ AD|
    - PAC = sum(AC_q for q in Ag∩Ar) / |Ar|
    - RAC = sum(AC_q for q in Ag∩Ar) / |Ag|
    - F1AC = 2 * PAC * RAC / (PAC + RAC)

    Returns:
        Dataset-level F1AC repeated for each input for DB compatibility.
    """
    if not metric_inputs:
        return []

    answered_count = 0
    answerable_count = 0
    summed_ac = 0.0

    for metric_input in metric_inputs:
        answerable = _is_answerable(metric_input)
        if answerable:
            answerable_count += 1

        refused = _judge_refusal(
            metric_input=metric_input,
            judge_mode=judge_mode,
            judge_llm=judge_llm,  # ty: ignore[arg-type]
            rejection_flag=rejection_flag,
            rejection_threshold=rejection_threshold,
        )
        if refused:
            continue

        answered_count += 1
        if not answerable:
            continue

        calibrated_claims = _calibrated_gold_claims(metric_input, use_retrieval_calibration)
        if not calibrated_claims:
            continue

        generated_text = metric_input.generated_texts or ""
        correct_claims = sum(_claim_in_text(claim, generated_text) for claim in calibrated_claims)
        summed_ac += _safe_divide(correct_claims, len(calibrated_claims))

    pac = _safe_divide(summed_ac, answered_count)
    rac = _safe_divide(summed_ac, answerable_count)
    f1_ac = _safe_f1(pac, rac)
    return [f1_ac] * len(metric_inputs)


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
class GroundedRefusalF1Config(BaseGenerationMetricConfig):
    """Configuration for grounded refusal F1 (F1GR)."""

    judge_llm: "BaseLanguageModel | str" = "openai-gpt5-mini"
    judge_mode: Literal["llm", "phrase"] = "llm"
    rejection_flag: str = "I apologize, but I couldn't find an answer to your question in the search results."
    rejection_threshold: int = 85

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return grounded_refusal_f1

    def get_metric_kwargs(self) -> dict[str, Any]:
        """Return kwargs for the metric function."""
        return {
            "judge_llm": self.judge_llm,
            "judge_mode": self.judge_mode,
            "rejection_flag": self.rejection_flag,
            "rejection_threshold": self.rejection_threshold,
        }

    def get_compute_granularity(self) -> Literal["query", "dataset"]:
        """Grounded refusal is a dataset-level metric by definition."""
        return "dataset"


@dataclass
class AnswerCorrectnessF1Config(BaseGenerationMetricConfig):
    """Configuration for calibrated answer correctness F1 (F1AC)."""

    judge_llm: "BaseLanguageModel | str" = "openai-gpt5-mini"
    judge_mode: Literal["llm", "phrase"] = "llm"
    rejection_flag: str = "I apologize, but I couldn't find an answer to your question in the search results."
    rejection_threshold: int = 85
    use_retrieval_calibration: bool = True

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return answer_correctness_f1

    def get_metric_kwargs(self) -> dict[str, Any]:
        """Return kwargs for the metric function."""
        return {
            "judge_llm": self.judge_llm,
            "judge_mode": self.judge_mode,
            "rejection_flag": self.rejection_flag,
            "rejection_threshold": self.rejection_threshold,
            "use_retrieval_calibration": self.use_retrieval_calibration,
        }

    def get_compute_granularity(self) -> Literal["query", "dataset"]:
        """Answer correctness is a dataset-level metric by definition."""
        return "dataset"
