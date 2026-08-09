"""Trust-Align exact generation metric implementations."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher

try:
    from fuzzywuzzy import fuzz
except ModuleNotFoundError:

    class _FallbackFuzz:
        """Fallback implementation when fuzzywuzzy isn't installed."""

        @staticmethod
        def partial_ratio(a: str, b: str) -> int:
            if not a or not b:
                return 0
            short, long = (a, b) if len(a) <= len(b) else (b, a)
            window = len(short)
            best = 0.0
            for idx in range(len(long) - window + 1):
                ratio = SequenceMatcher(None, short, long[idx : idx + window]).ratio()
                if ratio > best:
                    best = ratio
            return round(best * 100)

    fuzz = _FallbackFuzz()

from autorag_research.config import BaseGenerationMetricConfig
from autorag_research.schema import MetricInput
from autorag_research.util import convert_inputs_to_list, normalize_string
from trust_align_metrics_plugin.autoais import run_nli_autoais


def _safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two values and return 0 when denominator is 0."""
    return numerator / denominator if denominator > 0 else 0.0


def _safe_f1(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall with zero-division guard."""
    return _safe_divide(2.0 * precision * recall, precision + recall)


def _mean(values: list[float]) -> float:
    """Compute arithmetic mean with empty-list guard."""
    return sum(values) / len(values) if values else 0.0


def _remove_citations(text: str) -> str:
    """Remove inline citation markers to mirror Trust-Eval normalization."""
    return re.sub(r"\[\d+\]", "", text)


def _normalize_for_match(text: str) -> str:
    """Normalize text with the same normalizer used by AutoRAG-Research."""
    return normalize_string(text)


def _is_refusal(output: str, refusal_flag: str, refusal_threshold: int) -> bool:
    """Detect refusal exactly like Trust-Eval (fuzzy partial ratio)."""
    return fuzz.partial_ratio(_normalize_for_match(refusal_flag), _normalize_for_match(output)) > refusal_threshold


def _compute_answers_found(claims: list[str], retrieved_contents: list[str], autoais_model: str) -> list[list[int]]:
    """Build doc-by-claim entailment matrix using AutoAIS."""
    if not claims:
        return []

    answers_found: list[list[int]] = []
    for doc in retrieved_contents:
        answers_found.append([_run_nli_autoais(doc, claim, autoais_model) for claim in claims])
    return answers_found


def _supported_claims(answers_found: list[list[int]], num_claims: int) -> list[int]:
    """Compute claim support vector by OR-ing entailment over retrieved docs."""
    if num_claims == 0:
        return []
    if not answers_found:
        return [0] * num_claims

    supported = [0] * num_claims
    for doc_support in answers_found:
        for idx, support in enumerate(doc_support):
            if support:
                supported[idx] = 1
    return supported


def _is_answerable(supported_claims: list[int]) -> bool:
    """A question is answerable iff at least one claim is supported by retrieved docs."""
    return any(supported_claims)


def _run_nli_autoais(passage: str, claim: str, autoais_model: str) -> int:
    """Wrapper that keeps monkeypatching straightforward in tests."""
    return run_nli_autoais(passage=passage, claim=claim, autoais_model=autoais_model)


@convert_inputs_to_list
def trust_align_grounded_refusal_f1(  # noqa: C901
    metric_inputs: list[MetricInput],
    autoais_model: str = "google/t5_xxl_true_nli_mixture",
    refusal_flag: str = "I apologize, but I couldn't find an answer",
    refusal_threshold: int = 85,
) -> list[float]:
    """Compute Trust-Align grounded refusal macro F1 (percentage scale)."""
    if not metric_inputs:
        return []

    refuse_rec_num = 0
    refuse_rec = 0
    refuse_prec_num = 0
    refuse_prec = 0

    ans_rec_num = 0
    ans_rec = 0
    ans_prec_num = 0
    ans_prec = 0

    for metric_input in metric_inputs:
        claims = metric_input.generation_gt or []
        retrieved_contents = metric_input.retrieved_contents or []
        answers_found = _compute_answers_found(claims, retrieved_contents, autoais_model)
        supported_claims = _supported_claims(answers_found, len(claims))

        answerable = _is_answerable(supported_claims)
        refusal = _is_refusal(metric_input.generated_texts or "", refusal_flag, refusal_threshold)

        if not answerable:
            refuse_rec_num += 1
            if refusal:
                refuse_rec += 1

        if refusal:
            refuse_prec_num += 1
            if not answerable:
                refuse_prec += 1

        if answerable:
            ans_rec_num += 1
            if not refusal:
                ans_rec += 1

        if not refusal:
            ans_prec_num += 1
            if answerable:
                ans_prec += 1

    refuse_recall = 100.0 * _safe_divide(refuse_rec, refuse_rec_num)
    refuse_precision = 100.0 * _safe_divide(refuse_prec, refuse_prec_num)
    refuse_f1 = _safe_f1(refuse_precision, refuse_recall)

    answerable_recall = 100.0 * _safe_divide(ans_rec, ans_rec_num)
    answerable_precision = 100.0 * _safe_divide(ans_prec, ans_prec_num)
    answerable_f1 = _safe_f1(answerable_precision, answerable_recall)

    macro_f1 = 0.5 * (refuse_f1 + answerable_f1)
    return [macro_f1] * len(metric_inputs)


@convert_inputs_to_list
def trust_align_answer_correctness_f1(
    metric_inputs: list[MetricInput],
    autoais_model: str = "google/t5_xxl_true_nli_mixture",
    refusal_flag: str = "I apologize, but I couldn't find an answer",
    refusal_threshold: int = 85,
) -> list[float]:
    """Compute Trust-Align calibrated claim-match F1 (percentage scale)."""
    if not metric_inputs:
        return []

    calib_answered_scores: list[float] = []
    calib_answerable_scores: list[float] = []

    for metric_input in metric_inputs:
        claims = metric_input.generation_gt or []
        retrieved_contents = metric_input.retrieved_contents or []

        answers_found = _compute_answers_found(claims, retrieved_contents, autoais_model)
        supported_claims = _supported_claims(answers_found, len(claims))
        answerable = _is_answerable(supported_claims)

        refusal = _is_refusal(metric_input.generated_texts or "", refusal_flag, refusal_threshold)
        normalized_output = _remove_citations(metric_input.generated_texts or "")

        supported_total = sum(supported_claims)
        calib_entail = 0
        if supported_total > 0 and not refusal:
            for idx, claim in enumerate(claims):
                if supported_claims[idx] == 1:
                    calib_entail += _run_nli_autoais(normalized_output, claim, autoais_model)

        if not refusal:
            if answerable and supported_total > 0:
                calib_answered_scores.append(calib_entail / supported_total)
            else:
                calib_answered_scores.append(0.0)

        if answerable:
            if not refusal and supported_total > 0:
                calib_answerable_scores.append(calib_entail / supported_total)
            else:
                calib_answerable_scores.append(0.0)

    calib_answered = 100.0 * _mean(calib_answered_scores)
    calib_answerable = 100.0 * _mean(calib_answerable_scores)
    calib_f1 = _safe_f1(calib_answered, calib_answerable)

    return [calib_f1] * len(metric_inputs)


@dataclass
class TrustAlignGroundedRefusalF1Config(BaseGenerationMetricConfig):
    """Configuration for Trust-Align grounded refusal F1 metric."""

    autoais_model: str = "google/t5_xxl_true_nli_mixture"
    refusal_flag: str = "I apologize, but I couldn't find an answer"
    refusal_threshold: int = 85

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return trust_align_grounded_refusal_f1

    def get_metric_kwargs(self) -> dict[str, object]:
        """Return kwargs for the metric function."""
        return {
            "autoais_model": self.autoais_model,
            "refusal_flag": self.refusal_flag,
            "refusal_threshold": self.refusal_threshold,
        }

    def get_compute_granularity(self):
        """Trust-Align metric is computed over the full dataset."""
        return "dataset"


@dataclass
class TrustAlignAnswerCorrectnessF1Config(BaseGenerationMetricConfig):
    """Configuration for Trust-Align answer correctness F1 metric."""

    autoais_model: str = "google/t5_xxl_true_nli_mixture"
    refusal_flag: str = "I apologize, but I couldn't find an answer"
    refusal_threshold: int = 85

    def get_metric_func(self) -> Callable:
        """Return the metric function."""
        return trust_align_answer_correctness_f1

    def get_metric_kwargs(self) -> dict[str, object]:
        """Return kwargs for the metric function."""
        return {
            "autoais_model": self.autoais_model,
            "refusal_flag": self.refusal_flag,
            "refusal_threshold": self.refusal_threshold,
        }

    def get_compute_granularity(self):
        """Trust-Align metric is computed over the full dataset."""
        return "dataset"
