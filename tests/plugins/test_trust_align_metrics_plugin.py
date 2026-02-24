import importlib
import sys
from pathlib import Path

import pytest

from autorag_research.schema import MetricInput

PLUGIN_SRC = Path(__file__).resolve().parents[2] / "plugins" / "trust_align_metrics_plugin" / "src"
if str(PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SRC))

trust_align_metric = importlib.import_module("trust_align_metrics_plugin.metric")
TrustAlignAnswerCorrectnessF1Config = trust_align_metric.TrustAlignAnswerCorrectnessF1Config
TrustAlignGroundedRefusalF1Config = trust_align_metric.TrustAlignGroundedRefusalF1Config
trust_align_answer_correctness_f1 = trust_align_metric.trust_align_answer_correctness_f1
trust_align_grounded_refusal_f1 = trust_align_metric.trust_align_grounded_refusal_f1


def test_trust_align_metric_config_granularity():
    grounded_refusal_cfg = TrustAlignGroundedRefusalF1Config()
    answer_correctness_cfg = TrustAlignAnswerCorrectnessF1Config()

    assert grounded_refusal_cfg.get_compute_granularity() == "dataset"
    assert answer_correctness_cfg.get_compute_granularity() == "dataset"


def test_trust_align_grounded_refusal_f1_dataset_level(monkeypatch: pytest.MonkeyPatch):
    metric_inputs = [
        MetricInput(
            query="q1",
            generated_texts="I apologize, but I couldn't find an answer.",
            generation_gt=["alpha claim"],
            retrieved_contents=["unrelated evidence"],
        ),
        MetricInput(
            query="q2",
            generated_texts="alpha claim is true.",
            generation_gt=["alpha claim"],
            retrieved_contents=["Document evidence: alpha claim is supported."],
        ),
        MetricInput(
            query="q3",
            generated_texts="Here is a speculative answer.",
            generation_gt=["beta claim"],
            retrieved_contents=["No support for beta here."],
        ),
    ]

    support_map = {
        ("Document evidence: alpha claim is supported.", "alpha claim"): 1,
    }

    def fake_run_nli_autoais(passage: str, claim: str, autoais_model: str) -> int:
        return support_map.get((passage, claim), 0)

    monkeypatch.setattr(
        "trust_align_metrics_plugin.metric._run_nli_autoais",
        fake_run_nli_autoais,
    )

    scores = trust_align_grounded_refusal_f1(
        metric_inputs=metric_inputs,
        refusal_flag="I apologize, but I couldn't find an answer",
        refusal_threshold=85,
    )

    assert len(scores) == len(metric_inputs)
    assert all(score == pytest.approx(66.6666667) for score in scores)


def test_trust_align_answer_correctness_f1_dataset_level(monkeypatch: pytest.MonkeyPatch):
    metric_inputs = [
        MetricInput(
            query="q1",
            generated_texts="alpha claim is true.",
            generation_gt=["alpha claim", "beta claim"],
            retrieved_contents=["Document evidence: alpha claim is supported."],
        ),
        MetricInput(
            query="q2",
            generated_texts="This is an overconfident answer.",
            generation_gt=["gamma claim"],
            retrieved_contents=["No support in evidence."],
        ),
        MetricInput(
            query="q3",
            generated_texts="I apologize, but I couldn't find an answer.",
            generation_gt=["delta claim"],
            retrieved_contents=["Document evidence: delta claim is supported."],
        ),
    ]

    entailment_map = {
        ("Document evidence: alpha claim is supported.", "alpha claim"): 1,
        ("Document evidence: alpha claim is supported.", "beta claim"): 0,
        ("No support in evidence.", "gamma claim"): 0,
        ("Document evidence: delta claim is supported.", "delta claim"): 1,
        ("alpha claim is true.", "alpha claim"): 1,
        ("alpha claim is true.", "beta claim"): 0,
        ("This is an overconfident answer.", "gamma claim"): 0,
        ("I apologize, but I couldn't find an answer.", "delta claim"): 0,
    }

    def fake_run_nli_autoais(passage: str, claim: str, autoais_model: str) -> int:
        return entailment_map.get((passage, claim), 0)

    monkeypatch.setattr(
        "trust_align_metrics_plugin.metric._run_nli_autoais",
        fake_run_nli_autoais,
    )

    scores = trust_align_answer_correctness_f1(
        metric_inputs=metric_inputs,
        refusal_flag="I apologize, but I couldn't find an answer",
        refusal_threshold=85,
    )

    assert len(scores) == len(metric_inputs)
    assert all(score == pytest.approx(50.0) for score in scores)
