from dataclasses import dataclass

import pytest

from autorag_research.evaluation.metrics.generation import UniEvalConfig, unieval
from autorag_research.schema import MetricInput


@dataclass
class DummyUniEvalScorer:
    responses: list[float]

    def __post_init__(self) -> None:
        self.calls: list[list[str]] = []

    def score(self, inputs: list[str], batch_size: int = 8) -> list[float]:
        self.calls.append(inputs)
        return self.responses[: len(inputs)]


def test_unieval_fluency_averages_sentence_scores():
    scorer = DummyUniEvalScorer(responses=[0.2, 0.8])

    scores = unieval(
        metric_inputs=[MetricInput(generated_texts="First sentence. Second sentence.")],
        dimension="fluency",
        scorer=scorer,
    )

    assert scores == [pytest.approx(0.5)]
    assert scorer.calls == [
        [
            "question: Is this a fluent answer? </s> answer: First sentence.",
            "question: Is this a fluent answer? </s> answer: Second sentence.",
        ]
    ]


def test_unieval_relevance_uses_query_prompt():
    scorer = DummyUniEvalScorer(responses=[0.73])

    scores = unieval(
        metric_inputs=[
            MetricInput(
                query="What is the capital of France?",
                generated_texts="Paris is the capital of France.",
            )
        ],
        dimension="relevance",
        scorer=scorer,
    )

    assert scores == [pytest.approx(0.73)]
    assert scorer.calls == [
        [
            "question: Is this answer relevant to the question? </s> answer: Paris is the capital of France. "
            "</s> question: What is the capital of France?"
        ]
    ]


def test_unieval_consistency_requires_retrieved_context():
    scores = unieval(
        metric_inputs=[MetricInput(generated_texts="Paris is the capital of France.")],
        dimension="consistency",
        scorer=DummyUniEvalScorer(responses=[0.9]),
    )

    assert scores == [None]


def test_unieval_config_uses_dimension_specific_metric_name():
    config = UniEvalConfig(dimension="consistency", batch_size=4, device="cpu")

    assert config.get_metric_name() == "unieval_consistency"
    assert config.get_metric_kwargs()["dimension"] == "consistency"
    assert config.get_metric_kwargs()["batch_size"] == 4


def test_unieval_invalid_dimension_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported UniEval dimension"):
        UniEvalConfig(dimension="unsupported")


def test_unieval_model_loading_failure_surfaces_helpful_error(monkeypatch: pytest.MonkeyPatch):
    def raise_import_error(**_: object) -> object:
        import importlib

        importlib.import_module("transformers_missing_for_unieval_test")
        return object()

    monkeypatch.setattr(
        "autorag_research.evaluation.metrics.generation.get_unieval_scorer",
        raise_import_error,
    )

    with pytest.raises(ImportError, match="transformers_missing_for_unieval_test"):
        unieval(
            metric_inputs=[MetricInput(generated_texts="A coherent answer.")],
            dimension="coherence",
        )


def test_unieval_raises_when_scorer_returns_wrong_number_of_scores():
    with pytest.raises(ValueError, match="returned 1 scores for 2 prompts"):
        unieval(
            metric_inputs=[MetricInput(generated_texts="First sentence. Second sentence.")],
            dimension="fluency",
            scorer=DummyUniEvalScorer(responses=[0.2]),
        )
