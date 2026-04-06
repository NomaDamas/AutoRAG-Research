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
            "question: Is this a fluent paragraph? </s> paragraph: First sentence.",
            "question: Is this a fluent paragraph? </s> paragraph: Second sentence.",
        ]
    ]


def test_unieval_coherence_requires_retrieved_context_and_uses_document_prompt():
    scorer = DummyUniEvalScorer(responses=[0.61])

    scores = unieval(
        metric_inputs=[
            MetricInput(
                generated_texts="Paris is the capital of France. It is in Europe.",
                retrieved_contents=["France is a country in Europe.", "Paris is its capital city."],
            )
        ],
        dimension="coherence",
        scorer=scorer,
    )

    assert scores == [pytest.approx(0.61)]
    assert scorer.calls == [
        [
            "question: Is this a coherent summary to the document? </s> summary: Paris is the capital of France. "
            "It is in Europe. </s> document: France is a country in Europe. Paris is its capital city."
        ]
    ]


def test_unieval_coherence_returns_none_without_retrieved_context():
    scores = unieval(
        metric_inputs=[MetricInput(generated_texts="Paris is the capital of France.")],
        dimension="coherence",
        scorer=DummyUniEvalScorer(responses=[0.9]),
    )

    assert scores == [None]


def test_unieval_consistency_requires_retrieved_context():
    scorer = DummyUniEvalScorer(responses=[0.9])

    scores = unieval(
        metric_inputs=[MetricInput(generated_texts="Paris is the capital of France.")],
        dimension="consistency",
        scorer=scorer,
    )

    assert scores == [None]
    assert scorer.calls == []


def test_unieval_consistency_averages_sentence_scores_against_document():
    scorer = DummyUniEvalScorer(responses=[0.25, 0.75])

    scores = unieval(
        metric_inputs=[
            MetricInput(
                generated_texts="Paris is the capital of France. It is in Europe.",
                retrieved_contents=["France is a country in Europe.", "Paris is its capital city."],
            )
        ],
        dimension="consistency",
        scorer=scorer,
    )

    assert scores == [pytest.approx(0.5)]
    assert scorer.calls == [
        [
            "question: Is this claim consistent with the document? </s> claim: Paris is the capital of France. "
            "</s> document: France is a country in Europe. Paris is its capital city.",
            "question: Is this claim consistent with the document? </s> claim: It is in Europe. "
            "</s> document: France is a country in Europe. Paris is its capital city.",
        ]
    ]


def test_unieval_relevance_requires_generation_gt_and_uses_reference_prompt():
    scorer = DummyUniEvalScorer(responses=[0.73])

    scores = unieval(
        metric_inputs=[
            MetricInput(
                generation_gt=["Paris is France's capital."],
                generated_texts="Paris is the capital of France.",
            )
        ],
        dimension="relevance",
        scorer=scorer,
    )

    assert scores == [pytest.approx(0.73)]
    assert scorer.calls == [
        [
            "question: Is this summary relevant to the reference? </s> summary: Paris is the capital of France. "
            "</s> reference: Paris is France's capital."
        ]
    ]


def test_unieval_relevance_checks_all_references_without_order_dependence():
    first_scorer = DummyUniEvalScorer(responses=[0.14, 0.86])
    second_scorer = DummyUniEvalScorer(responses=[0.86, 0.14])

    first_scores = unieval(
        metric_inputs=[
            MetricInput(
                generation_gt=["Wrong first reference", "Paris is France's capital."],
                generated_texts="Paris is the capital of France.",
            )
        ],
        dimension="relevance",
        scorer=first_scorer,
    )
    second_scores = unieval(
        metric_inputs=[
            MetricInput(
                generation_gt=["Paris is France's capital.", "Wrong second reference"],
                generated_texts="Paris is the capital of France.",
            )
        ],
        dimension="relevance",
        scorer=second_scorer,
    )

    assert first_scores == [pytest.approx(0.86)]
    assert second_scores == [pytest.approx(0.86)]
    assert first_scorer.calls == [
        [
            "question: Is this summary relevant to the reference? </s> summary: Paris is the capital of France. "
            "</s> reference: Wrong first reference",
            "question: Is this summary relevant to the reference? </s> summary: Paris is the capital of France. "
            "</s> reference: Paris is France's capital.",
        ]
    ]
    assert second_scorer.calls == [
        [
            "question: Is this summary relevant to the reference? </s> summary: Paris is the capital of France. "
            "</s> reference: Paris is France's capital.",
            "question: Is this summary relevant to the reference? </s> summary: Paris is the capital of France. "
            "</s> reference: Wrong second reference",
        ]
    ]


def test_unieval_relevance_returns_none_without_generation_gt():
    scores = unieval(
        metric_inputs=[MetricInput(generated_texts="Paris is the capital of France.")],
        dimension="relevance",
        scorer=DummyUniEvalScorer(responses=[0.9]),
    )

    assert scores == [None]


@pytest.mark.parametrize(
    ("dimension", "metric_name"),
    [
        ("coherence", "unieval_coherence"),
        ("consistency", "unieval_consistency"),
        ("fluency", "unieval_fluency"),
        ("relevance", "unieval_relevance"),
    ],
)
def test_unieval_config_uses_dimension_specific_metric_name(dimension: str, metric_name: str):
    config = UniEvalConfig(dimension=dimension, batch_size=4, device="cpu")

    assert config.get_metric_name() == metric_name
    assert config.get_metric_kwargs()["dimension"] == dimension
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
            metric_inputs=[
                MetricInput(generated_texts="A coherent answer.", retrieved_contents=["supporting document"])
            ],
            dimension="coherence",
        )


def test_unieval_skips_scorer_loading_when_all_inputs_are_missing_required_fields(monkeypatch: pytest.MonkeyPatch):
    def fail_to_load(**_: object) -> object:
        raise AssertionError

    monkeypatch.setattr(
        "autorag_research.evaluation.metrics.generation.get_unieval_scorer",
        fail_to_load,
    )

    scores = unieval(
        metric_inputs=[MetricInput(generated_texts="Paris is the capital of France.")],
        dimension="relevance",
    )

    assert scores == [None]


def test_unieval_raises_when_scorer_returns_wrong_number_of_scores():
    with pytest.raises(ValueError, match="returned 1 scores for 2 prompts"):
        unieval(
            metric_inputs=[MetricInput(generated_texts="First sentence. Second sentence.")],
            dimension="fluency",
            scorer=DummyUniEvalScorer(responses=[0.2]),
        )
