from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake import FakeListLLM
from langchain_openai import OpenAIEmbeddings

from autorag_research import cli
from autorag_research.evaluation.metrics.generation import (
    BartScoreF1Config,
    BartScoreFaithfulnessConfig,
    BartScorePrecisionConfig,
    BartScoreRecallConfig,
    bart_score_f1,
    bart_score_faithfulness,
    bart_score_precision,
    bart_score_recall,
    bert_score,
    bleu,
    meteor,
    response_relevancy,
    rouge,
    sem_score,
)
from autorag_research.schema import MetricInput
from tests.mock import mock_embed_documents

cli.CONFIG_PATH = Path(__file__).parent.parent.parent.parent.parent / "configs"

generation_gts = [
    ["The dog had bit the man.", "The man had bitten the dog."],
    ["I want to be a artist, but I end up to be a programmer."],
    [
        "To be a artist these days, you can overcome by AI.",
        "To be a programmer these days, you can overcome by AI.",
        "To be a lawyer these days, you can overcome by AI.",
    ],
]

retrieval_gt_contents = [
    [["The dog bite something easily. Actually the dog can bite a human. When you see a dog, you should be careful."]],
    [["The artist is a person who creates art. The artist can be a painter, a sculptor, or a musician."]],
    [
        [
            "AI is a technology that can simulate human intelligence. AI can be used in various fields such as healthcare, finance, and transportation. So its potential is huge."
        ]
    ],
]

generations = [
    "The dog bit the man.",
    "It really like to be a programmer, but I think artist is my passion.",
    "To be a artist these days, you can overcome by AI.",
]

ko_generation_gts = [
    ["개가 남자를 물었다.", "남자가 개를 물었다."],
    ["나는 예술가가 되고 싶었지만, 결국 프로그래머가 되었다."],
    [
        "요즘 예술가가 되려면, AI를 이겨야 한다.",
        "요즘 프로그래머가 되려면, AI를 이겨야 한다.",
        "요즘 변호사가 되려면, AI를 이겨야 한다.",
    ],
]

ko_generations = [
    "개가 남자를 물었다.",
    "나는 정말이지 예술가가 되고 싶었지만, 결국 프로그래머가 되었다.",
    "요즘 세상에서는 예술가가 되려면, AI를 이겨야 한다.",
]

ja_generation_gts = [
    ["犬が男を噛んだ。", "男が犬を噛んだ。"],
    ["私は芸術家になりたかったが、結局プログラマーになった。"],
    [
        "最近では、芸術家になるためにはAIに打ち勝つ必要がある。",
        "最近では、プログラマーになるためにはAIに打ち勝つ必要がある。",
        "最近では、弁護士になるためにはAIに打ち勝つ必要がある。",
    ],
]

ja_generations = [
    "犬が男を噛んだ。",
    "本当にプログラマーになることになったが、芸術家になるのが自分の情熱だ。",
    "最近では、芸術家になるためにはAIに打ち勝つ必要がある。",
]

summarization_query_list = [
    """
The 'coverage score' is calculated as the percentage of assessment questions
for which both the summary and the original document provide a 'yes' answer. This
method ensures that the summary not only includes key information from the original
text but also accurately represents it. A higher coverage score indicates a
more comprehensive and faithful summary, signifying that the summary effectively
encapsulates the crucial points and details from the original content.
""",
    """The 'coverage score' is calculated as the percentage of assessment questions
for which both the summary and the original document provide a 'yes' answer. This
method ensures that the summary not only includes key information from the original
text but also accurately represents it. A higher coverage score indicates a
more comprehensive and faithful summary, signifying that the summary effectively
encapsulates the crucial points and details from the original content.""",
]
summarization_generated_texts_list = [
    """
The coverage score quantifies how well a summary captures and
accurately represents key information from the original text,
with a higher score indicating greater comprehensiveness.
""",
    """In the latest One Piece chapters, the story shifts focus to two key developments:

The Straw Hat Crew's Separation: As they head toward Elbaf, the crew gets separated. Nami wakes up in a strange, Lego-like kingdom where she faces a dangerous, transforming creature. Luffy, Zoro, and Sanji come to her rescue, while other crew members fates, like Chopper, remain unknown(
Dexerto
).

Kuma's Storyline: Bartholomew Kumas past continues to unfold. He's shown making tough choices regarding his daughter, Bonney, and a deal with Vegapunk involving clone technology. His deepening ties with the Revolutionary Army and the threat from the Marines add further tension(
OtakuKart
).""",
]

summarization_metric_inputs = [
    MetricInput(generated_texts=gen, query=q)
    for gen, q in zip(summarization_generated_texts_list, summarization_query_list, strict=True)
]
similarity_generation_metric_inputs = [
    MetricInput(
        generated_texts=gen,
        generation_gt=gen_gt,
        retrieval_gt_contents=retrieval_gt_content,
    )
    for gen, gen_gt, retrieval_gt_content in zip(generations, generation_gts, retrieval_gt_contents, strict=True)
]
ko_similarity_generation_metric_inputs = [
    MetricInput(generated_texts=gen, generation_gt=gen_gt)
    for gen, gen_gt in zip(ko_generations, ko_generation_gts, strict=True)
]
ja_similarity_generation_metric_inputs = [
    MetricInput(generated_texts=gen, generation_gt=gen_gt)
    for gen, gen_gt in zip(ja_generations, ja_generation_gts, strict=True)
]
general_metric_inputs_with_gt = [
    MetricInput(
        query="What are the benefits of space exploration?",
        retrieval_gt_contents=[
            [
                "Space exploration has led to technological advancements such as satellite communication, GPS, "
                "and weather forecasting."
            ],
            [
                "It also contributes to scientific research, expanding our understanding of the universe, and fosters international cooperation in space missions."
            ],
        ],
        retrieved_contents=[
            "Space exploration has resulted in numerous technological advancements, including satellite technology, which has revolutionized communication and weather prediction.",
            "It has also expanded our understanding of the cosmos and encouraged international collaboration in scientific research.",
        ],
        generated_texts="The benefits of space exploration include technological innovations like satellite communications and GPS, which have improved life on Earth. Additionally, space exploration contributes to scientific knowledge and fosters international cooperation.",
        generation_gt=[
            "Space exploration brings technological advancements, such as satellites and GPS, that improve daily life. It also enhances our scientific understanding of the universe and encourages cooperation between nations."
        ],
    ),
    MetricInput(
        query="What are the major causes of climate change?",
        retrieval_gt_contents=[
            [
                "The major causes of climate change include the burning of fossil fuels such as coal, oil, and gas, deforestation, and industrial activities."
            ],
            [
                "Human activities like agriculture and waste management also contribute to the increase in greenhouse gases, leading to climate change."
            ],
        ],
        retrieved_contents=[
            "Climate change is primarily driven by human activities like the burning of fossil fuels, which release carbon dioxide (CO2) and other greenhouse gases.",
            "Deforestation and certain industrial activities also play a significant role in global warming.",
        ],
        generated_texts="Climate change is caused by human activities such as the burning of fossil fuels, deforestation, and industrial production. These activities release large amounts of greenhouse gases into the atmosphere, leading to global warming and other climate-related changes.",
        generation_gt=[
            "The main causes of climate change are the burning of fossil fuels, deforestation, and industrial activities that emit greenhouse gases. These gases trap heat in the atmosphere, causing global warming."
        ],
    ),
]


def base_test_metrics(func, solution, metric_inputs, **kwargs):
    scores = func(metric_inputs, **kwargs)
    assert len(scores) == len(metric_inputs)
    assert all(isinstance(score, float) for score in scores)
    assert all(x[0] == pytest.approx(x[1], 0.001) for x in zip(scores, solution, strict=True))


def test_bleu():
    base_test_metrics(
        bleu,
        [51.1507, 23.5783, 100.0],
        similarity_generation_metric_inputs,
        lowercase=True,
    )


def test_meteor():
    base_test_metrics(
        meteor,
        [0.454033, 0.2985435, 0.64077828],
        similarity_generation_metric_inputs,
        alpha=0.85,
        beta=0.2,
        gamma=0.6,
    )


def test_rouge():
    base_test_metrics(rouge, [0.909, 0.35714, 1.0], similarity_generation_metric_inputs)


@patch.object(
    OpenAIEmbeddings,
    "embed_documents",
    mock_embed_documents,
)
@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
def test_sem_score_other_model():
    scores = sem_score(
        metric_inputs=similarity_generation_metric_inputs,
        embedding_model=OpenAIEmbeddings(),
    )
    assert len(scores) == len(generation_gts)
    assert all(isinstance(score, float) for score in scores)


def test_sem_score_from_string_configs():
    scores = sem_score(
        metric_inputs=similarity_generation_metric_inputs,
        embedding_model="mock",
    )

    assert len(scores) == len(generation_gts)
    assert all(isinstance(score, float) for score in scores)


@pytest.mark.gpu
def test_bert_score_en():
    base_test_metrics(
        bert_score,
        [0.981902, 0.93164, 1.0],
        similarity_generation_metric_inputs,
        n_threads=8,
    )


@pytest.mark.gpu
def test_bert_score_ko():
    base_test_metrics(
        bert_score,
        [1.0, 0.965312, 0.96309],
        ko_similarity_generation_metric_inputs,
        lang="ko",
    )


@pytest.mark.gpu
def test_bert_score_ja():
    base_test_metrics(
        bert_score,
        [1.0, 0.82659, 1.0],
        ja_similarity_generation_metric_inputs,
        lang="ja",
    )


class KeywordEmbeddings(Embeddings):
    """Deterministic embeddings for response relevancy tests."""

    @staticmethod
    def _encode(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float("france" in lowered),
            float("capital" in lowered),
            float("einstein" in lowered),
        ]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]


def test_response_relevancy_ragas_scoring_logic():
    metric_inputs = [
        MetricInput(
            query="Where is France and what is its capital?",
            generated_texts="France is in western Europe and Paris is its capital.",
        )
    ]
    llm = FakeListLLM(
        responses=[
            '{"question":"Where is France and what is its capital?","noncommittal":0}',
            '{"question":"What is the capital of France?","noncommittal":0}',
            '{"question":"What is France\'s capital?","noncommittal":0}',
        ]
    )

    scores = response_relevancy(metric_inputs, llm=llm, embedding_model=KeywordEmbeddings(), strictness=3)

    assert scores[0] == pytest.approx(1.0)


def test_response_relevancy_noncommittal_zeroes_score():
    metric_inputs = [
        MetricInput(
            query="Where is France and what is its capital?",
            generated_texts="I don't know.",
        )
    ]
    llm = FakeListLLM(
        responses=[
            '{"question":"What is France\'s capital?","noncommittal":1}',
            '{"question":"Where is France located?","noncommittal":1}',
            '{"question":"What is the capital of France?","noncommittal":1}',
        ]
    )

    scores = response_relevancy(metric_inputs, llm=llm, embedding_model=KeywordEmbeddings(), strictness=3)

    assert scores == [0.0]


def test_response_relevancy_mixed_noncommittal_keeps_score():
    metric_inputs = [
        MetricInput(
            query="Where is France and what is its capital?",
            generated_texts="France is in western Europe and Paris is its capital.",
        )
    ]
    llm = FakeListLLM(
        responses=[
            '{"question":"Where is France and what is its capital?","noncommittal":0}',
            '{"question":"What is the capital of France?","noncommittal":1}',
            '{"question":"What is France\'s capital?","noncommittal":0}',
        ]
    )

    scores = response_relevancy(metric_inputs, llm=llm, embedding_model=KeywordEmbeddings(), strictness=3)

    assert scores[0] == pytest.approx(1.0)


def test_response_relevancy_invalid_json_returns_nan():
    metric_inputs = [
        MetricInput(
            query="Where was Albert Einstein born?",
            generated_texts="Albert Einstein was born in Germany.",
        )
    ]
    llm = FakeListLLM(responses=["{}", "not-json", '```json\n{"foo":1}\n```'])

    scores = response_relevancy(metric_inputs, llm=llm, embedding_model=KeywordEmbeddings(), strictness=3)

    assert scores[0] != scores[0]  # NaN check


def test_bart_score_variants_use_expected_text_directions(monkeypatch: pytest.MonkeyPatch):
    import autorag_research.evaluation.metrics.generation as generation_module

    calls: list[tuple[list[str], list[str], dict[str, object]]] = []
    score_map = {
        ("ctx 1\n\nctx 2", "candidate answer"): -0.11,
        ("reference one", "candidate answer"): -0.42,
        ("reference two", "candidate answer"): -0.21,
        ("candidate answer", "reference one"): -0.17,
        ("candidate answer", "reference two"): -0.31,
    }

    def fake_score_bartscore_pairs(
        src_texts: list[str],
        tgt_texts: list[str],
        **kwargs: object,
    ) -> list[float]:
        calls.append((src_texts, tgt_texts, kwargs))
        return [score_map[(src_text, tgt_text)] for src_text, tgt_text in zip(src_texts, tgt_texts, strict=True)]

    monkeypatch.setattr(generation_module, "_score_bartscore_pairs", fake_score_bartscore_pairs)

    metric_inputs = [
        MetricInput(
            retrieved_contents=["ctx 1", "ctx 2"],
            generated_texts="candidate answer",
            generation_gt=["reference one", "reference two"],
        )
    ]

    assert bart_score_faithfulness(
        metric_inputs, checkpoint="checkpoint", batch_size=8, max_length=128, device="cpu"
    ) == [-0.11]
    assert bart_score_precision(metric_inputs, checkpoint="checkpoint", batch_size=8, max_length=128, device="cpu") == [
        -0.21
    ]
    assert bart_score_recall(metric_inputs, checkpoint="checkpoint", batch_size=8, max_length=128, device="cpu") == [
        -0.17
    ]
    assert bart_score_f1(metric_inputs, checkpoint="checkpoint", batch_size=8, max_length=128, device="cpu") == [
        pytest.approx((-0.21 + -0.17) / 2)
    ]

    assert calls[0] == (
        ["ctx 1\n\nctx 2"],
        ["candidate answer"],
        {"checkpoint": "checkpoint", "batch_size": 8, "max_length": 128, "device": "cpu"},
    )
    assert calls[1] == (
        ["reference one", "reference two"],
        ["candidate answer", "candidate answer"],
        {"checkpoint": "checkpoint", "batch_size": 8, "max_length": 128, "device": "cpu"},
    )
    assert calls[2] == (
        ["candidate answer", "candidate answer"],
        ["reference one", "reference two"],
        {"checkpoint": "checkpoint", "batch_size": 8, "max_length": 128, "device": "cpu"},
    )


def test_bart_score_configs_expose_metric_functions_and_names():
    faithfulness_config = BartScoreFaithfulnessConfig()
    precision_config = BartScorePrecisionConfig()
    recall_config = BartScoreRecallConfig()
    f1_config = BartScoreF1Config()

    expected_kwargs = {
        "checkpoint": "facebook/bart-large-cnn",
        "batch_size": 4,
        "device": "auto",
        "max_length": 1024,
    }

    assert faithfulness_config.get_metric_name() == "bart_score_faithfulness"
    assert faithfulness_config.get_metric_func() is bart_score_faithfulness
    assert faithfulness_config.get_metric_kwargs() == expected_kwargs

    assert precision_config.get_metric_name() == "bart_score_precision"
    assert precision_config.get_metric_func() is bart_score_precision
    assert precision_config.get_metric_kwargs() == expected_kwargs

    assert recall_config.get_metric_name() == "bart_score_recall"
    assert recall_config.get_metric_func() is bart_score_recall
    assert recall_config.get_metric_kwargs() == expected_kwargs

    assert f1_config.get_metric_name() == "bart_score_f1"
    assert f1_config.get_metric_func() is bart_score_f1
    assert f1_config.get_metric_kwargs() == expected_kwargs


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected_device"),
    [
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_resolve_bartscore_device_auto_prefers_accelerators(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    expected_device: str,
):
    import sys
    from types import SimpleNamespace

    import autorag_research.evaluation.metrics.generation as generation_module

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps_available)),
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert generation_module._resolve_bartscore_device("auto") == expected_device


def test_resolve_bartscore_device_auto_falls_back_to_cpu_without_mps_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    import sys
    from types import SimpleNamespace

    import autorag_research.evaluation.metrics.generation as generation_module

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(),
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert generation_module._resolve_bartscore_device("auto") == "cpu"


def test_bart_score_runtime_guard_points_to_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
):
    import autorag_research.evaluation.metrics.generation as generation_module

    original_import_module = generation_module.importlib.import_module

    def fake_import_module(name: str):
        if name == "transformers":
            msg = "missing transformers"
            raise ImportError(msg)
        if name == "torch":
            return object()
        return original_import_module(name)

    monkeypatch.setattr(generation_module.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match=r"autorag-research\[gpu\]") as exc_info:
        generation_module._import_bartscore_runtime()

    assert "uv sync --all-extras --all-groups" in str(exc_info.value)
