import math
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import yaml
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake import FakeListLLM
from langchain_openai import OpenAIEmbeddings

from autorag_research import cli
from autorag_research.evaluation.metrics.generation import (
    ALIGNSCORE_MAX_LENGTH,
    ALIGNSCORE_MODEL_NAME,
    ALIGNSCORE_REVISION,
    AlignScoreConfig,
    BartScoreF1Config,
    BartScoreFaithfulnessConfig,
    BartScorePrecisionConfig,
    BartScoreRecallConfig,
    ExactMatchConfig,
    TokenF1Config,
    align_score,
    bart_score_f1,
    bart_score_faithfulness,
    bart_score_precision,
    bart_score_recall,
    bert_score,
    bleu,
    exact_match,
    meteor,
    response_relevancy,
    rouge,
    sem_score,
    token_f1,
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


def test_exact_match_uses_squad_normalization():
    metric_inputs = [
        MetricInput(generated_texts="The Eiffel Tower!", generation_gt=["eiffel tower"]),
        MetricInput(generated_texts="Paris, France", generation_gt=["France Paris"]),
    ]

    scores = exact_match(metric_inputs)

    assert scores == [1.0, 0.0]


def test_exact_match_returns_best_score_across_references():
    metric_inputs = [
        MetricInput(
            generated_texts="Pacific Ocean",
            generation_gt=["atlantic ocean", "the pacific ocean"],
        )
    ]

    assert exact_match(metric_inputs) == [1.0]


def test_exact_match_handles_empty_normalized_answers():
    metric_inputs = [
        MetricInput(generated_texts="the", generation_gt=["an"]),
        MetricInput(generated_texts="the", generation_gt=["cat"]),
    ]

    scores = exact_match(metric_inputs)

    assert scores == [1.0, 0.0]


def test_token_f1_counts_repeated_tokens_with_bag_of_words_overlap():
    metric_inputs = [
        MetricInput(
            generated_texts="red red blue",
            generation_gt=["red blue blue"],
        )
    ]

    scores = token_f1(metric_inputs)

    assert scores == [pytest.approx(2 / 3)]


def test_token_f1_uses_best_reference_overlap():
    metric_inputs = [
        MetricInput(
            generated_texts="red blue",
            generation_gt=["red green", "red blue yellow"],
        )
    ]

    scores = token_f1(metric_inputs)

    assert scores == [pytest.approx(0.8)]


def test_token_f1_handles_empty_normalized_answers():
    metric_inputs = [
        MetricInput(generated_texts="the", generation_gt=["an"]),
        MetricInput(generated_texts="the", generation_gt=["cat"]),
    ]

    scores = token_f1(metric_inputs)

    assert scores == [1.0, 0.0]


def test_new_metric_configs_expose_metric_functions_and_names():
    exact_match_config = ExactMatchConfig()
    token_f1_config = TokenF1Config()

    assert exact_match_config.get_metric_name() == "exact_match"
    assert exact_match_config.get_metric_func() is exact_match
    assert exact_match_config.get_metric_kwargs() == {}

    assert token_f1_config.get_metric_name() == "token_f1"
    assert token_f1_config.get_metric_func() is token_f1
    assert token_f1_config.get_metric_kwargs() == {}


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


class DummyAlignScoreScorer:
    def __init__(self, responses: list[float]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[str], list[str], int]] = []

    def score(self, contexts: list[str], claims: list[str], batch_size: int = 8) -> list[float]:
        self.calls.append((contexts, claims, batch_size))
        return self.responses[: len(claims)]


class MappingAlignScoreScorer:
    def __init__(self, responses: dict[tuple[str, str], float]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[str], list[str], int]] = []

    def score(self, contexts: list[str], claims: list[str], batch_size: int = 8) -> list[float]:
        self.calls.append((contexts, claims, batch_size))
        return [self.responses[(context, claim)] for context, claim in zip(contexts, claims, strict=True)]


class KeywordAlignScoreScorer:
    tokenizer: "WhitespaceAlignScoreTokenizer"

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword
        self.tokenizer = WhitespaceAlignScoreTokenizer()
        self.calls: list[tuple[list[str], list[str], int]] = []

    def score(self, contexts: list[str], claims: list[str], batch_size: int = 8) -> list[float]:
        self.calls.append((contexts, claims, batch_size))
        return [0.99 if self.keyword in context else 0.01 for context in contexts]


class WhitespaceAlignScoreTokenizer:
    @staticmethod
    def _tokens(text: str) -> list[str]:
        return text.split()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        token_ids = list(range(len(self._tokens(text))))
        if add_special_tokens:
            return [-1, *token_ids, -2]
        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"tok{token_id}" for token_id in token_ids if not skip_special_tokens or token_id >= 0)

    @staticmethod
    def num_special_tokens_to_add(pair: bool = False) -> int:
        return 3 if pair else 2


class ExpandingDecodeAlignScoreTokenizer(WhitespaceAlignScoreTokenizer):
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"tok{token_id} expanded" for token_id in token_ids if not skip_special_tokens or token_id >= 0)


class FakeAlignScoreTensor:
    def __init__(self, values: list[list[float]] | list[float]) -> None:
        self.values = values

    def _matrix_values(self) -> list[list[float]]:
        assert self.values and isinstance(self.values[0], list)
        return cast(list[list[float]], self.values)

    @property
    def shape(self) -> tuple[int, ...]:
        if self.values and isinstance(self.values[0], list):
            matrix_values = self._matrix_values()
            return (len(matrix_values), len(matrix_values[0]))
        return (len(self.values),)

    def to(self, _device: str) -> "FakeAlignScoreTensor":
        return self

    def reshape(self, *_shape: int) -> "FakeAlignScoreTensor":
        if self.values and isinstance(self.values[0], list):
            return FakeAlignScoreTensor([item for row in self._matrix_values() for item in row])
        return self

    def tolist(self) -> list[float] | list[list[float]]:
        return self.values

    def __getitem__(self, key: tuple[slice, int]) -> "FakeAlignScoreTensor":
        rows, column = key
        assert rows == slice(None)
        return FakeAlignScoreTensor([row[column] for row in self._matrix_values()])


class FakeAlignScoreNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


class FakeAlignScoreTorch:
    @staticmethod
    def no_grad() -> FakeAlignScoreNoGrad:
        return FakeAlignScoreNoGrad()

    @staticmethod
    def softmax(tensor: FakeAlignScoreTensor, dim: int) -> FakeAlignScoreTensor:
        assert dim == -1
        normalized_rows = []
        for row in tensor._matrix_values():
            denominator = sum(math.exp(value) for value in row)
            normalized_rows.append([math.exp(value) / denominator for value in row])
        return FakeAlignScoreTensor(normalized_rows)

    @staticmethod
    def sigmoid(_tensor: FakeAlignScoreTensor) -> FakeAlignScoreTensor:
        pytest.fail("AlignScore default path should use tri_label_logits, not a generic logits sigmoid")


def _make_fake_alignscore_runtime():  # noqa: C901
    tokenizer_load_calls: list[dict[str, object]] = []
    tokenizer_encode_calls: list[dict[str, object]] = []
    model_load_calls: list[dict[str, object]] = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name_or_path: str, **kwargs: object) -> "FakeTokenizer":
            tokenizer_load_calls.append({"model_name_or_path": model_name_or_path, **kwargs})
            return cls()

        @staticmethod
        def encode(text: str, add_special_tokens: bool = True) -> list[int]:
            token_ids = list(range(len(text.split())))
            if add_special_tokens:
                return [-1, *token_ids, -2]
            return token_ids

        @staticmethod
        def decode(token_ids: list[int], skip_special_tokens: bool = True) -> str:
            return " ".join(f"tok{token_id}" for token_id in token_ids if not skip_special_tokens or token_id >= 0)

        @staticmethod
        def num_special_tokens_to_add(pair: bool = False) -> int:
            return 3 if pair else 2

        def __call__(
            self,
            contexts: list[str],
            claims: list[str],
            **kwargs: object,
        ) -> dict[str, FakeAlignScoreTensor]:
            tokenizer_encode_calls.append({"contexts": contexts, "claims": claims, **kwargs})
            return {"input_ids": FakeAlignScoreTensor([[1, 2], [3, 4]])}

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                model_type="alignscore",
                id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
            )

        @classmethod
        def from_pretrained(cls, model_name_or_path: str, **kwargs: object) -> "FakeModel":
            model_load_calls.append({"model_name_or_path": model_name_or_path, **kwargs})
            return cls()

        def eval(self) -> None:
            return None

        def to(self, _device: str) -> None:
            return None

        def __call__(self, **_encoded: FakeAlignScoreTensor) -> SimpleNamespace:
            return SimpleNamespace(tri_label_logits=FakeAlignScoreTensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))

    return (
        FakeAlignScoreTorch,
        FakeModel,
        FakeTokenizer,
        model_load_calls,
        tokenizer_load_calls,
        tokenizer_encode_calls,
    )


def test_align_score_scores_sentence_claims_against_retrieved_context():
    scorer = MappingAlignScoreScorer(
        responses={
            ("France is in Europe.", "Paris is the capital of France."): 0.25,
            ("Paris is France's capital.", "Paris is the capital of France."): 0.75,
            ("France is in Europe.", "It is in Europe."): 0.25,
            ("Paris is France's capital.", "It is in Europe."): 0.1,
        }
    )

    scores = align_score(
        metric_inputs=[
            MetricInput(
                retrieved_contents=["France is in Europe.", "Paris is France's capital."],
                generated_texts="Paris is the capital of France. It is in Europe.",
            )
        ],
        scorer=scorer,
        batch_size=4,
    )

    assert scores == [pytest.approx(0.5)]
    assert scorer.calls == [
        (
            [
                "France is in Europe.",
                "Paris is France's capital.",
                "France is in Europe.",
                "Paris is France's capital.",
            ],
            [
                "Paris is the capital of France.",
                "Paris is the capital of France.",
                "It is in Europe.",
                "It is in Europe.",
            ],
            4,
        )
    ]


def test_align_score_uses_later_retrieved_passage_when_it_best_supports_claim():
    scorer = MappingAlignScoreScorer(
        responses={
            ("Irrelevant early passage.", "Paris is the capital of France."): 0.05,
            ("Paris is the capital of France.", "Paris is the capital of France."): 0.95,
        }
    )

    scores = align_score(
        metric_inputs=[
            MetricInput(
                retrieved_contents=["Irrelevant early passage.", "Paris is the capital of France."],
                generated_texts="Paris is the capital of France.",
            )
        ],
        scorer=scorer,
        batch_size=4,
    )

    assert scores == [pytest.approx(0.95)]
    assert scorer.calls == [
        (
            ["Irrelevant early passage.", "Paris is the capital of France."],
            ["Paris is the capital of France.", "Paris is the capital of France."],
            4,
        )
    ]


def test_align_score_uses_later_sentence_window_when_passage_is_long():
    early_window = "One. Two. Three. Four. Five."
    later_window = "Paris is the capital of France."
    scorer = MappingAlignScoreScorer(
        responses={
            (early_window, "Paris is the capital of France."): 0.02,
            (later_window, "Paris is the capital of France."): 0.98,
        }
    )

    scores = align_score(
        metric_inputs=[
            MetricInput(
                retrieved_contents=[f"{early_window} {later_window}"],
                generated_texts="Paris is the capital of France.",
            )
        ],
        scorer=scorer,
    )

    assert scores == [pytest.approx(0.98)]
    assert scorer.calls == [
        (
            [early_window, later_window],
            ["Paris is the capital of France.", "Paris is the capital of France."],
            8,
        )
    ]


def test_align_score_min_aggregation_keeps_worst_supported_claim():
    scorer = DummyAlignScoreScorer(responses=[0.91, 0.12])

    scores = align_score(
        metric_inputs=[
            MetricInput(
                retrieved_contents=["Grounded context."],
                generated_texts="Grounded claim. Unsupported claim.",
            )
        ],
        scorer=scorer,
        aggregation="min",
    )

    assert scores == [pytest.approx(0.12)]


def test_align_score_splits_long_single_sentence_context_by_token_budget():
    late_evidence = "late_evidence_marker"
    long_sentence = " ".join([*(f"filler{i}" for i in range(60)), late_evidence]) + "."
    scorer = KeywordAlignScoreScorer(keyword="tok60")

    scores = align_score(
        metric_inputs=[
            MetricInput(
                retrieved_contents=[long_sentence],
                generated_texts="Supported claim.",
            )
        ],
        scorer=scorer,
        max_length=20,
    )

    assert scores == [pytest.approx(0.99)]
    contexts, claims, _batch_size = scorer.calls[0]
    assert len(contexts) > 1
    assert all(len(context.split()) <= 15 for context in contexts)
    assert claims == ["Supported claim."] * len(contexts)


def test_align_score_splits_long_single_sentence_context_at_default_token_budget():
    scorer = KeywordAlignScoreScorer(keyword="tok520")
    long_sentence = " ".join(f"filler{i}" for i in range(700)) + "."

    scores = align_score(
        metric_inputs=[
            MetricInput(
                retrieved_contents=[long_sentence],
                generated_texts="Supported claim.",
            )
        ],
        scorer=scorer,
    )

    assert scores == [pytest.approx(0.99)]
    contexts, claims, _batch_size = scorer.calls[0]
    assert len(contexts) > 1
    assert all(len(context.split()) <= 507 for context in contexts)
    assert any("tok520" in context for context in contexts)
    assert claims == ["Supported claim."] * len(contexts)


def test_align_score_shrinks_decoded_context_windows_that_retokenize_over_budget():
    scorer = KeywordAlignScoreScorer(keyword="tok20")
    scorer.tokenizer = ExpandingDecodeAlignScoreTokenizer()
    long_sentence = " ".join(f"filler{i}" for i in range(40)) + "."

    scores = align_score(
        metric_inputs=[
            MetricInput(
                retrieved_contents=[long_sentence],
                generated_texts="Supported claim.",
            )
        ],
        scorer=scorer,
        max_length=20,
    )

    assert scores == [pytest.approx(0.99)]
    contexts, claims, _batch_size = scorer.calls[0]
    assert len(contexts) > 1
    assert all(len(context.split()) <= 15 for context in contexts)
    assert any("tok20" in context for context in contexts)
    assert claims == ["Supported claim."] * len(contexts)


def test_align_score_rejects_claim_that_exceeds_token_budget():
    scorer = KeywordAlignScoreScorer(keyword="anything")
    over_budget_claim = " ".join(f"claim{i}" for i in range(18)) + "."

    with pytest.raises(ValueError, match="AlignScore claim exceeds the model token budget"):
        align_score(
            metric_inputs=[MetricInput(retrieved_contents=["Context evidence."], generated_texts=over_budget_claim)],
            scorer=scorer,
            max_length=20,
        )

    assert scorer.calls == []


def test_align_score_rejects_claim_that_leaves_no_context_token_budget():
    scorer = KeywordAlignScoreScorer(keyword="anything")
    budget_filling_claim = " ".join(f"claim{i}" for i in range(17)) + "."

    with pytest.raises(ValueError, match="at least one context token"):
        align_score(
            metric_inputs=[MetricInput(retrieved_contents=["Context evidence."], generated_texts=budget_filling_claim)],
            scorer=scorer,
            max_length=20,
        )

    assert scorer.calls == []


def test_align_score_allows_claim_that_leaves_one_context_token_budget():
    scorer = KeywordAlignScoreScorer(keyword="tok0")
    one_context_token_claim = " ".join(f"claim{i}" for i in range(16)) + "."

    scores = align_score(
        metric_inputs=[MetricInput(retrieved_contents=["Context evidence."], generated_texts=one_context_token_claim)],
        scorer=scorer,
        max_length=20,
    )

    assert scores == [pytest.approx(0.99)]
    contexts, claims, _batch_size = scorer.calls[0]
    assert all(len(context.split()) == 1 for context in contexts)
    assert claims == [one_context_token_claim] * len(contexts)


def test_align_score_short_text_keeps_existing_window_scores_with_tokenizer():
    scorer = KeywordAlignScoreScorer(keyword="capital")

    scores = align_score(
        metric_inputs=[
            MetricInput(
                retrieved_contents=["Paris is the capital of France."],
                generated_texts="Paris is the capital of France.",
            )
        ],
        scorer=scorer,
        max_length=30,
    )

    assert scores == [pytest.approx(0.99)]
    assert scorer.calls == [
        (
            ["Paris is the capital of France."],
            ["Paris is the capital of France."],
            8,
        )
    ]


def test_align_score_returns_none_without_context_or_generated_text():
    scorer = DummyAlignScoreScorer(responses=[0.9])

    scores = align_score(
        metric_inputs=[
            MetricInput(generated_texts="Paris is France's capital."),
            MetricInput(retrieved_contents=["Paris is France's capital."]),
            MetricInput(retrieved_contents=["   "], generated_texts="Paris is France's capital."),
        ],
        scorer=scorer,
    )

    assert scores == [None, None, None]
    assert scorer.calls == []


def test_align_score_rejects_unknown_aggregation():
    with pytest.raises(ValueError, match="Unsupported AlignScore aggregation"):
        align_score(
            metric_inputs=[MetricInput(retrieved_contents=["Context"], generated_texts="Claim")],
            scorer=DummyAlignScoreScorer(responses=[0.5]),
            aggregation="median",
        )


def test_align_score_config_exposes_metric_function_and_kwargs():
    config = AlignScoreConfig(
        model_name_or_path="custom-alignscore",
        batch_size=2,
        max_length=256,
        device="cpu",
        cache_dir="custom-cache",
        trust_remote_code=True,
        revision="custom-revision",
        aggregation="min",
    )

    assert config.get_metric_name() == "align_score"
    assert config.get_metric_func() is align_score
    assert config.get_metric_kwargs() == {
        "model_name_or_path": "custom-alignscore",
        "batch_size": 2,
        "max_length": 256,
        "device": "cpu",
        "cache_dir": "custom-cache",
        "aggregation": "min",
        "trust_remote_code": True,
        "revision": "custom-revision",
    }


def test_align_score_shipped_config_pins_revision_and_explicit_trust():
    config_path = Path("configs/metrics/generation/align_score.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["model_name_or_path"] == ALIGNSCORE_MODEL_NAME
    assert config["max_length"] == 512
    assert config["trust_remote_code"] is True
    assert config["revision"] == ALIGNSCORE_REVISION


def test_align_score_default_max_length_matches_pinned_checkpoint_limit():
    assert ALIGNSCORE_MAX_LENGTH == 512


def test_align_score_default_checkpoint_requires_explicit_remote_code_trust():
    import autorag_research.evaluation.metrics.generation as generation_module

    with pytest.raises(ValueError, match="trust_remote_code=True") as exc_info:
        generation_module.HuggingFaceAlignScoreScorer(
            model_name_or_path=ALIGNSCORE_MODEL_NAME,
            trust_remote_code=False,
        )

    assert ALIGNSCORE_MODEL_NAME in str(exc_info.value)


def test_align_score_default_checkpoint_rejects_unpinned_remote_code_revision():
    import autorag_research.evaluation.metrics.generation as generation_module

    with pytest.raises(ValueError, match="pinned commit revision") as exc_info:
        generation_module.HuggingFaceAlignScoreScorer(
            model_name_or_path=ALIGNSCORE_MODEL_NAME,
            trust_remote_code=True,
            revision=None,
        )

    assert ALIGNSCORE_MODEL_NAME in str(exc_info.value)


def test_align_score_custom_model_does_not_inherit_builtin_revision(monkeypatch: pytest.MonkeyPatch):
    import autorag_research.evaluation.metrics.generation as generation_module

    fake_torch, fake_model, fake_tokenizer, model_load_calls, tokenizer_load_calls, _tokenizer_encode_calls = (
        _make_fake_alignscore_runtime()
    )

    monkeypatch.setattr(
        generation_module,
        "_import_alignscore_runtime",
        lambda: (fake_torch, fake_model, fake_tokenizer),
    )

    generation_module.HuggingFaceAlignScoreScorer(model_name_or_path="local/custom-alignscore")

    assert model_load_calls == [
        {
            "model_name_or_path": "local/custom-alignscore",
            "cache_dir": None,
            "trust_remote_code": False,
            "revision": None,
        }
    ]
    assert tokenizer_load_calls == [
        {
            "model_name_or_path": "local/custom-alignscore",
            "cache_dir": None,
            "trust_remote_code": False,
            "revision": None,
        }
    ]


def test_align_score_default_huggingface_scorer_uses_alignscore_remote_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    import autorag_research.evaluation.metrics.generation as generation_module

    fake_torch, fake_model, fake_tokenizer, model_load_calls, tokenizer_load_calls, tokenizer_encode_calls = (
        _make_fake_alignscore_runtime()
    )

    monkeypatch.setattr(
        generation_module,
        "_import_alignscore_runtime",
        lambda: (fake_torch, fake_model, fake_tokenizer),
    )

    scorer = generation_module.HuggingFaceAlignScoreScorer(
        model_name_or_path="liuyanyi/AlignScore-large-hf",
        trust_remote_code=True,
        revision=ALIGNSCORE_REVISION,
    )
    scores = scorer.score(["context 1", "context 2"], ["claim 1", "claim 2"], batch_size=2)

    assert model_load_calls == [
        {
            "model_name_or_path": "liuyanyi/AlignScore-large-hf",
            "cache_dir": None,
            "trust_remote_code": True,
            "revision": ALIGNSCORE_REVISION,
        }
    ]
    assert tokenizer_load_calls == [
        {
            "model_name_or_path": "liuyanyi/AlignScore-large-hf",
            "cache_dir": None,
            "trust_remote_code": True,
            "revision": ALIGNSCORE_REVISION,
        }
    ]
    assert tokenizer_encode_calls == [
        {
            "contexts": ["context 1", "context 2"],
            "claims": ["claim 1", "claim 2"],
            "max_length": 512,
            "truncation": "only_first",
            "padding": "max_length",
            "return_tensors": "pt",
        }
    ]
    assert scores == [pytest.approx(0.786986), pytest.approx(0.106507)]


def test_align_score_default_huggingface_scorer_rejects_over_budget_claim_before_tokenization(
    monkeypatch: pytest.MonkeyPatch,
):
    import autorag_research.evaluation.metrics.generation as generation_module

    fake_torch, fake_model, fake_tokenizer, _model_load_calls, _tokenizer_load_calls, tokenizer_encode_calls = (
        _make_fake_alignscore_runtime()
    )

    monkeypatch.setattr(
        generation_module,
        "_import_alignscore_runtime",
        lambda: (fake_torch, fake_model, fake_tokenizer),
    )

    scorer = generation_module.HuggingFaceAlignScoreScorer(
        model_name_or_path="liuyanyi/AlignScore-large-hf",
        trust_remote_code=True,
        revision=ALIGNSCORE_REVISION,
    )

    with pytest.raises(ValueError, match="AlignScore claim exceeds the model token budget"):
        scorer.score(["short context"], [" ".join(f"claim{i}" for i in range(510))], batch_size=1)

    assert tokenizer_encode_calls == []


def test_align_score_runtime_guard_points_to_optional_dependencies(monkeypatch: pytest.MonkeyPatch):
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
        generation_module._import_alignscore_runtime()

    assert "uv sync --all-extras --all-groups" in str(exc_info.value)
