from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_openai import OpenAIEmbeddings

from autorag_research import cli
from autorag_research.evaluation.metrics.generation import (
    AnswerCorrectnessF1Config,
    GroundedRefusalF1Config,
    answer_correctness_f1,
    bert_score,
    bleu,
    grounded_refusal_f1,
    meteor,
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


def test_dataset_level_metric_config_granularity():
    grounded_refusal_cfg = GroundedRefusalF1Config()
    answer_correctness_cfg = AnswerCorrectnessF1Config()

    assert grounded_refusal_cfg.get_compute_granularity() == "dataset"
    assert answer_correctness_cfg.get_compute_granularity() == "dataset"


def test_grounded_refusal_f1_dataset_level():
    metric_inputs = [
        MetricInput(
            query="q1",
            generated_texts="I apologize, but I couldn't find an answer to your question in the search results.",
            generation_gt=None,
        ),
        MetricInput(
            query="q2",
            generated_texts="Paris is the capital of France.",
            generation_gt=["Paris is the capital of France."],
        ),
        MetricInput(
            query="q3",
            generated_texts="I apologize, but I couldn't find an answer to your question in the search results.",
            generation_gt=["The sun rises in the east."],
        ),
        MetricInput(
            query="q4",
            generated_texts="Here is a non-refusal answer.",
            generation_gt=None,
        ),
    ]

    scores = grounded_refusal_f1(
        metric_inputs=metric_inputs,
        judge_mode="phrase",
    )

    assert len(scores) == len(metric_inputs)
    assert all(score == pytest.approx(0.5) for score in scores)


def test_answer_correctness_f1_dataset_level():
    metric_inputs = [
        MetricInput(
            query="q1",
            generated_texts="alpha claim is true.",
            generation_gt=["alpha claim", "beta claim"],
            retrieval_gt_contents=[["Document evidence: alpha claim is supported."]],
        ),
        MetricInput(
            query="q2",
            generated_texts="This misses the expected claim.",
            generation_gt=["gamma claim"],
            retrieval_gt_contents=[["Document evidence: gamma claim is supported."]],
        ),
        MetricInput(
            query="q3",
            generated_texts="Here is an over-responsive answer.",
            generation_gt=None,
            retrieval_gt_contents=None,
        ),
    ]

    scores = answer_correctness_f1(
        metric_inputs=metric_inputs,
        judge_mode="phrase",
    )

    assert len(scores) == len(metric_inputs)
    assert all(score == pytest.approx(0.4) for score in scores)


def test_grounded_refusal_f1_with_llm_string_config():
    metric_inputs = [
        MetricInput(
            query="q1",
            generated_texts="This is a normal answer.",
            generation_gt=["normal answer"],
        )
    ]

    scores = grounded_refusal_f1(
        metric_inputs=metric_inputs,
        judge_llm="mock",
        judge_mode="llm",
    )

    assert len(scores) == 1
    assert isinstance(scores[0], float)
    assert 0.0 <= scores[0] <= 1.0
