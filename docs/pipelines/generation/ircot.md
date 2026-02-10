# IRCoT

Interleaving Retrieval with Chain-of-Thought reasoning for multi-step question answering.

## Overview

| Field | Value |
|-------|-------|
| Type | Generation |
| Algorithm | Iterative Retrieve + Reason |
| Modality | Text |
| Paper | [ACL 2023](https://aclanthology.org/2023.acl-long.557/) |

## How It Works

IRCoT alternates between reasoning and retrieval in an iterative loop:

```
┌─────────────────────────────────────────────────────┐
│  1. Initial Retrieval                               │
│     Query → Retrieve k paragraphs                   │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  2. Iterative Loop (up to max_steps)                │
│                                                     │
│     a) Generate CoT sentence (reasoning step)       │
│     b) Check: contains "answer is:"? → Exit         │
│     c) Use CoT sentence as query → Retrieve more    │
│     d) Add to paragraph collection (cap at budget)  │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  3. Final QA Generation                             │
│     All paragraphs + question → Final answer        │
└─────────────────────────────────────────────────────┘
```

**Key insight**: Each reasoning step guides the next retrieval, and each retrieval informs subsequent reasoning, creating a symbiotic improvement cycle.

## Configuration

```yaml
_target_: autorag_research.pipelines.generation.ircot.IRCoTGenerationPipelineConfig
name: ircot
retrieval_pipeline_name: bm25
llm: gpt-4o-mini
k_per_step: 4
max_steps: 8
paragraph_budget: 15
stop_sequence: "answer is:"
top_k: 4
batch_size: 10
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| name | str | required | Unique pipeline instance name |
| retrieval_pipeline_name | str | required | Name of retrieval pipeline (BM25 recommended) |
| llm | str or BaseLLM | required | LLM instance or config name |
| k_per_step | int | 4 | Paragraphs to retrieve per reasoning step |
| max_steps | int | 8 | Maximum reasoning-retrieval iterations |
| paragraph_budget | int | 15 | Maximum total paragraphs to collect |
| stop_sequence | str | "answer is:" | Termination string (case-insensitive) |
| reasoning_prompt_template | str | default | Template for reasoning steps |
| qa_prompt_template | str | default | Template for final QA |
| top_k | int | 4 | Alias for k_per_step |
| batch_size | int | 10 | Queries per batch |

## Prompt Template Variables

### Reasoning Prompt

| Variable | Description |
|----------|-------------|
| `{query}` | Original question |
| `{paragraphs}` | Retrieved paragraphs (numbered) |
| `{cot_history}` | Previous reasoning steps |

### QA Prompt

| Variable | Description |
|----------|-------------|
| `{query}` | Original question |
| `{paragraphs}` | All collected paragraphs |

## Custom Prompt Templates

```yaml
reasoning_prompt_template: |
  Question: {query}

  Context:
  {paragraphs}

  Previous reasoning:
  {cot_history}

  Think step-by-step. Write "The answer is: X" when ready.

qa_prompt_template: |
  Question: {query}

  Documents:
  {paragraphs}

  Answer concisely:
```

## Algorithm Details

### Termination Conditions

The iterative loop terminates when:

1. **Answer detected**: Generated CoT contains "answer is:" (case-insensitive)
2. **Max steps reached**: Completed `max_steps` iterations

### Paragraph Budget

- Paragraphs are capped at `paragraph_budget` using FIFO strategy
- Earlier paragraphs (from initial retrieval) are retained
- Prevents unbounded context growth

### First Sentence Extraction

Only the first sentence from each CoT generation is kept:

- Prevents runaway generation
- Keeps reasoning steps focused
- Matches original paper implementation

## Performance

From the original paper (GPT-3 + BM25):

| Dataset | Retrieval Recall | QA Accuracy |
|---------|------------------|-------------|
| HotpotQA | +21 points vs OneR | +15 points |
| 2WikiMultihopQA | +18 points | +12 points |
| MuSiQue | +15 points | +10 points |

## When to Use

**Good for:**

- Multi-hop reasoning questions
- Questions requiring information synthesis
- Complex factual queries
- Knowledge-intensive tasks

**Consider BasicRAG for:**

- Simple single-fact questions
- Low-latency requirements (IRCoT makes multiple LLM calls)
- Cost-sensitive applications

## Example Usage

### Python

```python
from langchain_openai import ChatOpenAI

from autorag_research.orm.connection import DBConnection
from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline
from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline

db = DBConnection.from_config()
session_factory = db.get_session_factory()

# Create retrieval pipeline
retrieval = BM25RetrievalPipeline(
    session_factory=session_factory,
    name="bm25_retriever",
)

# Create IRCoT pipeline
pipeline = IRCoTGenerationPipeline(
    session_factory=session_factory,
    name="ircot_gpt4",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    retrieval_pipeline=retrieval,
    k_per_step=4,
    max_steps=8,
    paragraph_budget=15,
)

# Run on all queries
results = pipeline.run(top_k=4)
print(f"Processed {results['total_queries']} queries")
```

### Single Query

```python
result = pipeline._generate("What is the relationship between X and Y?", top_k=4)
print(f"Answer: {result.text}")
print(f"Steps taken: {result.metadata['steps']}")
print(f"CoT history: {result.metadata['cot_sentences']}")
```

## Output Metadata

The `GenerationResult.metadata` contains:

| Field | Type | Description |
|-------|------|-------------|
| `cot_sentences` | list[str] | Chain-of-thought reasoning history |
| `chunk_ids` | list[int] | IDs of all retrieved chunks |
| `steps` | int | Number of reasoning steps completed |

## References

- Paper: [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://aclanthology.org/2023.acl-long.557/)
- arXiv: [2212.10509](https://arxiv.org/abs/2212.10509)
- Code: [StonyBrookNLP/ircot](https://github.com/StonyBrookNLP/ircot)

## Citation
```bibtex
@inproceedings{trivedi2023interleaving,
  title={Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions},
  author={Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
  booktitle={Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: long papers)},
  pages={10014--10037},
  year={2023}
}
```
