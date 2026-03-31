# AutoThinkRAG

Complexity-aware generation pipeline with adaptive retrieval, reranking, and reasoning depth.

## Overview

| Field | Value |
|-------|-------|
| Type | Generation |
| Algorithm | Query Complexity Routing + Adaptive Retrieval + Multi-step Reasoning |
| Modality | Text + Vision (optional) |
| Paper | [arxiv:2603.05551](https://arxiv.org/abs/2603.05551) |

## How It Works

AutoThinkRAG classifies each query by complexity and routes it through a differentiated pipeline:

```
┌──────────────────────────────────────────────────────────────┐
│  1. QCR — Query Complexity Router                            │
│     LLM classifies query → simple / moderate / complex       │
└──────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌────────────┐  ┌───────────────┐  ┌──────────────────┐
     │   SIMPLE   │  │   MODERATE    │  │     COMPLEX      │
     │            │  │               │  │                  │
     │ retrieve   │  │ retrieve      │  │ decompose query  │
     │ (top_k)    │  │ (fetch_k)     │  │ multi-retrieve   │
     │            │  │ rerank → top_k│  │ merge + rerank   │
     └─────┬──────┘  └──────┬────────┘  └────────┬─────────┘
           │                │                     │
           ▼                ▼                     ▼
     ┌────────────┐  ┌───────────────┐  ┌──────────────────┐
     │ Direct     │  │ Synthesis     │  │ Iterative        │
     │ answer     │  │ prompt with   │  │ reasoning loop   │
     │ generation │  │ reasoning     │  │ (N steps) +      │
     │            │  │               │  │ follow-up retr.  │
     └────────────┘  └───────────────┘  └──────────────────┘
```

**Key insight**: Not all queries need the same retrieval depth or reasoning effort. Simple factual lookups skip reranking and decomposition entirely, while complex multi-hop questions get sub-query decomposition, broader retrieval, reranking, and iterative reasoning.

## Configuration

```yaml
_target_: autorag_research.pipelines.generation.autothinkrag.AutoThinkRAGPipelineConfig
name: autothinkrag
retrieval_pipeline_name: hybrid_rrf
llm: gpt-4o
vlm: gpt-4o              # optional, for document images
reranker: flashrank       # optional, enables reranking for moderate/complex
max_reasoning_steps: 3
max_subquestions: 3
fetch_k_multiplier: 2
temperature: 0.0
top_k: 10
batch_size: 10
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| name | str | required | Unique pipeline instance name |
| retrieval_pipeline_name | str | required | Base retrieval pipeline to use |
| llm | str or BaseLLM | required | Main LLM for routing and generation |
| vlm | str or BaseChatModel | None | Vision LM for image interpretation (DPR) |
| reranker | str or BaseReranker | None | Reranker for moderate/complex paths |
| max_reasoning_steps | int | 3 | Max iterative reasoning steps (complex path) |
| max_subquestions | int | 3 | Max sub-questions for query decomposition |
| fetch_k_multiplier | int | 2 | Over-fetch multiplier when reranker is set |
| temperature | float | 0.0 | LLM generation temperature |
| max_tokens | int | None | Max generation tokens |
| complexity_tiers | list[str] | ["simple", "moderate", "complex"] | Tier labels |
| complexity_prompt_template | str | default | Prompt for complexity classification |
| simple_prompt_template | str | default | Prompt for simple-tier generation |
| moderate_prompt_template | str | default | Prompt for moderate-tier generation |
| complex_prompt_template | str | default | Prompt for complex-tier reasoning |
| decomposition_prompt_template | str | default | Prompt for sub-query decomposition |
| visual_interpretation_prompt_template | str | default | Prompt for VLM interpretation |

## Complexity Routing Paths

### Simple

- **Retrieval**: Single `retrieve(top_k)` — no over-fetching
- **Reranking**: Skipped (even if reranker is configured)
- **Generation**: Direct answer from context

### Moderate

- **Retrieval**: Single `retrieve(top_k * fetch_k_multiplier)` when reranker is set
- **Reranking**: Applied — narrows candidates to `top_k`
- **Generation**: Synthesis-oriented prompt that asks for reasoning steps

### Complex

- **Decomposition**: LLM breaks query into sub-questions (deduplicated)
- **Retrieval**: Original query + each sub-question retrieved separately, results merged
- **Reranking**: Applied to merged results — narrows to `top_k`
- **Reasoning**: Iterative loop (up to `max_reasoning_steps`):
    - Generate a reasoning step
    - Use reasoning text as follow-up retrieval query
    - Accumulate context across iterations
- **Generation**: Final answer from accumulated context + full reasoning chain

## Output Metadata

The `GenerationResult.metadata` contains:

| Field | Type | Present | Description |
|-------|------|---------|-------------|
| `complexity_tier` | str | Always | Routed tier: simple, moderate, or complex |
| `retrieved_chunk_ids` | list[int] | Always | Final set of chunk IDs used |
| `retrieved_scores` | list[float] | Always | Corresponding relevance scores |
| `visual_interpretation` | str | If VLM + images | Textual description of visual content |
| `sub_queries` | list[str] | Complex only | Decomposed sub-questions |
| `reasoning_steps` | list[str] | Complex only | Chain of reasoning steps |

## Example Usage

### Python

```python
from langchain_openai import ChatOpenAI

from autorag_research.orm.connection import DBConnection
from autorag_research.pipelines.generation.autothinkrag import AutoThinkRAGPipeline
from autorag_research.pipelines.retrieval.hybrid import HybridRRFRetrievalPipeline
from autorag_research.rerankers.flashrank import FlashRankReranker

db = DBConnection.from_config()
session_factory = db.get_session_factory()

# Retrieval pipeline (any type works)
retrieval = HybridRRFRetrievalPipeline(
    session_factory=session_factory,
    name="hybrid_retriever",
)

# AutoThinkRAG with reranker
pipeline = AutoThinkRAGPipeline(
    session_factory=session_factory,
    name="autothinkrag_gpt4",
    llm=ChatOpenAI(model="gpt-4o"),
    retrieval_pipeline=retrieval,
    reranker=FlashRankReranker(),
    max_reasoning_steps=3,
    max_subquestions=3,
)

# Run on all queries
results = pipeline.run(top_k=10)
print(f"Processed {results['total_queries']} queries")
```

### Single Query

```python
result = await pipeline._generate(query_id=1, top_k=10)
print(f"Tier: {result.metadata['complexity_tier']}")
print(f"Answer: {result.text}")
if result.metadata.get("reasoning_steps"):
    for i, step in enumerate(result.metadata["reasoning_steps"], 1):
        print(f"  Step {i}: {step}")
```

---

## Relationship to the Original Paper

This implementation is based on [AutoThinkRAG: Complexity-Aware Control of Retrieval-Augmented Reasoning for Image-Text Interaction](https://arxiv.org/abs/2603.05551) (2025). We implement the paper's **core algorithmic contributions** while adapting the infrastructure components to fit the AutoRAG-Research framework.

### What We Implement from the Paper

| Paper Contribution | Our Implementation |
|---|---|
| **QCR (Query Complexity Router)** — classify queries into Simple/Moderate/Complex and route to differentiated paths | LLM-based router with three distinct retrieval and generation paths |
| **DPR (Decomposition of Perception and Reasoning)** — separate visual interpretation from logical reasoning | VLM generates textual descriptions; LLM reasons over text only |
| **Adaptive retrieval depth** — simple queries use lightweight retrieval, complex queries use deeper search | Simple skips reranking; Moderate adds reranking; Complex adds sub-query decomposition + multi-retrieval + reranking |
| **Adaptive reasoning depth** — reasoning effort scales with complexity | Simple = direct answer; Moderate = synthesis prompt; Complex = iterative multi-step reasoning with follow-up retrieval |
| **Sub-query decomposition** — break complex queries into simpler parts for broader evidence gathering | LLM decomposes query, retrieves per sub-question, merges and deduplicates results |

### What Differs from the Paper

| Paper Component | Paper's Approach | Our Approach | Why |
|---|---|---|---|
| **QCR Router model** | Dedicated 3B SLM (Qwen2.5-VL-3B) extracting semantic/element/dependency features | Main LLM with classification prompt | Avoids requiring a separate model deployment; the routing logic (3-tier classification) is preserved |
| **Knowledge Base** | Graph Knowledge Base (GKB) with entity disambiguation + vector store dual storage | Pluggable retrieval pipeline (vector, BM25, hybrid RRF/CC) | AutoRAG-Research provides composable retrieval pipelines that achieve the same goal (broad, deep retrieval) without requiring graph infrastructure |
| **Complex retrieval** | Hypergraph modeling for high-order multi-hop dependencies | Sub-query decomposition + multi-retrieval + merge + rerank | Hypergraph requires specialized graph infrastructure; decomposition + reranking is a practical alternative that broadens evidence coverage |
| **Offline IE pipeline** | MinerU parsing + entity/relation extraction + VLM captioning at ingestion | Standard chunk-based ingestion with optional image storage | The paper's IE pipeline is an ingestion-time concern orthogonal to the generation algorithm |
| **Reranker** | bge-reranker-v2-m3 | Any `BaseReranker` implementation (15+ options) | Framework provides pluggable rerankers; users choose based on their needs |

### Why This Is Still AutoThinkRAG

The paper's two core contributions as stated by the authors are:

1. **QCR**: "We design a Query Complexity Router (QCR) module, which adaptively controls the retrieval search scope and reasoning depth for queries at different complexity levels." (Section 3.2)

2. **DPR**: "We propose a Decomposition of Perception and Reasoning (DPR) strategy that explicitly decomposes visual perception and logical reasoning." (Section 3.3)

Both are fully implemented here. The GKB and hypergraph are **retrieval backend choices** — the paper's Equations 1-3 are agnostic to the retrieval mechanism:

```
Eq.1: A = LLM(T_v, R, I_p)        — R is any retrieval result
Eq.2: T_v = VLM_small(v, P)        — DPR visual perception
Eq.3: A = LLM(C, Instruction(c(Q)))— QCR-guided adaptive reasoning
```

Our implementation preserves the algorithmic structure (complexity-aware routing with differentiated retrieval and reasoning paths) while substituting infrastructure components with the equivalents available in AutoRAG-Research.

## References

- Paper: [AutoThinkRAG: Complexity-Aware Control of Retrieval-Augmented Reasoning for Image-Text Interaction](https://arxiv.org/abs/2603.05551)
- arXiv: [2603.05551](https://arxiv.org/abs/2603.05551)
