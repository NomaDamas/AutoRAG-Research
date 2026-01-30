# Text RAG Benchmark

Run full RAG: retrieval + generation with LLM.

## Prerequisites

- LLM API key (OpenAI, Anthropic, or local model)
- Dataset with generation ground truth (RAGBench recommended)

## Download Dataset

```bash
autorag-research data restore ragbench covidqa_openai-small
```

RAGBench includes generation ground truth (expected answers).

## Create Experiment Config

```yaml
# configs/experiment.yaml
db_name: ragbench_covidqa_test_openai_small

pipelines:
  retrieval:
    - bm25
  generation:
    - basic_rag

metrics:
  retrieval:
    - recall
    - ndcg
  generation:
    - rouge
    - bleu
    - bert_score
```

## Configure LLM

```yaml
# configs/pipelines/generation/basic_rag.yaml
_target_: autorag_research.pipelines.generation.basic_rag.BasicRAGPipelineConfig
name: basic_rag
retrieval_pipeline_name: bm25
llm: gpt-4o-mini
prompt_template: |
  Context:
  {context}

  Question: {question}

  Answer:
top_k: 5
```

## Run

```bash
autorag-research run --config-name=experiment
```

## Expected Output

```
Pipeline: bm25
  Recall@10: 0.823
  NDCG@10: 0.698

Pipeline: basic_rag
  ROUGE-L: 0.412
  BLEU: 0.287
  BERTScore: 0.856
```

## Recommended Datasets

| Dataset | Has Generation GT |
|---------|-------------------|
| RAGBench | Yes |
| BEIR | No (retrieval only) |
| MTEB | No (retrieval only) |

## Next

- [Multimodal](multimodal.md) - Visual documents
- [Custom Metric](custom-metric.md) - Add evaluation
