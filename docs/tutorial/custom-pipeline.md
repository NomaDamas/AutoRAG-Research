# Custom Pipeline

Implement your own retrieval or generation algorithm.

## Retrieval Pipeline

```python
from autorag_research.pipelines.retrieval import (
    BaseRetrievalPipeline,
    BaseRetrievalPipelineConfig,
)
from dataclasses import dataclass


@dataclass
class MyRetrievalConfig(BaseRetrievalPipelineConfig):
    name: str = "my_retrieval"
    custom_param: float = 0.5

    def get_pipeline_class(self):
        return MyRetrievalPipeline

    def get_pipeline_kwargs(self):
        return {"custom_param": self.custom_param}


class MyRetrievalPipeline(BaseRetrievalPipeline):
    def __init__(self, session_factory, name, schema, custom_param):
        super().__init__(session_factory, name, schema)
        self.custom_param = custom_param

    def _get_retrieval_func(self):
        def retrieve(queries: list[str], top_k: int) -> list[list[dict]]:
            results = []
            for query in queries:
                # Your retrieval logic here
                docs = self._search(query, top_k)
                results.append([{"doc_id": d.id, "score": d.score} for d in docs])
            return results

        return retrieve

    def _get_pipeline_config(self):
        return {"type": "my_retrieval", "custom_param": self.custom_param}
```

## Generation Pipeline

```python
from autorag_research.pipelines.generation import (
    BaseGenerationPipeline,
    BaseGenerationPipelineConfig,
    GenerationResult,
)
from dataclasses import dataclass


@dataclass
class MyRAGConfig(BaseGenerationPipelineConfig):
    name: str = "my_rag"
    retrieval_pipeline_name: str = "bm25"

    def get_pipeline_class(self):
        return MyRAGPipeline


class MyRAGPipeline(BaseGenerationPipeline):
    def _generate(self, query: str, top_k: int) -> GenerationResult:
        # Step 1: Retrieve documents
        retrieved = self._retrieval_pipeline.retrieve(query, top_k)

        # Step 2: Build context
        context = self._build_context(retrieved)

        # Step 3: Generate answer
        answer = self._llm.complete(f"Context: {context}\nQuestion: {query}")

        return GenerationResult(
            text=answer.text,
            token_usage={"prompt": 100, "completion": 50},
            metadata={"retrieved_ids": [r["doc_id"] for r in retrieved]},
        )

    def _get_pipeline_config(self):
        return {"type": "my_rag"}
```

## Add Configuration

```yaml
# configs/pipelines/retrieval/my_retrieval.yaml
_target_: my_module.MyRetrievalConfig
name: my_retrieval
custom_param: 0.7
```

## Benchmark Against Baselines

```yaml
# configs/experiment.yaml
pipelines:
  retrieval:
    - bm25           # baseline
    - my_retrieval   # your algorithm
metrics:
  retrieval: [recall, ndcg, mrr]
```

```bash
autorag-research run --config-name=experiment
```

## Next

- [Custom Metric](custom-metric.md) - Add evaluation
- [Pipelines](../pipelines/index.md) - See existing implementations
