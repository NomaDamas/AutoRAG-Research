# Generation Pipelines

Algorithms that take a query and retrieved documents to generate an answer.

## Available Pipelines

| Pipeline | Algorithm |
|----------|-----------|
| [BasicRAG](basic-rag.md) | Single retrieve + generate |
| [IRCoT](ircot.md) | Iterative retrieve + chain-of-thought reasoning |

## Base Class

All generation pipelines extend `BaseGenerationPipeline`:

```python
from autorag_research.pipelines.generation import (
    BaseGenerationPipeline,
    GenerationResult,
)


class MyRAGPipeline(BaseGenerationPipeline):
    def _generate(self, query: str, top_k: int) -> GenerationResult:
        # 1. Retrieve documents
        retrieved = self._retrieval_pipeline.retrieve(query, top_k)

        # 2. Build context and generate
        answer = self._llm.complete(...)

        return GenerationResult(
            text=answer.text,
            token_usage={"prompt": 100, "completion": 50},
            metadata={},
        )

    def _get_pipeline_config(self):
        return {"type": "my_rag"}
```

## Composition

Generation pipelines compose with retrieval pipelines:

```yaml
_target_: autorag_research.pipelines.generation.basic_rag.BasicRAGPipelineConfig
name: my_rag
retrieval_pipeline_name: bm25  # References a retrieval pipeline
llm: gpt-4o-mini
```

## Methods

| Method | Description |
|--------|-------------|
| `run(top_k, batch_size)` | Batch generation for all queries |
