# Custom Dataset

Ingest your own dataset for benchmarking.

## Dataset Requirements

Your dataset needs:

1. **Documents**: Text or images to search
2. **Queries**: Questions to answer
3. **Ground Truth**: Which documents answer each query

## Format

```python
corpus = {
    "doc_1": "Document text here...",
    "doc_2": "Another document...",
}

queries = {
    "q_1": "What is the capital of France?",
    "q_2": "How does photosynthesis work?",
}

qrels = {
    "q_1": ["doc_1"],  # doc_1 answers q_1
    "q_2": ["doc_2", "doc_5"],  # multiple relevant docs
}
```

## Create Ingestor

```python
from autorag_research.data.ingestor import TextEmbeddingDataIngestor, register_ingestor
from typing import Literal

@register_ingestor(name="my_dataset", description="My custom dataset")
class MyDatasetIngestor(TextEmbeddingDataIngestor):
    def __init__(
        self,
        embedding_model,
        split: Literal["train", "test"] = "test",
    ):
        super().__init__(embedding_model)
        self.split = split

    def detect_primary_key_type(self):
        return "string"  # or "bigint" for integer IDs

    def ingest(self):
        # Load your data
        corpus = self._load_corpus()
        queries = self._load_queries()
        qrels = self._load_qrels()

        # Add to database
        for doc_id, text in corpus.items():
            self._service.add_chunk(doc_id, text)

        for query_id, query_text in queries.items():
            self._service.add_query(query_id, query_text)

        for query_id, doc_ids in qrels.items():
            self._service.add_retrieval_gt(query_id, doc_ids)
```

## Ingest

```bash
autorag-research ingest \
  --name=my_dataset \
  --extra split=test \
  --embedding-model=openai-small
```

## Run Benchmark

```yaml
# configs/experiment.yaml
db_name: my_dataset_test_openai_small

pipelines:
  retrieval: [bm25]
metrics:
  retrieval: [recall, ndcg]
```

```bash
autorag-research run --config-name=experiment
```

## Share Dataset

Export and upload to HuggingFace Hub:

```bash
autorag-research data dump --db-name=my_dataset_test_openai_small
autorag-research data upload ./dump.file my_dataset openai-small
```

## Next

- [Custom Pipeline](custom-pipeline.md) - Test your algorithm
- [Datasets](../datasets/index.md) - See existing formats
