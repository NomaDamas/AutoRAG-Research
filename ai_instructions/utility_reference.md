# Utility & Service Reference Catalog

**MANDATORY READING** before implementing any pipeline, ingestor, or evaluation code.
If your code does something listed here, you MUST use the existing function instead of reimplementing it.

---

## Quick Lookup Table

| Need to... | Use | Location |
|---|---|---|
| Limit async concurrency | `run_with_concurrency_limit()` | `util.py` |
| Run sync func as async | `to_async_func()` | `util.py` |
| Normalize scores (min-max) | `normalize_minmax()` | `util.py` |
| Normalize scores (z-score) | `normalize_zscore()` | `util.py` |
| Normalize scores (3-sigma) | `normalize_dbsf()` | `util.py` |
| Normalize with known min | `normalize_tmm()` | `util.py` |
| Truncate text to token limit | `truncate_texts()` | `util.py` |
| Normalize string (SQuAD-style) | `normalize_string()` | `util.py` |
| Convert collections to lists | `to_list()` | `util.py` |
| Decorator: convert inputs to lists | `convert_inputs_to_list()` | `util.py` |
| Flatten sublists, run func, regroup | `unpack_and_run()` | `util.py` |
| Convert PIL image to bytes | `pil_image_to_bytes()` | `util.py` |
| Parse base64 data URI | `extract_image_from_data_uri()` | `util.py` |
| Sum token usage from results | `aggregate_token_usage()` | `util.py` |
| BM25 search by query IDs | `service.bm25_search()` | `RetrievalPipelineService` |
| BM25 search by raw text | `service.bm25_search_by_text()` | `RetrievalPipelineService` |
| Vector search by query IDs | `service.vector_search()` | `RetrievalPipelineService` |
| Vector search by embedding | `service.vector_search_by_embedding()` | `RetrievalPipelineService` |
| Get query text by ID | `service.get_query_text()` | `GenerationPipelineService` |
| Batch fetch query texts | `service.fetch_query_texts()` | `RetrievalPipelineService` |
| Get chunk contents by IDs | `service.get_chunk_contents()` | `GenerationPipelineService` |
| Add chunks to DB | `service.add_chunks()` | `BaseIngestionService` |
| Add queries to DB | `service.add_queries()` | `BaseIngestionService` |
| Embed all queries | `service.embed_all_queries()` | `BaseIngestionService` |
| Embed all chunks | `service.embed_all_chunks()` | `BaseIngestionService` |
| Add retrieval ground truth | `service.add_retrieval_gt()` | `BaseIngestionService` |
| Get or create metric | `service.get_or_create_metric()` | `BaseEvaluationService` |
| Run evaluation pipeline | `service.evaluate()` | `BaseEvaluationService` |

---

## Utility Functions (`autorag_research/util.py`)

### Data Conversion

```python
def to_list(item: Any) -> Any
```
Recursively convert numpy arrays, pandas Series, and iterables to Python lists.

```python
def convert_inputs_to_list(func: Callable) -> Callable
```
Decorator that converts all function inputs to Python lists via `to_list()`.

### Text Processing

```python
def truncate_texts(str_list: list[str], max_tokens: int) -> list[str]
```
Truncate each string to a maximum number of tokens using tiktoken (`cl100k_base` encoding).

```python
def normalize_string(s: str) -> str
```
SQuAD-style normalization: lowercase, remove punctuation, remove articles, fix whitespace.

### Score Normalization

All normalizers handle `None` values (preserve them, exclude from statistics).

```python
def normalize_minmax(scores: list[float | None]) -> list[float | None]
```
Min-max normalization to [0, 1]. Returns 0.5 for equal scores.

```python
def normalize_tmm(scores: list[float | None], theoretical_min: float) -> list[float | None]
```
Theoretical min-max: uses known minimum (e.g., 0 for BM25, -1 for cosine) and actual max.

```python
def normalize_zscore(scores: list[float | None]) -> list[float | None]
```
Z-score standardization (mean=0, std=1). Returns 0.0 for equal scores.

```python
def normalize_dbsf(scores: list[float | None]) -> list[float | None]
```
3-sigma distribution-based score fusion. Clips to [0, 1]. Robust to outliers.

### Async Helpers

```python
async def run_with_concurrency_limit(
    items: Iterable[T],
    async_func: Callable[[T], Awaitable[R]],
    max_concurrency: int,
    error_message: str = "Task failed",
) -> list[R | None]
```
Run async function on items with semaphore-based concurrency control. Returns `None` for failed items.

```python
def to_async_func(func: Callable[..., R]) -> Callable[..., Awaitable[R]]
```
Convert sync function to async via `asyncio.to_thread`. No-op if already async.

### Data Structure

```python
def unpack_and_run(
    target_list: list[list[Any]],
    func: Callable,
    *args: tuple,
    **kwargs: Any,
) -> list[Any]
```
Flatten sublists, run func on the flat list, then regroup results by original lengths.

### Image Processing

```python
def pil_image_to_bytes(image: Image.Image) -> tuple[bytes, str]
```
Convert PIL image to `(bytes, mimetype)`. Uses PNG for RGBA/LA/P modes, JPEG otherwise.

```python
def extract_image_from_data_uri(data_uri: str) -> tuple[bytes, str]
```
Extract `(bytes, mimetype)` from a `data:<mime>;base64,<data>` URI string.

### Aggregation

```python
def aggregate_token_usage(results: list[dict]) -> tuple[int, int, int, int]
```
Sum `(prompt_tokens, completion_tokens, embedding_tokens, execution_time_ms)` from result dicts containing `token_usage` and `execution_time` keys.

---

## Service Layer Methods

### RetrievalPipelineService (`orm/service/retrieval_pipeline.py`)

| Method | Signature | Purpose |
|---|---|---|
| `save_pipeline` | `(name: str, config: dict) -> int` | Create pipeline, return ID |
| `run_pipeline` | `(retrieval_func, pipeline_id, top_k=10, batch_size=128, max_concurrency=16, max_retries=3, retry_delay=1.0) -> dict` | Batch retrieval with retry |
| `get_pipeline_config` | `(pipeline_id: int) -> dict \| None` | Get pipeline config |
| `delete_pipeline_results` | `(pipeline_id: int) -> int` | Delete results for pipeline |
| `bm25_search` | `(query_ids, top_k=10, tokenizer="bert", index_name="idx_chunk_bm25") -> list[list[dict]]` | BM25 search by query IDs |
| `bm25_search_by_text` | `(query_text, top_k=10, tokenizer="bert", index_name="idx_chunk_bm25") -> list[dict]` | BM25 search by raw text |
| `vector_search` | `(query_ids, top_k=10, search_mode="single") -> list[list[dict]]` | Vector/MaxSim search |
| `vector_search_by_embedding` | `(embedding, top_k=10) -> list[dict]` | Vector search with raw embedding |
| `fetch_query_texts` | `(query_ids: list[int]) -> list[str]` | Batch fetch query texts |
| `find_query_by_text` | `(query_text: str) -> Any \| None` | Find query by text content |

### GenerationPipelineService (`orm/service/generation_pipeline.py`)

| Method | Signature | Purpose |
|---|---|---|
| `save_pipeline` | `(name: str, config: dict) -> int` | Create pipeline, return ID |
| `run_pipeline` | `(generate_func, pipeline_id, top_k=10, batch_size=128, max_concurrency=16, max_retries=3, retry_delay=1.0) -> dict` | Batch generation with retry |
| `get_pipeline_config` | `(pipeline_id: int) -> dict \| None` | Get pipeline config |
| `delete_pipeline_results` | `(pipeline_id: int) -> int` | Delete results for pipeline |
| `get_chunk_contents` | `(chunk_ids: list[int \| str]) -> list[str]` | Get chunk contents by IDs |
| `get_query_text` | `(query_id: int) -> str` | Get query text (prefers `query_to_llm`) |

### BaseIngestionService (`orm/service/base_ingestion.py`)

| Method | Signature | Purpose |
|---|---|---|
| `add_chunks` | `(chunks: list[dict]) -> list[int \| str]` | Bulk add chunks |
| `add_queries` | `(queries: list[dict]) -> list[int \| str]` | Bulk add queries |
| `link_pages_to_chunks` | `(relations: list[dict]) -> list[tuple]` | Link pages to chunks (M:N) |
| `link_page_to_chunks` | `(page_id, chunk_ids) -> list[tuple]` | Link one page to multiple chunks |
| `set_query_embeddings` | `(query_ids, embeddings) -> int` | Set single-vector query embeddings |
| `set_query_multi_embeddings` | `(query_ids, embeddings) -> int` | Set multi-vector query embeddings |
| `set_chunk_embeddings` | `(chunk_ids, embeddings) -> int` | Set single-vector chunk embeddings |
| `set_chunk_multi_embeddings` | `(chunk_ids, embeddings) -> int` | Set multi-vector chunk embeddings |
| `embed_all_queries` | `(embed_func, batch_size=100, max_concurrency=10, bm25_tokenizer="bert") -> int` | Embed all unembedded queries |
| `embed_all_queries_multi_vector` | `(embed_func, batch_size=100, max_concurrency=10, bm25_tokenizer="bert") -> int` | Multi-vector embed queries |
| `embed_all_chunks` | `(embed_func, batch_size=100, max_concurrency=10, bm25_tokenizer="bert") -> int` | Embed all unembedded chunks |
| `embed_all_chunks_multi_vector` | `(embed_func, batch_size=100, max_concurrency=10, bm25_tokenizer="bert") -> int` | Multi-vector embed chunks |
| `add_retrieval_gt` | `(query_id, gt, chunk_type="mixed", upsert=False) -> list[tuple]` | Add retrieval ground truth |
| `add_retrieval_gt_batch` | `(items, chunk_type="mixed", upsert=False) -> list[tuple]` | Batch add retrieval GT |

### BaseEvaluationService (`orm/service/base_evaluation.py`)

| Method | Signature | Purpose |
|---|---|---|
| `set_metric` | `(metric_id: int, metric_func: MetricFunc) -> None` | Set active metric |
| `get_metric` | `(metric_name, metric_type=None) -> Any \| None` | Get metric by name |
| `get_or_create_metric` | `(name: str, metric_type: str) -> int` | Get or create metric, return ID |
| `evaluate` | `(pipeline_id, batch_size=100) -> tuple[int, float \| None]` | Run full evaluation pipeline |
| `is_evaluation_complete` | `(pipeline_id, metric_id, batch_size=100) -> bool` | Check evaluation completeness |
| `verify_pipeline_completion` | `(pipeline_id, batch_size=100) -> bool` | Verify all queries have results |

---

## Common Duplication Patterns

These are the top patterns that get reimplemented instead of reusing existing code. **DO NOT** write these patterns; use the replacement instead.

### 1. Manual asyncio.Semaphore + gather

```python
# BAD - reimplements run_with_concurrency_limit
semaphore = asyncio.Semaphore(10)
async def limited(item):
    async with semaphore:
        return await process(item)
results = await asyncio.gather(*[limited(x) for x in items])
```

```python
# GOOD
from autorag_research.util import run_with_concurrency_limit
results = await run_with_concurrency_limit(items, process, max_concurrency=10)
```

### 2. Manual min/max normalization

```python
# BAD - reimplements normalize_minmax
min_s, max_s = min(scores), max(scores)
normalized = [(s - min_s) / (max_s - min_s) for s in scores]
```

```python
# GOOD
from autorag_research.util import normalize_minmax
normalized = normalize_minmax(scores)
```

### 3. asyncio.to_thread wrapper

```python
# BAD - reimplements to_async_func
async def async_version(*args):
    return await asyncio.to_thread(sync_func, *args)
```

```python
# GOOD
from autorag_research.util import to_async_func
async_version = to_async_func(sync_func)
```

### 4. Manual token usage summation

```python
# BAD - reimplements aggregate_token_usage
total_prompt = sum(r["token_usage"]["prompt_tokens"] for r in results)
total_completion = sum(r["token_usage"]["completion_tokens"] for r in results)
```

```python
# GOOD
from autorag_research.util import aggregate_token_usage
prompt, completion, embedding, exec_time = aggregate_token_usage(results)
```

### 5. BytesIO + image.save pattern

```python
# BAD - reimplements pil_image_to_bytes
buffer = io.BytesIO()
image.save(buffer, format="PNG")
image_bytes = buffer.getvalue()
```

```python
# GOOD
from autorag_research.util import pil_image_to_bytes
image_bytes, mimetype = pil_image_to_bytes(image)
```

### 6. Base64 data URI parsing

```python
# BAD - reimplements extract_image_from_data_uri
match = re.match(r"data:([^;]+);base64,(.+)", data_uri)
mimetype = match.group(1)
image_bytes = base64.b64decode(match.group(2))
```

```python
# GOOD
from autorag_research.util import extract_image_from_data_uri
image_bytes, mimetype = extract_image_from_data_uri(data_uri)
```

### 7. Direct UoW search calls from pipelines

```python
# BAD - bypasses service layer
with self._service._create_uow() as uow:
    results = uow.chunks.bm25_search(query_text=text, ...)
```

```python
# GOOD - use service methods
results = self._service.bm25_search(query_ids, top_k, tokenizer=tokenizer)
# or for raw text:
results = self._service.bm25_search_by_text(query_text, top_k)
```

### 8. Direct uow.queries.get_by_id for text

```python
# BAD - bypasses service convenience methods
with self._service._create_uow() as uow:
    query = uow.queries.get_by_id(query_id)
    text = query.contents
```

```python
# GOOD - use service methods
text = self._service.get_query_text(query_id)         # GenerationPipelineService
texts = self._service.fetch_query_texts(query_ids)     # RetrievalPipelineService
```

---

## When to Add New Utilities

Add a new utility to `util.py` only when ALL of these conditions are met:
1. The logic is used (or will be used) in **2+ unrelated modules**
2. The logic is **>5 lines** of non-trivial code
3. The logic is **generic** (not tied to a specific pipeline or model)
4. No existing utility covers the use case (check above catalog first)

When you DO add a new utility:
- Add it to `autorag_research/util.py` with full type hints
- Update THIS document (`ai_instructions/utility_reference.md`) with the new entry
- Add it to the Quick Lookup Table
- If it replaces a common pattern, add a duplication pattern entry
