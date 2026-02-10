# Utility & Service Reference

**MANDATORY**: Check this catalog before implementing. If it exists here, use it — don't reimplement.

## Utility Functions

### `autorag_research/util.py` — Core Utilities

| Function | Purpose |
|---|---|
| `to_list(item)` | Recursively convert numpy/pandas/iterables to Python lists |
| `convert_inputs_to_list(func)` | Decorator: convert all function inputs to lists |
| `truncate_texts(str_list, max_tokens)` | Truncate strings to token limit (tiktoken cl100k_base) |
| `normalize_string(s)` | SQuAD-style: lowercase, remove punctuation/articles, fix whitespace |
| `normalize_minmax(scores)` | Min-max normalization to [0,1], handles None |
| `normalize_tmm(scores, theoretical_min)` | Theoretical min-max (known min, actual max) |
| `normalize_zscore(scores)` | Z-score standardization (mean=0, std=1), handles None |
| `normalize_dbsf(scores)` | 3-sigma distribution-based score fusion, clips to [0,1] |
| `run_with_concurrency_limit(items, async_func, max_concurrency)` | Async semaphore-based concurrency control |
| `to_async_func(func)` | Convert sync function to async via `asyncio.to_thread` |
| `unpack_and_run(target_list, func)` | Flatten sublists, run func, regroup by original lengths |
| `load_image(img)` | ImageType (str/Path/bytes/BytesIO) → PIL Image (RGB) |
| `pil_image_to_bytes(image)` | PIL Image → `(bytes, mimetype)` |
| `extract_image_from_data_uri(data_uri)` | Data URI → `(bytes, mimetype)` |
| `aggregate_token_usage(results)` | Sum `(prompt, completion, embedding, exec_time)` from result dicts |
| `validate_plugin_name(name)` | Check plugin name is valid (`^[a-z][a-z0-9_]*$`) |

### `autorag_research/data/util.py` — Data Ingestion

| Function | Purpose |
|---|---|
| `make_id(*parts)` | Join parts with underscore to generate IDs |

### `autorag_research/evaluation/metrics/util.py` — Metric Helpers

| Function | Purpose |
|---|---|
| `calculate_cosine_similarity(a, b)` | Cosine similarity between two vectors |
| `calculate_l2_distance(a, b)` | L2 (Euclidean) distance between two vectors |
| `calculate_inner_product(a, b)` | Inner product between two vectors |
| `metric(fields_to_check)` | Decorator: run metric per-input with None field validation |
| `metric_loop(fields_to_check)` | Decorator: run metric on batch with None field validation |

### `autorag_research/orm/util.py` — Database Management

| Function | Purpose |
|---|---|
| `create_database(host, user, password, database)` | Create PostgreSQL database |
| `drop_database(host, user, password, database)` | Drop PostgreSQL database |
| `database_exists(host, user, password, database)` | Check if database exists |
| `install_vector_extensions(host, user, password, database)` | Install vchord/vector/BM25 extensions |

### `autorag_research/cli/utils.py` — CLI Helpers

| Function | Purpose |
|---|---|
| `discover_configs(config_dir)` | Scan YAML configs → `{name: description}` |
| `discover_pipelines(pipeline_type)` | Discover pipeline configs |
| `discover_metrics(pipeline_type)` | Discover metric configs |
| `discover_embedding_configs()` | Discover embedding configs |
| `get_config_dir()` | Get configs directory path |
| `setup_logging(verbose)` | Configure CLI logging |

## Service Layer Methods

### RetrievalPipelineService

| Method | Purpose |
|---|---|
| `bm25_search(query_ids, top_k)` | BM25 search by query IDs |
| `bm25_search_by_text(query_text, top_k)` | BM25 search by raw text |
| `vector_search(query_ids, top_k, search_mode)` | Vector/MaxSim search |
| `vector_search_by_embedding(embedding, top_k)` | Vector search with raw embedding |
| `fetch_query_texts(query_ids)` | Batch fetch query texts |
| `find_query_by_text(query_text)` | Find query by text content |
| `save_pipeline(name, config)` | Create pipeline, return ID |
| `run_pipeline(retrieval_func, pipeline_id, top_k)` | Batch retrieval with retry |

### GenerationPipelineService

| Method | Purpose |
|---|---|
| `get_query_text(query_id)` | Get query text (prefers `query_to_llm`) |
| `get_chunk_contents(chunk_ids)` | Get chunk contents by IDs |
| `save_pipeline(name, config)` | Create pipeline, return ID |
| `run_pipeline(generate_func, pipeline_id, top_k)` | Batch generation with retry |

### BaseIngestionService

| Method | Purpose |
|---|---|
| `add_chunks(chunks)` | Bulk add text chunks |
| `add_queries(queries)` | Bulk add queries |
| `embed_all_queries(embed_func)` | Embed all unembedded queries |
| `embed_all_chunks(embed_func)` | Embed all unembedded chunks |
| `add_retrieval_gt(query_id, gt)` | Add retrieval ground truth |
| `add_retrieval_gt_batch(items)` | Batch add retrieval ground truth |

### BaseEvaluationService

| Method | Purpose |
|---|---|
| `set_metric(metric_id, metric_func)` | Set active metric for evaluation |
| `get_or_create_metric(name, metric_type)` | Get or create metric, return ID |
| `evaluate(pipeline_id)` | Run full evaluation pipeline |

## Common Duplication Patterns

DO NOT write these — use the replacement instead.

| Pattern (BAD) | Use Instead |
|---|---|
| `asyncio.Semaphore` + `gather` | `run_with_concurrency_limit()` |
| `min(scores)` / `max(scores)` manual normalization | `normalize_minmax()` / `normalize_tmm()` / etc. |
| `asyncio.to_thread` wrapper | `to_async_func()` |
| Manual `token_usage` summation loop | `aggregate_token_usage()` |
| `Image.open(path).convert("RGB")` with type checks | `load_image()` |
| `BytesIO()` + `image.save()` | `pil_image_to_bytes()` |
| `base64.b64decode` + data URI regex | `extract_image_from_data_uri()` |
| `np.dot` / `np.linalg.norm` for similarity | `calculate_cosine_similarity()` |
| Direct `uow.chunks.bm25_search()` from pipeline | `service.bm25_search()` / `service.bm25_search_by_text()` |
| Direct `uow.queries.get_by_id()` for text | `service.get_query_text()` / `service.fetch_query_texts()` |

## Adding New Utilities

When you add a new utility function to any `util.py`, you **MUST** update this document.
