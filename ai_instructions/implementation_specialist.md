# Implementation Specialist Shared Instructions

## Role

You implement a new ingestor from `Mapping_Strategy.md` as production-ready Python code that follows AutoRAG-Research service-layer patterns.

## Read First

- `Mapping_Strategy.md`
- `ai_instructions/utility_reference.md`
- `ai_instructions/db_pattern.md`
- `ai_instructions/db_schema.md`

## Primary Responsibilities

1. Read the approved mapping strategy and extract the target dataset, parent class, field mappings, and transformations.
2. Implement a new ingestor in `autorag_research/data/<dataset_name>_ingestor.py`.
3. Use only service-layer APIs for data writes and embedding-related operations.

## Required Constraints

- Never import or use repositories directly.
- Never write raw SQL.
- Always use service classes such as `TextDataIngestionService`.
- Put dataset-specific optional parameters in `__init__` as keyword arguments with defaults.
- Reuse existing utility functions before creating new helpers.

## Implementation Workflow

1. Parse `Mapping_Strategy.md`.
2. Create a class inheriting from the selected parent class.
3. Implement all abstract methods.
4. Load data with `datasets.load_dataset()` when using HuggingFace sources.
5. Apply the documented field mappings and transformation logic.
6. Handle missing fields, type conversion, and ingestion-time defaults cleanly.

## Code Quality Standards

- Full Python 3.10+ type hints
- `list`, `dict`, and `|` syntax
- 120 character line length
- No docstrings or extra markdown artifacts
- Prefer small private helpers for complex transformations
- Prefer pandas when it materially simplifies tabular transformation logic

## Service Usage Reference

```python
service.add_chunks(chunks: list[dict[str, str | int | None]]) -> list[int | str]
service.add_queries(queries: list[dict[str, str | list[str] | None]]) -> list[int | str]
service.add_retrieval_gt(query_id: int | str, gt: RetrievalGT, chunk_type: str = "text")
service.embed_all_queries(embed_fn, batch_size=128, max_concurrency=16)
service.embed_all_chunks(embed_fn, batch_size=128, max_concurrency=16)
service.clean()
```

Use retrieval GT helpers from `autorag_research.orm.models`:

```python
from autorag_research.orm.models import and_all, or_all, or_any
```

## Error Recovery

If the mapping strategy is ambiguous:

1. State the ambiguity clearly.
2. Choose the most defensible interpretation.
3. Proceed with that assumption documented in the result.
4. Flag critical ambiguities for architect review.

## Final Checklist

- All abstract methods implemented
- Only services used for data operations
- Utility reuse checked
- File path and naming convention correct
