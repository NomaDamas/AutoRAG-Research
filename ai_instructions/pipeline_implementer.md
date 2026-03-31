# Pipeline Implementer Shared Instructions

## Role

You implement a RAG pipeline from `Pipeline_Design.md` and the existing tests. The goal is production code that passes the pre-written test suite.

## Read First

- `Pipeline_Design.md`
- The relevant test file in `tests/autorag_research/pipelines/[type]/`
- `autorag_research/pipelines/retrieval/base.py`
- `autorag_research/pipelines/generation/base.py`
- `autorag_research/config.py`
- `autorag_research/pipelines/retrieval/bm25.py`
- `autorag_research/pipelines/generation/basic_rag.py`
- `ai_instructions/utility_reference.md`

## Primary Responsibilities

1. Follow the approved design exactly.
2. Implement only what is needed to satisfy the tests.
3. Match established codebase patterns and style.
4. Add config wiring and exports required for the new pipeline.

## Required Deliverables

- `autorag_research/pipelines/[type]/[name].py`
- Config additions in `autorag_research/config.py`
- Export updates in `autorag_research/pipelines/[type]/__init__.py`

## Critical Pattern

Store constructor parameters before `super().__init__()` so `_get_pipeline_config()` can read them during base initialization.

## Retrieval Guidance

- Implement `_get_retrieval_func()`.
- The retrieval callable signature is `(query_ids: list[int | str], top_k: int) -> list[list[dict]]`.
- Result items should follow the pipeline conventions, typically including `doc_id`, `score`, and content metadata as expected by the framework.
- Prefer service-layer helpers like `bm25_search` and `vector_search` when they match the design.

## Generation Guidance

- Implement `_generate()`.
- Keep the flow explicit: retrieve context, format prompt, call the LLM, return the response object expected by the base pipeline.

## Code Style

- Python 3.10+ typing syntax
- Absolute imports only
- 120 character line length
- Type hints on all methods
- Docstrings where the surrounding code patterns expect them

## Rules

- Do not deviate from `Pipeline_Design.md` without a clear blocker.
- Reuse existing utilities and services before introducing new helpers.
- Do not add features that are not in the approved design or required by tests.
- Verify behavior by running the relevant tests when possible.

## Final Checklist

- Tests pass
- Config dataclass implemented
- `__init__.py` exports updated
- Parameter initialization order is correct
