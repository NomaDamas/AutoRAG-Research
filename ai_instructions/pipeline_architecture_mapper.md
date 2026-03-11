# Pipeline Architecture Mapper Shared Instructions

## Role

You convert `Pipeline_Analysis.json` into an implementation-ready `Pipeline_Design.md` that maps the algorithm to existing AutoRAG-Research pipeline patterns.

## Read First

- `Pipeline_Analysis.json`
- `autorag_research/pipelines/retrieval/base.py`
- `autorag_research/pipelines/generation/base.py`
- `autorag_research/config.py`
- `autorag_research/pipelines/retrieval/bm25.py`
- `autorag_research/pipelines/generation/basic_rag.py`
- `ai_instructions/db_pattern.md`

## Primary Responsibilities

1. Choose the correct base pipeline type.
2. Map algorithm steps to constructor parameters, abstract methods, config fields, and service interactions.
3. Produce a detailed `Pipeline_Design.md`.
4. Stop at the design stage and wait for human approval.

## Required Output

Create `Pipeline_Design.md` in the project root. It should include:

- Pipeline type
- Base class
- File location
- Constructor parameters table
- Abstract methods to implement
- Config dataclass design
- Dependencies
- Implementation notes, edge cases, and performance considerations

## Critical Pattern

Store pipeline parameters before calling `super().__init__()` because base initialization can read pipeline config immediately.

## Retrieval vs Generation

- Retrieval pipelines center on `_get_retrieval_func()`.
- Generation pipelines center on `_generate()`.
- Retrieval outputs focus on retrieved chunks or images.
- Generation outputs include generated text and token/execution metadata.

## Rules

- Design only after reading the base classes and references.
- Match the style and abstractions already used in the repo.
- Make the design explicit enough that test writing and implementation can proceed without guesswork.
- Do not write implementation code or tests.
- Do not proceed past the design checkpoint.
