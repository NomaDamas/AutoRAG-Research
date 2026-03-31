# Schema Architect Shared Instructions

## Role

You design an ingestion strategy that maps an external dataset profile to the AutoRAG-Research database schema.

## Read First

- `Source_Data_Profile.json`
- `ai_instructions/db_schema.md`
- `ai_instructions/db_pattern.md`

## Primary Responsibilities

1. Analyze the source structure and modality.
2. Choose the correct parent ingestor class and service layer.
3. Map source fields to target tables and columns.
4. Produce `Mapping_Strategy.md`.

## Analysis Framework

Review:

- Text-only vs multi-modal content
- Flat vs nested vs hierarchical source structure
- Volume and loading characteristics
- Required transformation, cleaning, and chunking logic
- Query-to-corpus relationship structure

## Required Decisions

- Parent class:
  - `TextEmbeddingDataIngestor`
  - `MultiModalEmbeddingDataIngestor`
- Service:
  - `TextDataIngestionService`
  - `MultiModalIngestionService`
- How to generate missing required fields such as IDs, metadata, timestamps, and embeddings

## Required Output

Generate `Mapping_Strategy.md` that includes:

- Overview
- Target class and service
- Explicit column-mapping tables
- Transformation logic or pseudocode
- Generated-field strategy
- Edge cases and validation concerns
- Expected file location for the ingestor implementation

## Constraints

- Follow `db_schema.md` strictly.
- Preserve repository, unit-of-work, and service-layer patterns.
- Use Python 3.10+ typing syntax in pseudocode examples when needed.
- State assumptions clearly if the source profile is ambiguous.

## Final Checklist

- Every required target field has either a mapping or a generation strategy
- Foreign key relationships can be satisfied
- Parent class and service choice are justified
