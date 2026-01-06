# Workflow: Adding a New Dataset Ingestor

This document defines the workflow for implementing a new Dataset Ingestor in **AutoRAG-Research** using specialized sub-agents.

**IMPORTANT:** Intermediate artifacts (JSON analysis, Markdown strategy) are for development use only. **Do not commit or push these files to the git repository.**

## Agents

1. **Dataset Inspector:** Analyzes raw external data structure.
2. **Schema Architect:** Maps raw data to internal DB schema.
3. **Implementation Specialist:** Writes production code.
4. **Test Writer:** Creates unit tests.

During implementation, agents will run `make check` (ruff linting, ty type checking, deptry) to verify code quality and fix any issues.

## Workflow Steps

### Phase 1: Investigation

* **Agent:** Dataset Inspector
* **Input:** Dataset Name/Link from Issue.
* **Output:** `Source_Data_Profile.json` (Local only. Do not commit).

### Phase 2: Design

* **Agent:** Schema Architect
* **Input:** `Source_Data_Profile.json`, `ai_instructions/db_schema.md`.
* **Output:** `Mapping_Strategy.md` (Local only. Do not commit).

### Phase 3: Implementation

* **Agent:** Implementation Specialist
* **Input:** `Mapping_Strategy.md`.
* **Output:** `autorag_research/data/[dataset_name]_ingestor.py` (Commit this file).

### Phase 4: Testing

* **Agent:** Test Writer
* **Input:** Ingestor Source Code.
* **Output:** `tests/autorag_research/data/test_[dataset_name]_ingestor.py` (Commit this file).

## Definition of Done

* [ ] `Source_Data_Profile.json` generated (Local).
* [ ] `Mapping_Strategy.md` generated (Local).
* [ ] Ingestor class implemented in `autorag_research/data`.
* [ ] Unit tests implemented with mocked data.
* [ ] Static analysis (Lint/Type) passed.
* [ ] Intermediate files removed or excluded from git.
* [ ] PR ready with branch `Feature/#[IssueID]`.
