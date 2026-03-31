# AutoRAG-Research Agent Workflows

This document defines how the main Codex agent orchestrates specialized sub-agents for multi-step workflows. Each workflow has mandatory human checkpoints where the user must approve before proceeding.

## Pipeline Implementation Workflow

When asked to implement a new RAG pipeline from a research paper:

### Phase 1: Paper Analysis
Spawn `pipeline-paper-analyst` agent with the paper URL/path.
- **Input:** Research paper (arxiv link, PDF path, or URL)
- **Output:** `Pipeline_Analysis.json` in project root
- Present a summary of the extracted algorithm to the user.

### Phase 2: Architecture Design (HUMAN CHECKPOINT)
Spawn `pipeline-architecture-mapper` agent.
- **Input:** `Pipeline_Analysis.json`
- **Output:** `Pipeline_Design.md` in project root
- **Present the design to the user. Wait for explicit approval before proceeding.**
- If the user requests changes, re-invoke the agent with feedback.

### Phase 3: Test Writing - TDD (HUMAN CHECKPOINT)
Spawn `pipeline-test-writer` agent.
- **Input:** Approved `Pipeline_Design.md`
- **Output:** `tests/autorag_research/pipelines/[type]/test_[name]_pipeline.py`
- **Present the tests to the user. Wait for explicit approval before proceeding.**

### Phase 4: Implementation
Spawn `pipeline-implementer` agent.
- **Input:** Approved test file + `Pipeline_Design.md`
- **Output:** Pipeline class, config dataclass, `__init__.py` exports

### Phase 5: Validation (HUMAN CHECKPOINT)
Spawn `pipeline-validator` agent.
- **Output:** Validation report (test results + `make check` results)
- **Present results to the user. Wait for approval before committing.**

### Phase 6: Wrap Up
- Write a config YAML in `configs/` for the new pipeline.
- Remind user to delete intermediate files (`Pipeline_Analysis.json`, `Pipeline_Design.md`).

### Quick Reference

| Phase | Agent | Output | Human Checkpoint |
|-------|-------|--------|------------------|
| 1 | pipeline-paper-analyst | Pipeline_Analysis.json | No (summary shown) |
| 2 | pipeline-architecture-mapper | Pipeline_Design.md | **YES** |
| 3 | pipeline-test-writer | test_[name]_pipeline.py | **YES** |
| 4 | pipeline-implementer | [name].py + config | No |
| 5 | pipeline-validator | Validation report | **YES** |

---

## Data Ingestor Workflow

When asked to ingest a new dataset into AutoRAG-Research:

### Phase 1: Dataset Inspection
Spawn `dataset-inspector` agent with the dataset name/URL.
- **Input:** HuggingFace dataset identifier or URL
- **Output:** `Source_Data_Profile.json` in project root
- Present a summary of the dataset structure to the user.

### Phase 2: Schema Design (HUMAN CHECKPOINT)
Spawn `schema-architect` agent.
- **Input:** `Source_Data_Profile.json` + `ai_instructions/db_schema.md`
- **Output:** `Mapping_Strategy.md` in project root
- **Present the mapping strategy to the user. Wait for explicit approval.**
- If the user requests changes, re-invoke the agent with feedback.

### Phase 3: Test Writing
Spawn `test-writer` agent.
- **Input:** Approved `Mapping_Strategy.md`
- **Output:** Test file in `tests/autorag_research/data/`

### Phase 4: Implementation
Spawn `implementation-specialist` agent.
- **Input:** `Mapping_Strategy.md` + test file
- **Output:** Ingestor class in `autorag_research/data/`

### Quick Reference

| Phase | Agent | Output | Human Checkpoint |
|-------|-------|--------|------------------|
| 1 | dataset-inspector | Source_Data_Profile.json | No (summary shown) |
| 2 | schema-architect | Mapping_Strategy.md | **YES** |
| 3 | test-writer | test file | No |
| 4 | implementation-specialist | ingestor class | No |

---

## Critical Rules

1. **Human checkpoints are mandatory** - Never skip approval steps.
2. **TDD approach** - Tests are always written BEFORE implementation.
3. **Intermediate files are local only** - Never commit `Pipeline_Analysis.json`, `Pipeline_Design.md`, `Source_Data_Profile.json`, or `Mapping_Strategy.md`.
4. **Loop on changes** - Keep revising until the user approves.
5. **Sequential phases** - Each phase depends on the previous one completing successfully.
