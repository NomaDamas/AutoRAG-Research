---
allowed-tools: Task, Read, Write, Glob, Grep, AskUserQuestion, TodoWrite, Bash
description: Orchestrate full pipeline implementation workflow from paper to validated code. Use when implementing a new retrieval or generation pipeline from a research paper.
---

# Pipeline Implementation Workflow Orchestrator

This skill orchestrates the complete pipeline implementation workflow using specialized sub-agents with mandatory human checkpoints.

## Arguments

- `$ARGUMENTS`: Paper URL (arxiv, PDF path, or web URL) or algorithm name if paper already analyzed

## Prerequisites

- Research paper accessible (URL, arxiv link, or local PDF)
- Docker PostgreSQL running for test validation (`make docker-up`)

## Workflow Overview

```
Phase 1: Paper Analysis      → Pipeline_Analysis.json
Phase 2: Architecture Design → Pipeline_Design.md (HUMAN CHECKPOINT)
Phase 3: Test Writing        → test_[name]_pipeline.py (HUMAN CHECKPOINT)
Phase 4: Implementation      → [name].py
Phase 5: Validation          → All checks pass (HUMAN CHECKPOINT)
```

## Phase 1: Paper Analysis

**Agent:** `pipeline-paper-analyst`

```
Task(subagent_type="pipeline-paper-analyst", prompt="Analyze paper: $ARGUMENTS. Extract algorithm details and create Pipeline_Analysis.json")
```

**Output:** `Pipeline_Analysis.json` in project root

**After Phase 1:**
- Present summary of extracted algorithm
- Confirm pipeline type (retrieval/generation)
- Ask if user wants to proceed to design phase

---

## Phase 2: Architecture Design (HUMAN CHECKPOINT)

**Agent:** `pipeline-architecture-mapper`

**Prerequisites:** `Pipeline_Analysis.json` exists

```
Task(subagent_type="pipeline-architecture-mapper", prompt="Design architecture for [algorithm_name] based on Pipeline_Analysis.json")
```

**Output:** `Pipeline_Design.md` in project root

**MANDATORY Human Review:**
```
AskUserQuestion:
  question: "Review the Pipeline_Design.md. How would you like to proceed?"
  header: "Design Review"
  options:
    - label: "Approve Design"
      description: "Design looks good. Proceed to test writing (TDD Phase)."
    - label: "Request Changes"
      description: "I have feedback. Revise the design."
    - label: "Reject"
      description: "Stop workflow. Design needs major rework."
```

**Handle Response:**
- **Approve:** Proceed to Phase 3
- **Request Changes:** Get feedback, re-invoke agent, repeat until approved
- **Reject:** Stop workflow, acknowledge rejection

---

## Phase 3: Test Writing - TDD (HUMAN CHECKPOINT)

**Agent:** `pipeline-test-writer`

**Prerequisites:** Approved `Pipeline_Design.md`

```
Task(subagent_type="pipeline-test-writer", prompt="Write tests for [algorithm_name] pipeline based on Pipeline_Design.md")
```

**Output:** `tests/autorag_research/pipelines/[type]/test_[name]_pipeline.py`

**MANDATORY Human Review:**
```
AskUserQuestion:
  question: "Review the test file. Tests should be written BEFORE implementation (TDD). Proceed?"
  header: "Test Review"
  options:
    - label: "Approve Tests"
      description: "Tests cover requirements. Proceed to implementation."
    - label: "Add More Tests"
      description: "Need additional test cases."
    - label: "Reject"
      description: "Stop workflow."
```

---

## Phase 4: Implementation

**Agent:** `pipeline-implementer`

**Prerequisites:** Approved test file, `Pipeline_Design.md`

```
Task(subagent_type="pipeline-implementer", prompt="Implement [algorithm_name] pipeline to pass the tests based on Pipeline_Design.md")
```

**Output:**
- `autorag_research/pipelines/[type]/[name].py`
- Config additions to `autorag_research/config.py`
- Exports in `__init__.py`

**After Phase 4:**
- Automatically proceed to validation

---

## Phase 5: Validation (HUMAN CHECKPOINT)

**Agent:** `pipeline-validator`

```
Task(subagent_type="pipeline-validator", prompt="Validate [algorithm_name] pipeline implementation. Run tests and make check.")
```

**Output:** Validation report with pass/fail status

**MANDATORY Human Review:**
```
AskUserQuestion:
  question: "Validation complete. Review the results and decide next step."
  header: "Final Review"
  options:
    - label: "Approve & Commit"
      description: "All checks pass. Ready to commit and create PR."
    - label: "Fix Issues"
      description: "Some issues need fixing. Return to implementation."
    - label: "Cancel"
      description: "Abort the workflow."
```

---

## Cleanup After Workflow

Remind user to delete intermediate files (DO NOT commit these):
- `Pipeline_Analysis.json`
- `Pipeline_Design.md`

## Critical Rules

1. **Human checkpoints are MANDATORY** - Never skip AskUserQuestion at Phase 2, 3, 5
2. **TDD approach** - Tests MUST be written before implementation (Phase 3 before Phase 4)
3. **Intermediate files are local only** - Never commit Pipeline_Analysis.json or Pipeline_Design.md
4. **Loop on changes** - Keep revising until user approves
5. **Track progress** - Use TodoWrite to track which phase is complete

## Quick Reference

| Phase | Agent | Output | Human Checkpoint |
|-------|-------|--------|------------------|
| 1 | pipeline-paper-analyst | Pipeline_Analysis.json | No (summary shown) |
| 2 | pipeline-architecture-mapper | Pipeline_Design.md | **YES** |
| 3 | pipeline-test-writer | test_[name]_pipeline.py | **YES** |
| 4 | pipeline-implementer | [name].py + config | No |
| 5 | pipeline-validator | Validation report | **YES** |

## Example Usage

User: `/implement-pipeline https://arxiv.org/abs/2212.10496`

The workflow will:
1. Analyze the HyDE paper
2. Design architecture (wait for approval)
3. Write tests (wait for approval)
4. Implement pipeline
5. Validate (wait for approval)
6. Ready for commit/PR
