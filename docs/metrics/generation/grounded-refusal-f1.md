# Grounded Refusal F1 (F1GR)

Dataset-level refusal quality metric from the Trust-Align paper.

## Overview

| Field | Value |
|-------|-------|
| Type | Generation |
| Scope | Dataset-level |
| Range | [0, 1] |
| Higher is better | Yes |

## Definition

F1GR evaluates both sides of refusal behavior:

- `F1ref`: correctly refusing unanswerable questions
- `F1ans`: correctly answering answerable questions (non-refusal)

Final score:

```text
F1GR = (F1ref + F1ans) / 2
```

This metric is computed over the full dataset, not by averaging per-query F1.

## Answerability Rule in AutoRAG-Research

- `generation_gt` exists: answerable
- `generation_gt` is missing/null: unanswerable

## Refusal Detection

Default mode follows the paper's evaluator style:

- `judge_mode: llm` and `judge_llm: <your-llm-config>`
- Judge returns `REFUSED` or `NOT REFUSED`

Fallback mode:

- `judge_mode: phrase` with phrase matching on `rejection_flag`

## Configuration

```yaml
_target_: autorag_research.evaluation.metrics.generation.GroundedRefusalF1Config
judge_llm: openai-gpt5-mini
judge_mode: llm
rejection_flag: "I apologize, but I couldn't find an answer to your question in the search results."
rejection_threshold: 85
```

## Notes

- Stored in DB via existing per-query schema for compatibility.
- Score is dataset-level and repeated per evaluated query row.
