# Answer Correctness F1 (F1AC)

Dataset-level calibrated answer correctness metric from the Trust-Align paper.

## Overview

| Field | Value |
|-------|-------|
| Type | Generation |
| Scope | Dataset-level |
| Range | [0, 1] |
| Higher is better | Yes |

## Definition

Per answerable, answered query:

```text
AC_q = |AG ∩ AD ∩ AR| / |AG ∩ AD|
```

- `AG`: gold claims (`generation_gt`)
- `AD`: claims supported by retrieved grounding contexts
- `AR`: claims present in generated answer

Dataset aggregation:

```text
PAC = sum(AC_q over Ag∩Ar) / |Ar|
RAC = sum(AC_q over Ag∩Ar) / |Ag|
F1AC = 2 * PAC * RAC / (PAC + RAC)
```

- `Ag`: answerable queries
- `Ar`: non-refusal queries

## Answerability Rule in AutoRAG-Research

- `generation_gt` exists: answerable
- `generation_gt` is missing/null: unanswerable

## Refusal Detection

Same evaluator path as F1GR:

- `judge_mode: llm` (paper-style judge)
- `judge_mode: phrase` (fallback)

## Configuration

```yaml
_target_: autorag_research.evaluation.metrics.generation.AnswerCorrectnessF1Config
judge_llm: openai-gpt5-mini
judge_mode: llm
rejection_flag: "I apologize, but I couldn't find an answer to your question in the search results."
rejection_threshold: 85
use_retrieval_calibration: true
```

## Notes

- Computed on the full dataset in one pass.
- Stored using existing per-query evaluation table for compatibility.
