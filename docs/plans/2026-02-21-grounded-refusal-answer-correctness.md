# Grounded Refusal And Answer Correctness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add paper-aligned generation metrics for grounded refusal (F1GR) and answer correctness (F1AC), with dataset-level computation and configurable LLM judge injection.

**Architecture:** Extend metric configuration with compute granularity (`query` vs `dataset`) and route dataset-level metrics through a full-dataset evaluation pass. Implement refusal-judge and correctness aggregation in generation metrics using the paper’s dataset-level formulas. Keep schema unchanged by persisting dataset-level final score to per-query evaluation rows for compatibility.

**Tech Stack:** Python 3.10+, pytest, LangChain BaseLanguageModel injection (`with_llm`), SQLAlchemy service/UoW layer.

---

### Task 1: Add Granularity Contract

**Files:**
- Modify: `autorag_research/config.py`
- Modify: `autorag_research/executor.py`
- Test: `tests/autorag_research/evaluation/metrics/test_generation.py`

**Step 1: Write the failing test**

```python
def test_dataset_level_metric_config_granularity():
    config = GroundedRefusalF1Config()
    assert config.get_compute_granularity() == "dataset"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/autorag_research/evaluation/metrics/test_generation.py::test_dataset_level_metric_config_granularity -v`
Expected: FAIL because config class/granularity method does not exist.

**Step 3: Write minimal implementation**

```python
class BaseMetricConfig(ABC):
    def get_compute_granularity(self) -> Literal["query", "dataset"]:
        return "query"
```

And pass this granularity from executor to evaluation service.

**Step 4: Run test to verify it passes**

Run: `pytest tests/autorag_research/evaluation/metrics/test_generation.py::test_dataset_level_metric_config_granularity -v`
Expected: PASS.

### Task 2: Add Dataset-Level Evaluation Path

**Files:**
- Modify: `autorag_research/orm/service/base_evaluation.py`
- Modify: `autorag_research/orm/service/generation_evaluation.py`
- Test: `tests/autorag_research/orm/service/test_generation_evaluation.py`

**Step 1: Write the failing test**

```python
def test_evaluate_dataset_level_metric_persists_same_score_for_all_queries(...):
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/autorag_research/orm/service/test_generation_evaluation.py::test_evaluate_dataset_level_metric_persists_same_score_for_all_queries -v`
Expected: FAIL because dataset-level mode is unsupported.

**Step 3: Write minimal implementation**

- Add metric granularity to `set_metric(...)`.
- Branch `evaluate(...)` for dataset mode.
- For dataset mode: collect all missing query metric inputs, run metric once, persist returned values.
- Keep existing query-level flow unchanged.

**Step 4: Run test to verify it passes**

Run: `pytest tests/autorag_research/orm/service/test_generation_evaluation.py::test_evaluate_dataset_level_metric_persists_same_score_for_all_queries -v`
Expected: PASS.

### Task 3: Implement Paper Metrics (F1GR/F1AC)

**Files:**
- Modify: `autorag_research/evaluation/metrics/generation.py`
- Modify: `autorag_research/evaluation/metrics/__init__.py`
- Create: `configs/metrics/generation/grounded_refusal_f1.yaml`
- Create: `configs/metrics/generation/answer_correctness_f1.yaml`
- Test: `tests/autorag_research/evaluation/metrics/test_generation.py`

**Step 1: Write failing tests**

- Grounded refusal metric returns identical dataset-level score for each input.
- Answer correctness metric computes PAC/RAC/F1AC from calibrated claims and returns identical dataset-level score.
- Refusal judge supports injected `judge_llm` string config.

**Step 2: Run tests to verify they fail**

Run:
- `pytest tests/autorag_research/evaluation/metrics/test_generation.py::test_grounded_refusal_f1_dataset_level -v`
- `pytest tests/autorag_research/evaluation/metrics/test_generation.py::test_answer_correctness_f1_dataset_level -v`
Expected: FAIL because functions/configs are missing.

**Step 3: Write minimal implementation**

- Add GPT evaluator prompt (paper Table 27 style).
- Add refusal parser (`Judgement: REFUSED|NOT REFUSED`).
- Add `grounded_refusal_f1` and `answer_correctness_f1` functions.
- Add dataclass configs with `get_compute_granularity() == "dataset"`.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/autorag_research/evaluation/metrics/test_generation.py -v`
Expected: PASS for newly added tests.

### Task 4: Docs + Integration Wiring

**Files:**
- Modify: `docs/metrics/generation/index.md`
- Create: `docs/metrics/generation/grounded-refusal-f1.md`
- Create: `docs/metrics/generation/answer-correctness-f1.md`
- Modify: `docs/metrics/index.md`
- Modify: `mkdocs.yml`

**Step 1: Write docs tests/check assumptions**

- Verify doc links resolve via MkDocs nav.

**Step 2: Run docs-oriented checks**

Run: `mkdocs build -q`
Expected: build succeeds.

**Step 3: Minimal docs implementation**

- Add formulas and usage examples.
- Explain dataset-level semantics and unchanged schema.

**Step 4: Re-run docs check**

Run: `mkdocs build -q`
Expected: PASS.

### Task 5: Full Verification

**Files:**
- Modify as needed from previous tasks.

**Step 1: Run targeted test suite**

Run:
- `pytest tests/autorag_research/evaluation/metrics/test_generation.py -v`
- `pytest tests/autorag_research/orm/service/test_generation_evaluation.py -v`

Expected: PASS.

**Step 2: Run quality checks**

Run: `ruff check autorag_research tests`
Expected: PASS with no violations.

**Step 3: Final sanity**

Run: `pytest -q`
Expected: no regressions in touched areas (if environment allows full run).
