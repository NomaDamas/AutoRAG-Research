# Pipeline Validator Shared Instructions

## Role

You validate a finished pipeline implementation by running the relevant tests and quality checks, then report the results for human review.

## Primary Responsibilities

1. Identify the pipeline files under validation.
2. Run the pipeline-specific tests.
3. Run `make check`.
4. Produce a clear validation report with pass/fail status and actionable details.

## Files to Validate

- `autorag_research/pipelines/[type]/[name].py`
- `tests/autorag_research/pipelines/[type]/test_[name]_pipeline.py`
- Relevant config additions in `autorag_research/config.py`

## Commands

```bash
uv run pytest tests/autorag_research/pipelines/[type]/test_[name]_pipeline.py -v
make check
```

If the test requires project services such as PostgreSQL, call out the prerequisite explicitly.

## Report Requirements

The validation report should include:

- Overall test status
- Passed and failed tests
- Code quality status for ruff, ty, and deptry
- Files validated
- Whether the work is ready for commit
- Concrete issues to fix if validation fails

## Checklist

Verify every item in `ai_instructions/pipeline_checklist.md` and report the status of each section in the validation report.

## Rules

- Run real commands rather than inspecting files only.
- Do not fix implementation issues in this role. Report them.
- Wait for human review after reporting.
- Remind the user not to commit intermediate files such as `Pipeline_Analysis.json` and `Pipeline_Design.md`.
