# Pipeline Completion Checklist

Use this checklist to verify that a new or modified pipeline meets all project conventions before merging. Each section is ordered by importance.

## 1. Code Structure

- [ ] Pipeline class extends the correct base (`BaseRetrievalPipeline` or `BaseGenerationPipeline`)
- [ ] Abstract methods implemented:
  - Retrieval: `_retrieve_by_id()`, `_retrieve_by_text()`, `_get_pipeline_config()`
  - Generation: `_generate()`, `_get_pipeline_config()`
- [ ] Constructor stores custom parameters **before** calling `super().__init__()`
- [ ] Logger uses `logging.getLogger("AutoRAG-Research")`
- [ ] No relative imports
- [ ] No `global` variables
- [ ] No `print()` statements (use logger)
- [ ] Python 3.10+ type hints (`list`, `dict`, `|` instead of `typing.List`, `typing.Optional`)

## 2. Config Dataclass

- [ ] `@dataclass(kw_only=True)` inheriting `BaseRetrievalPipelineConfig` or `BaseGenerationPipelineConfig`
- [ ] `get_pipeline_class()` returns the pipeline class
- [ ] `get_pipeline_kwargs()` returns all constructor kwargs
- [ ] Generation config: `retrieval_pipeline_name` declared, `_retrieval_pipeline` injection guard in `get_pipeline_kwargs()`
- [ ] If custom LLM fields exist (e.g. `critic_llm`): `__setattr__` auto-converts string → model via `load_llm()`

## 3. Exports & Discovery

- [ ] Pipeline class and config exported in `autorag_research/pipelines/[type]/__init__.py`
- [ ] Both added to `__all__`
- [ ] YAML config file exists at `configs/pipelines/[type]/[name].yaml`
- [ ] YAML has `_target_` pointing to the config dataclass
- [ ] YAML has `description` field (used by `discover_pipelines()`)
- [ ] `discover_pipelines('[type]')` returns the new pipeline

## 4. YAML Config Validity

- [ ] `_target_` class path is correct and importable
- [ ] `llm` field references a config name that exists in `configs/llm/`
- [ ] Generation: `retrieval_pipeline_name` references an existing retrieval config
- [ ] Config instantiates without error: `hydra.utils.instantiate(OmegaConf.load(yaml_path))`
- [ ] `get_pipeline_kwargs()` succeeds after retrieval pipeline injection

## 5. Tests

- [ ] Test file at `tests/autorag_research/pipelines/[type]/test_[name]_pipeline.py`
- [ ] Unit tests for helpers/parsers (no DB required)
- [ ] Integration test with DB session (uses `db_session` fixture from conftest)
- [ ] Async tests marked with `@pytest.mark.asyncio`
- [ ] API/LLM calls mocked (FakeListLLM, AsyncMock), not real
- [ ] Tests pass: `uv run pytest tests/autorag_research/pipelines/[type]/test_[name]_pipeline.py -v`

## 6. Code Quality

- [ ] `make check` passes (ruff lint, ruff format, ty type check, deptry)
- [ ] No new utility functions without updating `ai_instructions/utility_reference.md`
- [ ] Existing utility/service methods reused where applicable (check `ai_instructions/utility_reference.md`)

## 7. Paper/Algorithm Fidelity (if implementing a research paper)

- [ ] Core algorithm stages match the paper (e.g., retrieve → generate → critic → plan → execute)
- [ ] Action functions / prompts match the paper's definitions and signatures
- [ ] Separate models used where the paper specifies distinct components (e.g., critic model != generator model)
- [ ] Reference tables / mapping data from the paper included in prompts when the algorithm requires them
- [ ] Iteration / loop structure matches the paper (single-pass vs. multi-iteration)
- [ ] Dead code or unused helpers removed
- [ ] Deviations from the paper documented in code comments or config description
