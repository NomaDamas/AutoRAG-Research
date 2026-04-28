# AutoRAG-Research Full-Scale Reproducibility Roadmap
## Comprehensive Reimplementation Benchmarking Across RAG Pipelines, Datasets, Metrics, and Modalities

**Target Venue:** NeurIPS / ICLR / ACL Datasets & Benchmarks-style track, JMLR, or TMLR
**Branch:** `experiment/paper-publication`
**Working Title:** *AutoRAG-Research: Reproducible Reimplementation and Benchmarking of Retrieval-Augmented Generation Pipelines*

---

## 0. Full-Scale Direction

This roadmap expands the reproducibility-centered paper direction into a larger benchmark program.

The full-scale paper should still be framed around:

> reproducible RAG reimplementation, transparent fidelity/deviation documentation, shared experiment execution, and released artifacts.

It should not become primarily a metric-sensitivity critique. Ranking shifts across metrics, datasets, and LLM backends are useful observations, but they are secondary to the reproducibility framework contribution.

---

## 1. Full-Scale Contributions

### Contribution 1 — Unified RAG Reimplementation Framework

AutoRAG-Research provides a single execution and evaluation stack for retrieval pipelines, generation pipelines, dataset ingestors, metrics, database persistence, result reporting, and plugin extension.

### Contribution 2 — Reimplemented RAG Method Library

The project collects representative RAG methods into one codebase with shared interfaces and reproducible configs.

### Contribution 3 — Fidelity and Deviation Documentation

Each method should have a fidelity card documenting what was implemented, what differs from the original paper, and what assumptions were needed.

### Contribution 4 — Reproducible Artifact Release

The benchmark should release configs, dataset dump manifests, cached outputs, metric scores, figure scripts, and environment manifests.

### Contribution 5 — Empirical Benchmark Report

The framework is validated by running reimplemented methods across text, image, and multimodal datasets, producing benchmark tables and cost/runtime analysis.

---

## 2. Implemented Component Inventory

### 2.1 Retrieval Pipelines

| # | Pipeline | Family | Full-Scale Role |
|---|----------|--------|-----------------|
| 1 | BM25 | sparse lexical | deterministic baseline |
| 2 | Vector Search | dense retrieval | neural baseline |
| 3 | Hybrid RRF | sparse+dense fusion | compositional retrieval wrapper |
| 4 | Hybrid CC | score fusion | alternative fusion strategy |
| 5 | HyDE | LLM query expansion | paper reimplementation case |
| 6 | Query Rewrite | LLM query reformulation | paper reimplementation case |
| 7 | RETRO\* | LLM reranking | high-cost reranking case study |
| 8 | Power of Noise | retrieval perturbation/ablation | robustness/diagnostic case |
| 9 | Question Decomposition Retrieval | decomposition | complex query handling case |
| 10 | HEAVEN | image retrieval | multimodal extension case |

### 2.2 Generation Pipelines

| # | Pipeline | Family | Full-Scale Role |
|---|----------|--------|-----------------|
| 1 | BasicRAG | retrieve-then-generate | baseline |
| 2 | IRCoT | interleaved retrieval + reasoning | published multi-hop case |
| 3 | ET2RAG | ensemble/subset selection | multi-call context selection case |
| 4 | MAIN-RAG | multi-agent filtering | agentic RAG case |
| 5 | Question Decomposition Generation | decomposition | complex query handling case |
| 6 | Self-RAG | self-reflective generation | advanced paper reimplementation |
| 7 | RAG-Critic | critic-guided correction | advanced paper reimplementation |
| 8 | SPD-RAG | speculative/parallel RAG | high-cost generation case |
| 9 | AutoThinkRAG | adaptive reasoning | adaptive pipeline case |
| 10 | VisRAG Generation | vision-language generation | multimodal generation case |

### 2.3 Dataset/Ingestor Families

| Dataset | Modality | Use in Full-Scale Plan |
|---------|----------|------------------------|
| BEIR | text retrieval | retrieval reproducibility |
| MTEB | text retrieval | retrieval reproducibility |
| RAGBench | text generation/RAG | generation reproducibility |
| MrTyDi | text retrieval | large retrieval reproducibility |
| BRIGHT | text retrieval + answers | retrieval/generation bridge |
| CRAG | web/search-grounded text | generation/RAG evaluation |
| ViDoRe | visual document retrieval | multimodal retrieval |
| ViDoRe v2 | visual document retrieval | multimodal retrieval |
| ViDoRe v3 | visual document retrieval | multimodal retrieval |
| VisRAG | visual/chart QA | multimodal RAG |
| Open-RAGBench | multimodal PDF RAG | full multimodal extension |

### 2.4 Metrics

#### Retrieval

- Recall@k
- Full Recall
- Precision@k
- F1@k
- nDCG@k
- MRR
- MAP

#### Generation

- ROUGE-L
- BERTScore F1
- SemScore
- Response Relevancy
- Exact Match
- Token F1
- BARTScore variants
- UniEval dimensions
- Optional plugin metrics such as Trust-Align grounded refusal / answer correctness when dependencies and dataset assumptions are satisfied

---

## 3. Full-Scale Experiment Tracks

### Track A — Text Retrieval Reimplementation

**Goal:** Demonstrate that retrieval papers and baselines can be run on shared corpora/qrels with reusable metrics.

| Scope | Default Full-Scale Choice |
|-------|---------------------------|
| Datasets | BEIR, MTEB, MrTyDi, BRIGHT, CRAG retrieval portions where valid |
| Pipelines | BM25, Vector Search, Hybrid RRF/CC, HyDE, Query Rewrite, RETRO\*, Power of Noise, Question Decomposition |
| Metrics | Recall@10, nDCG@10, MRR, MAP |
| Reproducibility checks | deterministic rerun agreement, config hash, dump hash, metric recomputation |

### Track B — Text Generation/RAG Reimplementation

**Goal:** Demonstrate end-to-end RAG generation methods on datasets with answer ground truth.

| Scope | Default Full-Scale Choice |
|-------|---------------------------|
| Datasets | RAGBench, BRIGHT, CRAG, and any BEIR/MTEB subsets only if answer GT is added and documented |
| Retrieval inputs | selected retrieval pipelines from Track A |
| Generation pipelines | BasicRAG, IRCoT, ET2RAG, MAIN-RAG, Question Decomposition, Self-RAG, RAG-Critic, SPD-RAG, AutoThinkRAG |
| Metrics | ROUGE-L, BERTScore, Token F1, Exact Match, BARTScore, UniEval where requirements are met |
| Reproducibility checks | cached generations, metric recomputation, token/cost manifests |

### Track C — Multimodal Reimplementation

**Goal:** Demonstrate that the same framework supports visual/multimodal RAG.

| Scope | Default Full-Scale Choice |
|-------|---------------------------|
| Datasets | ViDoRe, ViDoRe v2/v3, VisRAG, Open-RAGBench |
| Retrieval pipelines | HEAVEN, Vector Search with multimodal embeddings, Hybrid RRF |
| Generation pipelines | VisRAG Generation, AutoThinkRAG multimodal variants where valid |
| Metrics | retrieval metrics and dataset-appropriate answer metrics |
| Reproducibility checks | image/PDF artifact manifests, embedding manifests, VLM prompt/cache manifests |

### 3.4 Dataset Scaling and Subsetting Policy

Full-scale means every implemented, modality-compatible pipeline is evaluated, but it does not require indexing every corpus
item or generating answers for every query. The benchmark should separate **full pipeline coverage on capped deterministic
runs** from **secondary exhaustive all-data reruns** so that the artifact remains reproducible.

| Control | Default Policy |
|---------|----------------|
| Query subset | Keep full splits when the test set is already below roughly 2,000--3,000 queries; otherwise use deterministic `query_limit` |
| Retrieval query cap | `query_limit=3000` for large retrieval-only benchmarks unless a task-specific reason is documented |
| Generation/VLM query cap | `query_limit=2000` for expensive generation, judge, or multimodal runs unless cost is already amortized by cached outputs |
| Corpus subset | Use `min_corpus_cnt` where supported; treat it as a corpus target, not a strict hard cap |
| Randomness | Use seed 42 and publish selected query IDs plus selected corpus IDs |

`min_corpus_cnt` is the implemented corpus-size control in the current ingestor API. It should not be described as
`corpus_limit` in configs or manifests. Its semantics are: all gold/required corpus items for the selected queries are kept,
then random non-gold items are added until the target count is reached. If the required gold set is larger than the requested
target, the final corpus can exceed `min_corpus_cnt`.

Dataset-specific implications:

| Dataset Family | Query-Subset Policy | Corpus-Subset Policy |
|----------------|---------------------|----------------------|
| BEIR / MTEB | Small tasks run full; large tasks use `query_limit=3000` | `min_corpus_cnt` supported; always preserve qrel/gold documents |
| MrTyDi | English test queries can run full; multilingual or larger splits may use `query_limit` | Use `min_corpus_cnt` to avoid indexing the full large corpus when needed |
| BRIGHT | Single domains run full; all-domain aggregate can remain full by query count but may be domain-stratified for cost | Use `min_corpus_cnt` for large short-document corpora; long-document mode is naturally smaller |
| RAGBench | Selected small configs run full; large configs or all-config aggregate use `query_limit=2000` | `min_corpus_cnt` is ineffective; documents are bundled per example |
| CRAG | Use `query_limit=2000` or `3000` for main runs | `min_corpus_cnt` is ineffective; search context is query-specific |
| ViDoRe v1 | Selected subsampled datasets run full | 1:1 query-image structure makes `query_limit` and `min_corpus_cnt` effectively equivalent |
| ViDoRe v2/v3 / VisRAG | Single configs usually run full; all-config aggregates should be stratified or capped | `min_corpus_cnt` supported where the ingestor has a shared corpus |
| Open-RAGBench | Use `query_limit` if the selected path exceeds the cap | `min_corpus_cnt` is ineffective for the current query-to-section structure |

### Track D — Reproducibility Artifact Evaluation

**Goal:** Evaluate the artifact itself, not just model scores.

Measurements:

1. Figure-only reproduction time.
2. Small rerun time.
3. Full rerun resource estimate.
4. Artifact completeness.
5. Config/result checksum validation.
6. Deterministic score agreement for non-LLM portions.
7. Cached-output consistency for LLM portions.

### Track E — Extensibility and Community Reimplementation

**Goal:** Show that future researchers can add new RAG methods.

Measurements:

1. New plugin/pipeline implementation time.
2. Files changed and lines of method-specific code.
3. Tests required.
4. Whether executor/evaluator/reporting required modification.
5. Whether the new method can be included in benchmark configs immediately.

---

## 4. Reproducibility Protocol

### 4.1 Manifests

Each run must emit or archive:

- git commit;
- dependency lock hash;
- dataset source and dump hash;
- ingestion config;
- embedding model and embedding artifact hash;
- pipeline configs;
- metric configs;
- LLM model/version/provider/temperature/max token settings;
- prompt templates;
- cache keys and cached generations;
- per-query outputs;
- per-query metric scores;
- aggregate result tables;
- figure-generation script version.

### 4.2 Fidelity Cards

Each reimplemented method must include:

| Field | Description |
|-------|-------------|
| Citation | Original paper/source |
| Algorithm summary | One-paragraph method description |
| Implemented modules | Code/config files |
| Shared interfaces used | Retrieval, generation, metric, ingestor, plugin |
| Defaults | Hyperparameters in this benchmark |
| Deviations | Paper/code differences |
| Missing details | Assumptions because original paper was underspecified |
| Tests | Evidence of implementation behavior |
| Artifact links | Config/results/cache |

### 4.3 Dataset Compatibility Rules

- Retrieval metrics require retrieval ground truth/qrels.
- Reference-based generation metrics require `generation_gt` or equivalent answer annotations.
- LLM-judge metrics require prompt, model, sampling, and cached output manifests.
- Multimodal metrics require explicit modality support and artifact manifests.

These rules prevent accidentally overstating results on datasets that do not support a metric.

### 4.4 Closed-Model Policy

Closed or hosted LLM calls are allowed, but reproducibility depends on artifacts:

1. Record model name/version and provider.
2. Use temperature 0 for main runs.
3. Save prompts and responses.
4. Publish cached generations where license/privacy permits.
5. Treat reruns through the same API as approximate due to model drift.
6. Prefer figure reproduction from cached outputs for review.

---

## 5. Analysis Plan

### 5.1 Framework Coverage

Report coverage counts for:

- ingestors;
- retrieval pipelines;
- generation pipelines;
- metrics;
- plugin/config surfaces;
- tests/docs per component.

### 5.2 Fidelity and Deviation Analysis

Report:

- method-by-method fidelity card summary;
- common sources of non-exact reproduction: prompts, unavailable checkpoints, closed APIs, missing hyperparameters, dataset preprocessing differences;
- how AutoRAG-Research makes these deviations explicit.

### 5.3 Benchmark Results

Report:

- retrieval leaderboards;
- generation leaderboards;
- runtime/token/cost tables;
- storage and artifact sizes;
- representative failure cases;
- optional metric/dataset/backend sensitivity as secondary observations.

### 5.4 Reproduction Results

Report:

- time to regenerate figures;
- time to restore and rerun a small dataset;
- deterministic score agreement;
- metric recomputation match;
- artifact completeness score;
- reviewer instructions.

### 5.5 Extensibility Results

Report:

- new method/plugin case study;
- implementation effort;
- integration path;
- whether benchmark inclusion required only a YAML config plus component code.

---

## 6. Resource Planning

Full-scale experiments are expensive because generation pipelines may make many LLM calls per query and multimodal pipelines require larger models and artifacts.

### 6.1 Pilot vs Full-Scale Split

| Tier | Scope | Purpose |
|------|-------|---------|
| **Pilot / smoke run** | small representative text retrieval + generation + artifact evaluation | validate configs, executor, evaluator, caching, and reporting before scaling |
| **Full-scale text** | all text datasets, all modality-compatible text retrieval/generation pipelines, all valid metrics | core benchmark report demonstrating framework breadth |
| **Full-scale multimodal** | visual/multimodal datasets and all modality-compatible retrieval/generation pipelines | required extension showing that the same framework covers multimodal RAG |
| **Living benchmark** | plugin/community submissions | long-term project direction |

### 6.2 Cost Controls

- Run deterministic retrieval first to validate dataset dumps, qrels, and result persistence.
- Use generation datasets only where answers exist.
- Execute simple pipelines first as a smoke-test sequence, then run all advanced/high-cost pipelines.
- Include RETRO\*, Self-RAG, RAG-Critic, SPD-RAG, AutoThinkRAG, and multimodal generation in the full-scale matrix wherever valid.
- If the main paper has space limits, move detailed per-pipeline tables to appendix; do not drop the experiments.
- Cache all LLM calls.
- Publish outputs so reviewers do not need to pay API costs.
- Sample expensive LLM-judge metrics only if they are secondary.

### 6.3 Hardware/API Planning

Use local vLLM or hosted APIs only after a pilot validates:

- throughput;
- average prompt/output length;
- token cost;
- failure/retry rate;
- cache hit behavior;
- artifact sizes.

Do not make paper claims about unreleased or unvalidated model versions. If a planned model is unavailable, use the actual available model and update all manifests and claims.

---

## 7. Paper Structure for Full-Scale Version

### Abstract
AutoRAG-Research is a framework for reproducible RAG reimplementation. The paper reports a large-scale benchmark produced with this framework and releases artifacts enabling figure reproduction and selected reruns.

### 1. Introduction
- RAG reproducibility challenge.
- Cost of ad hoc reimplementation.
- AutoRAG-Research solution.
- Contributions and artifacts.

### 2. Related Work
- RAG frameworks and benchmarks.
- Reproducibility and artifact evaluation.
- Reimplemented RAG method families.

### 3. Framework
- Architecture.
- Data schema.
- Pipeline abstractions.
- Metrics and evaluator.
- Reporting and plugin system.

### 4. Reimplementation Protocol
- Fidelity cards.
- Dataset compatibility.
- Config manifests.
- Closed-model/cache policy.
- Artifact release.

### 5. Benchmark Setup
- Datasets.
- Pipelines.
- Metrics.
- Environment.
- Cost/runtime assumptions.

### 6. Results
- Coverage and fidelity.
- Retrieval benchmark.
- Generation benchmark.
- Multimodal full-scale results, with detailed tables in appendix if needed.
- Reproducibility measurements.
- Extensibility case study.

### 7. Discussion
- Lessons for RAG reproducibility.
- What exact reproduction cannot solve.
- How future researchers can use/extend artifacts.

### 8. Conclusion
AutoRAG-Research makes RAG method reimplementation more systematic, comparable, and reusable.

---

## 8. Artifact Package

Required artifacts:

1. Anonymized code repository.
2. Experiment YAML configs.
3. Dataset dump manifests and restore instructions.
4. Precomputed embeddings or embedding manifests.
5. Cached retrieval outputs.
6. Cached generation outputs.
7. Metric-score exports.
8. Figure/table reproduction scripts.
9. Fidelity/deviation cards.
10. Environment manifest.
11. README with figure-only, small-rerun, and full-rerun paths.

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Tool-paper framing | Medium | High | Center reproducible reimplementation, not UI/features |
| Exact reproduction overclaim | Medium | High | Use faithful documented reimplementation and deviation cards |
| Dataset/metric incompatibility | Medium | High | Enforce retrieval-only vs generation-GT tracks |
| Pipeline name/result contamination | Medium | High | Unique names and config manifests |
| Closed model drift | High | Medium | Cache outputs and make figure reproduction artifact-based |
| Full-scale cost | High | Medium | Deterministic subsets, cached outputs, staged execution, and token/runtime manifests; do not remove implemented pipelines |
| Multimodal instability | Medium | Medium | Include multimodal-compatible pipelines in full-scale runs and report limitations/failure modes transparently |
| Missing reproduction scripts | Medium | High | Prioritize figure/table reproduction scripts before submission |
| Artifact size too large | Medium | Medium | Separate minimal figure artifacts from full DB dumps |
| Similarity to existing benchmarks | Medium | Medium | Emphasize reimplementation fidelity, artifact reproducibility, and extensibility |

---

## 10. Success Criteria

The full-scale project succeeds if it delivers:

1. A documented library of reimplemented RAG methods.
2. A benchmark suite generated through shared configs and schema.
3. Fidelity/deviation cards for representative paper methods.
4. Reproducible artifacts allowing fast figure/table regeneration.
5. A small rerun path demonstrating executor/evaluator correctness.
6. Runtime/cost/storage reporting for realistic reproduction planning.
7. An extension path for future community reimplementations.

---

*Full-scale roadmap corrected for reproducibility-framework positioning: 2026-04-24*
*Scope: Reimplemented RAG methods + reproducible artifacts + empirical benchmark report*
