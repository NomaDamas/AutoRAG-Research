# AutoRAG-Research NeurIPS 2026 Experimentation Plan — Reproducibility-Centered Revision
## A Reimplementation Framework for Reproducible RAG Research

**Target:** NeurIPS 2026 Datasets & Benchmarks / Evaluations-style submission, or TMLR/JMLR if the artifact package needs a longer review cycle
**Immediate deadlines:** Abstract May 4, 2026 | Full Paper May 6, 2026
**Branch:** `experiment/paper-publication`
**Working Title:** *AutoRAG-Research: A Unified Reimplementation Framework for Reproducible RAG Experiments*

---

## 0. Corrected Paper Direction

This paper is **not** primarily a critique of RAG metrics, nor should it be framed as a paper whose main claim is that evaluation metrics change the winner.

The correct central claim is:

> AutoRAG-Research increases the reproducibility of RAG research by providing a unified framework for reimplementing published retrieval and generation pipelines, ingesting benchmark datasets into a common schema, running experiments from declarative configs, and publishing reproducible artifacts. The paper introduces the framework and reports experiments produced with that framework.

Metric-dependent ranking changes may appear as a **secondary observation**, but they are not the thesis. The thesis is **reproducible reimplementation infrastructure plus empirical validation**.

### What the paper should emphasize

1. **Reimplementation fidelity:** How faithfully existing RAG methods can be reconstructed under a shared abstraction.
2. **Reproducible execution:** How experiments can be rerun from configs, database dumps, precomputed embeddings, and result artifacts.
3. **Comparability:** How the same datasets, retrieval outputs, generation models, and metric implementations enable fairer comparisons than ad hoc per-paper setups.
4. **Extensibility:** How new pipelines, metrics, and datasets can be added without rebuilding the experiment harness.
5. **Empirical report:** What happens when representative RAG pipelines are reimplemented and run under the framework.

### What the paper should avoid as the main story

- “RAG metrics are wrong.”
- “Metric choice invalidates prior RAG work.”
- “The main novelty is ranking instability.”
- “Exact reproduction of every prior paper result.”

The stronger and safer claim is:

> AutoRAG-Research supports faithful, documented, and rerunnable RAG reimplementation studies; exact numerical equivalence to every original paper is not always possible because model versions, closed APIs, prompts, preprocessing, and unreported hyperparameters differ.

---

## 1. Scientific Questions

### RQ1 — Framework Reimplementation Coverage
**Can a single framework express a diverse set of published RAG retrieval and generation pipelines without method-specific experiment harnesses?**

Evidence to report:
- Number of reimplemented pipeline families.
- Which parts are faithful to the source papers.
- Which parts required documented deviations.
- Shared abstractions used by all methods: data schema, pipeline configs, executor, evaluator, service/UoW/database layers.

### RQ2 — Reproducible Execution and Artifact Reuse
**Can another researcher reproduce the reported experiment tables and figures from released configs, pre-ingested datasets, cached outputs, and result artifacts?**

Evidence to report:
- Figure-only reproduction time from precomputed result files.
- Database restore + metric recomputation time on a small public subset.
- Hashes/manifests for configs, dataset dumps, and result artifacts.
- Deterministic rerun agreement for retrieval pipelines and non-LLM metrics.
- Cached/recorded generations for closed or remote LLM calls.

### RQ3 — Empirical Benchmarking Under a Shared Protocol
**What performance patterns are observed when representative RAG pipelines are reimplemented and evaluated under one controlled protocol?**

Evidence to report:
- Retrieval results across text datasets.
- Generation results on datasets with generation ground truth.
- Runtime/cost/token-usage comparison across methods.
- Failure examples illustrating why reproducible comparison matters.

### RQ4 — Extensibility for Future Reimplementation
**How much work is required to add a new method or metric to the framework?**

Evidence to report:
- A small plugin or new-pipeline case study.
- Files changed, lines of method-specific code, tests required, config required.
- Whether the new component automatically participates in executor/evaluator/reporting flows.

---

## 2. Minimum Viable Experiment Matrix

The matrix is intentionally split into **retrieval-only**, **generation**, and **multimodal** tracks so that metrics are only
used when the dataset actually supports them. For the full-scale benchmark, every implemented pipeline should be evaluated on
all modality-compatible datasets.

### 2.1 Track A — Retrieval Reimplementation Benchmark

| Dataset | Type | Why Included | Metrics |
|---------|------|--------------|---------|
| **BEIR (scifact)** | scientific text retrieval | Common public IR benchmark; clean qrels | Recall@10, nDCG@10, MRR, MAP |
| **MrTyDi (english)** | open-domain retrieval | Larger multilingual-derived retrieval corpus | Recall@10, nDCG@10, MRR, MAP |
| **BRIGHT (biology)** | reasoning-oriented retrieval | Harder reasoning-style retrieval setting; also has answer fields | Recall@10, nDCG@10, MRR, MAP |

**Important compatibility rule:** BEIR and MrTyDi should be treated as retrieval datasets unless answer annotations are explicitly added. They must not be used for reference-based generation metrics by default.

### 2.2 Track B — Generation/RAG Reimplementation Benchmark

Use only datasets with `generation_gt` or answer fields in the ingested schema.

| Dataset | Type | Why Included | Metrics |
|---------|------|--------------|---------|
| **RAGBench (techqa or covidqa)** | long-form RAG QA | Native generated/reference answer field | ROUGE-L, BERTScore F1, Token F1, Exact Match, BARTScore F1 |
| **CRAG (dev)** | web/search-grounded QA | Query-specific search results and generation answers | ROUGE-L, BERTScore F1, Token F1, Exact Match |
| **BRIGHT (biology)** | reasoning QA | Retrieval + answer field supports end-to-end RAG study | ROUGE-L, BERTScore F1, Token F1, BARTScore F1 |

LLM-judge metrics such as response relevancy may be used as **optional auxiliary metrics** if API cost and access are stable, but they are not required for the core reproducibility claim.

### 2.3 Track C — Multimodal Reimplementation Benchmark

| Dataset | Type | Why Included | Status |
|---------|------|--------------|--------|
| **ViDoRe (arxivqa_test_subsampled)** | visual document retrieval | Shows multimodal schema/pipeline support | Full-scale visual retrieval track |
| **VisRAG (ChartQA)** | chart/document QA | Shows visual RAG generation support | Full-scale visual generation track |

If page limits are tight, multimodal details can be summarized in the main text and expanded in appendix tables, but the
full-scale artifact should include the runs.

---

## 3. Pipeline Selection

Full-scale evaluation should include **every implemented pipeline wherever the dataset modality and ground truth make the
pipeline valid**. Execution may be staged for engineering convenience, but expensive pipelines should not be excluded from
the full-scale experiment matrix. If page limits are tight, the main paper can summarize full-scale results and place detailed
per-pipeline tables in the appendix; the experiments themselves should still be run and released as artifacts.

### 3.1 Retrieval Pipelines

| Pipeline | Family | Role in Paper |
|----------|--------|---------------|
| **BM25** | sparse lexical baseline | Baseline and sanity check |
| **Vector Search** | dense retrieval | Standard neural retrieval reference |
| **Hybrid RRF** | sparse+dense fusion | Demonstrates compositional retrieval wrappers |
| **Hybrid CC** | score-level sparse+dense fusion | Alternative fusion implementation under the same retrieval abstraction |
| **HyDE** | LLM-assisted query expansion | Demonstrates reimplementation of a published LLM-assisted retriever |
| **Query Rewrite** | LLM query reformulation | Demonstrates reusable LLM-backed preprocessing |
| **RETRO\*** | LLM reranking | High-cost but required full-scale reranking case study |
| **Power of Noise** | retrieval perturbation/diagnostic wrapper | Shows robustness/ablation-style retrieval support |
| **Question Decomposition Retrieval** | sub-query retrieval | Demonstrates complex-query decomposition in retrieval |
| **HEAVEN** | visual document retrieval | Included on multimodal/visual datasets where valid |

### 3.2 Generation Pipelines

| Pipeline | Family | Role in Paper |
|----------|--------|---------------|
| **BasicRAG** | retrieve-then-generate baseline | Baseline |
| **IRCoT** | interleaved retrieval + reasoning | Published multi-step reimplementation case |
| **ET2RAG** | ensemble/subset selection | Demonstrates multi-call context selection |
| **MAIN-RAG** | multi-agent filtering | Demonstrates agentic/multi-call RAG reimplementation |
| **Question Decomposition** | decomposition-based RAG | Demonstrates query decomposition abstraction |
| **Self-RAG** | self-reflective generation | Required advanced self-reflection reimplementation case |
| **RAG-Critic** | critic-guided correction | Required critic/revision-style generation case |
| **SPD-RAG** | speculative/parallel RAG | Required high-cost generation strategy case |
| **AutoThinkRAG** | adaptive reasoning | Required adaptive reasoning path case, including multimodal variants where valid |
| **VisRAG Generation** | vision-language generation | Included on visual/multimodal datasets where valid |

The cost-control mechanism is not to drop these pipelines, but to use deterministic subsets, cached LLM outputs, local/vLLM
serving when possible, and clear token/runtime manifests.

---

## 4. Experimental Protocol

### 4.1 Reproducibility Controls

| Control | Required Practice | Why It Matters |
|---------|-------------------|----------------|
| **Config manifest** | Save every YAML config used for each run | Enables exact experiment reconstruction |
| **Unique pipeline names** | Use names such as `basic_rag__bm25__gpt4omini` | Prevents DB result reuse or contamination |
| **Dataset manifest** | Record ingestor, subset/split, `query_limit`, `min_corpus_cnt`, selected query/corpus IDs, embedding model, dump hash | Makes preprocessing reproducible |
| **Environment manifest** | Record git commit, lockfile, Python version, Docker/PostgreSQL/VectorChord versions | Makes execution environment auditable |
| **LLM manifest** | Record model name, provider, temperature, max tokens, prompt template, and cache key | Closed/remote model outputs are otherwise hard to reproduce |
| **Seed policy** | Use seed 42 for sampling and deterministic components | Enables rerunnable subsets |
| **Caching** | Cache LLM calls and publish generation outputs | Reviewers can reproduce figures without API keys |
| **Result schema** | Export per-query retrieval/generation/metric rows | Supports independent reanalysis |
| **Deviation log** | Document every divergence from original papers | Prevents overclaiming exact reproduction |

### 4.1.1 Dataset Subset and Corpus-Size Policy

The paper should use bounded, deterministic subsets for large benchmarks while keeping naturally small benchmarks intact.
The goal is reproducible reimplementation evidence, not exhaustive leaderboard coverage.

- **Query cap:** use the full test split when it is already below roughly 2,000--3,000 queries. For larger datasets, use
  `query_limit=3000` for retrieval-only experiments and `query_limit=2000` for expensive generation or multimodal runs unless
  a specific ablation requires a different cap.
- **Corpus target:** AutoRAG-Research ingestors do not expose a hard `corpus_limit` parameter. The current corpus-size control
  is `min_corpus_cnt`.
- **`min_corpus_cnt` semantics:** include all gold/required corpus items for the selected queries first, then sample random
  non-gold corpus items until the requested target count is reached. Because gold items are mandatory, the final corpus size
  may exceed `min_corpus_cnt`; it is a reproducible corpus target, not a strict maximum.
- **Supported corpus sampling:** BEIR, MTEB, MrTyDi, BRIGHT, ViDoRe v2/v3, and VisRAG can use `min_corpus_cnt` to build
  reusable corpus subsets. ViDoRe v1 has a 1:1 query-image structure, so `query_limit` and `min_corpus_cnt` are effectively
  equivalent. RAGBench, CRAG, and Open-RAGBench are query-centric datasets where `min_corpus_cnt` is ineffective; use
  `query_limit` there.
- **Manifest requirement:** every subset must publish the random seed, query IDs, gold/required corpus IDs, sampled non-gold
  corpus IDs, `query_limit`, `min_corpus_cnt`, ingestor name/version, and dump hash.

Recommended main-plan subset decisions:

| Dataset | Main-plan Decision | Rationale |
|---------|--------------------|-----------|
| BEIR SciFact | Full query split; optional `min_corpus_cnt` only if indexing cost matters | Small enough to keep intact |
| MrTyDi English | Full test queries; use `min_corpus_cnt` if full-corpus indexing is too costly | Query count is modest, corpus is large |
| BRIGHT biology | Full domain | Very small query set; corpus target only if using short-document corpus at scale |
| RAGBench techqa/covidqa | Full selected config | Selected configs are below the query cap |
| CRAG dev | Use `query_limit=2000` or `3000` | Larger query-centric generation benchmark |
| ViDoRe arxivqa_test_subsampled / VisRAG ChartQA | Full selected dataset | Already small full-scale multimodal datasets |

### 4.2 Fair Comparison Controls

These controls support comparability but are secondary to the reproducibility thesis.

| Control | Default |
|---------|---------|
| Text embedding model | `BAAI/bge-large-en-v1.5` or the exact precomputed embedding model recorded in the dump |
| Generation backend | One fixed model for the main paper, e.g. GPT-4o-mini-class or local vLLM model |
| Temperature | 0.0 for main runs |
| Retrieval top-k | 10 unless the pipeline paper requires a different internal setting |
| Context budget | Fixed maximum context tokens across generation pipelines |
| Retry policy | 3 retries with exponential backoff |
| Metrics | Retrieval metrics for retrieval datasets; reference-based generation metrics only where `generation_gt` exists |

### 4.3 Fidelity Policy

Each reimplementation must have a short fidelity card:

| Field | Description |
|-------|-------------|
| Original paper/reference | Citation and method name |
| Implemented components | What is implemented in AutoRAG-Research |
| Required assumptions | Hyperparameters or prompts not fully specified by paper |
| Deviations | Differences from original implementation |
| Test evidence | Unit/integration tests or smoke runs |
| Reproducibility artifact | Config, result file, and cached output path |

This fidelity card is more important than claiming exact score matching.

---

## 5. Analysis Plan

### 5.1 Framework Coverage and Fidelity

Primary tables:
1. Pipeline inventory: method, source paper, abstraction used, config file, test evidence.
2. Dataset inventory: ingestor, schema fields, retrieval GT availability, generation GT availability, precomputed embedding status.
3. Deviation table: paper method vs AutoRAG-Research implementation choices.

### 5.2 Reproducibility Evaluation

Primary measurements:
1. **Figure-only reproduction:** time to regenerate all paper figures from released result artifacts.
2. **Small rerun reproduction:** time to restore one dataset dump and rerun a small subset.
3. **Deterministic retrieval agreement:** exact top-k match or metric-score match on rerun.
4. **Metric recomputation agreement:** regenerated metric scores from cached outputs match reported tables.
5. **Artifact completeness:** config/result/cache/manifests are present for each reported run.

Suggested table:

| Reproduction Mode | Required Inputs | Expected Time | Expected Output |
|-------------------|-----------------|---------------|-----------------|
| Figure-only | result parquet/csv + scripts | <10 minutes | all paper figures/tables |
| Small public rerun | one DB dump + configs | <1 hour | retrieval/generation subset scores |
| Full rerun | all datasets + API/local models | days/weeks | full benchmark regeneration |

### 5.3 Empirical Benchmark Results

Report results as evidence that the framework can run real reimplementation studies:

- Retrieval leaderboard per dataset.
- Generation leaderboard per dataset with runtime/cost columns.
- Per-query outputs and error examples.
- Simple baseline vs complex method comparison as an empirical observation, not the central thesis.
- Metric sensitivity can appear as an appendix or secondary analysis, not the paper title or main contribution.

### 5.4 Extensibility Case Study

If time permits, add one bounded case study:

- Implement a small new retrieval/generation plugin or metric plugin.
- Report files changed, lines of method-specific code, tests, and config.
- Show that the plugin can be run by the existing executor and reported by the existing evaluator/reporting layer.

---

## 6. Paper Structure

### Abstract
- Problem: RAG research is hard to reproduce because each paper uses different data formats, pipeline code, metrics, prompts, and execution harnesses.
- Method: AutoRAG-Research unifies dataset ingestion, retrieval/generation pipelines, metric evaluation, database-backed persistence, configs, and reporting.
- Evidence: Reimplement representative RAG pipelines and run them across retrieval and generation datasets with released artifacts.
- Contribution: A reproducible reimplementation framework plus empirical benchmark artifacts.

### 1. Introduction
- RAG reproducibility problem.
- Why ad hoc reimplementation is costly.
- AutoRAG-Research as a unified reimplementation framework.
- Contributions: framework, reimplemented methods, reproducible artifacts, empirical results.

### 2. Related Work
- RAG frameworks and benchmarks.
- Reproducibility and artifact evaluation in ML/NLP.
- RAG pipeline papers reimplemented in this work.
- Gap: lack of a reusable reimplementation harness that spans retrieval, generation, metrics, data, and persistence.

### 3. AutoRAG-Research Framework
- Layered architecture: config/executor → pipeline → service → UoW/repository → schema.
- Retrieval and generation pipeline abstractions.
- Dataset ingestors and precomputed embeddings.
- Metric system and evaluator.
- Plugin and config system.
- Reporting/leaderboard service.

### 4. Reimplementation Protocol
- Dataset selection and compatibility rules.
- Pipeline fidelity cards.
- Config manifests and unique naming.
- LLM/prompt/cache policy.
- Artifact packaging.

### 5. Experiments
- Track A: retrieval reimplementation benchmark.
- Track B: generation/RAG benchmark.
- Track C: multimodal reimplementation benchmark; detailed tables may be appendix, but full-scale runs are included.
- Reproducibility rerun evaluation.
- Extensibility case study if available.

### 6. Results
- Framework coverage and fidelity table.
- Retrieval benchmark results.
- Generation benchmark results.
- Reproduction-time and artifact-completeness results.
- Cost/runtime analysis.

### 7. Discussion and Limitations
- Exact reproduction vs faithful documented reimplementation.
- Closed model drift and prompt/version sensitivity.
- Dataset licensing and redistribution limits.
- Multimodal and human-evaluation limitations.

### 8. Conclusion
- AutoRAG-Research lowers the cost of reproducible RAG reimplementation and produces reusable artifacts for future RAG research.

---

## 7. Artifact Requirements

Minimum artifact package:

1. **Anonymized repository** with framework code and experiment configs.
2. **Experiment configs** for every reported run.
3. **Dataset dump manifests** and restoration instructions.
4. **Precomputed retrieval/generation outputs** for figure reproduction.
5. **Metric score exports** at per-query granularity.
6. **Figure/table reproduction script**.
7. **Fidelity/deviation cards** for every reimplemented paper method.
8. **Environment manifest** with git commit, dependency lock, Docker/PostgreSQL/VectorChord versions.

Reviewer-facing reproduction modes:

| Mode | Goal | Required Resources |
|------|------|--------------------|
| Figure-only | Verify reported numbers/figures quickly | CPU only |
| Small rerun | Verify executor/evaluator path | PostgreSQL + one small dump + optional API/local mock |
| Full rerun | Recreate all benchmark outputs | Full datasets + configured LLM/embedding backends |

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Paper is perceived as “just a tool” | High | Frame around reproducible reimplementation and artifact evaluation; empirical runs validate framework utility |
| Exact reproduction claims are challenged | High | Use “faithful documented reimplementation,” publish deviation cards, avoid exact-score overclaims |
| Some datasets lack generation ground truth | High | Split retrieval-only and generation tracks; do not run reference-based generation metrics where `generation_gt` is absent |
| Configs do not cover the full matrix | High | Create paper-specific configs with unique pipeline names and archived manifests |
| Closed model/API drift | Medium | Cache outputs, record model versions, publish generations; temperature 0 for main runs |
| Pipeline costs grow too large | Medium | Keep deterministic subset caps, cache all LLM outputs, stage execution, and report token/runtime manifests; do not remove implemented pipelines from full-scale |
| Reviewers want stronger empirical novelty | Medium | Emphasize reproducibility measurements, fidelity audit, and cost/runtime analysis, not only leaderboard scores |
| Artifact package too heavy | Medium | Provide figure-only artifacts plus optional DB dumps/full rerun path |

---

## 9. What Was Changed From the Metric-Sensitivity Revision

| Previous Emphasis | Corrected Emphasis |
|-------------------|--------------------|
| “When evaluation changes the winner” | “Reproducible RAG reimplementation framework” |
| Metric-centered critique as core thesis | Metric analysis as optional secondary observation |
| 3 datasets × 4 × 4 primarily for metric-stability science | Representative benchmark to validate framework utility |
| Main contribution: evaluation design changes conclusions | Main contribution: reproducible framework + reimplemented methods + artifacts |
| Strong critique of SOTA claims | Constructive infrastructure for rerunnable RAG research |

---

*Revised for corrected paper direction: 2026-04-24*
*Scope: Reproducibility-centered framework paper with representative experiments and artifact evaluation*
