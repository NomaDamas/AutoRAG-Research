# AutoRAG-Research NeurIPS 2026 Experimentation Plan
## A Reimplementation Framework for Reproducible RAG Research

**Target:** NeurIPS 2026 Datasets & Benchmarks / Evaluations-style submission, TMLR, or JMLR
**Deadlines:** Abstract May 4, 2026 | Full Paper May 6, 2026
**Branch:** `experiment/paper-publication`
**Working Title:** *AutoRAG-Research: A Unified Reimplementation Framework for Reproducible Retrieval-Augmented Generation Experiments*

---

## 1. Paper Positioning and Core Contribution

### 1.1 Problem Statement

RAG research is difficult to reproduce and extend because each paper often differs in:

- dataset preprocessing and chunking;
- retrieval and generation pipeline wiring;
- prompt templates and LLM backends;
- metric implementations;
- result persistence format;
- experiment orchestration scripts;
- undocumented hyperparameters and implementation assumptions.

As a result, researchers who want to reproduce or compare RAG methods often need to rebuild substantial infrastructure before they can test the method itself.

### 1.2 Core Contribution

AutoRAG-Research is a **reproducible reimplementation framework** for RAG research.

It provides:

1. **Unified pipeline abstractions** for retrieval and generation methods.
2. **Dataset ingestors** that map heterogeneous benchmarks into a common database schema.
3. **Config-driven execution** through YAML pipeline/metric/experiment definitions.
4. **Database-backed result persistence** for retrieval results, generation outputs, and metric scores.
5. **Metric evaluation services** that can recompute scores from stored outputs.
6. **Reporting tools** for cross-dataset and cross-pipeline comparison.
7. **Plugin surfaces** for adding new methods without rewriting the executor/evaluator.
8. **Artifact packaging practices** that allow reviewers and future researchers to reproduce figures from released outputs.

### 1.3 Scientific Claims

1. **Reimplementation Framework Claim:** AutoRAG-Research can express diverse published RAG methods under shared retrieval/generation/data/metric abstractions.
2. **Reproducibility Claim:** Experiments run through AutoRAG-Research can be reproduced from configs, dataset dumps, cached outputs, and result artifacts.
3. **Fidelity Claim:** Reimplemented methods can be accompanied by explicit fidelity/deviation cards, making differences from original papers auditable.
4. **Empirical Utility Claim:** Running representative reimplemented RAG pipelines under one protocol yields useful benchmark results and diagnostic comparisons.
5. **Extensibility Claim:** New pipelines, metrics, and datasets can be added through configs/plugins while reusing the same execution and evaluation infrastructure.

### 1.4 Non-Claims

The paper should avoid overclaiming:

- It does **not** claim exact numerical reproduction of every original paper result.
- It does **not** claim that all RAG metrics are unreliable.
- It does **not** make metric-induced ranking instability the central contribution.
- It does **not** claim that one benchmark ranking establishes universal pipeline superiority.

The intended claim is faithful, documented, and rerunnable **reimplementation**, not perfect historical score replication.

---

## 2. Experimental Design

The experiments are designed to demonstrate that AutoRAG-Research supports real RAG reimplementation work end-to-end.

### 2.1 Experiment Tracks

| Track | Purpose | Output |
|-------|---------|--------|
| **A. Retrieval Reimplementation** | Show that retrieval methods can be reimplemented and compared under shared qrels/schema/metrics | Retrieval result tables and reproducibility checks |
| **B. Generation/RAG Reimplementation** | Show that generation pipelines can compose with retrieval pipelines and be evaluated on datasets with answer GT | Generation benchmark tables and token/runtime/cost analysis |
| **C. Reproducibility Artifact Evaluation** | Show that results can be regenerated from released configs/artifacts | Figure-only and small-rerun reproduction measurements |
| **D. Extensibility Case Study** | Show that a new method/metric can be added without rewriting the harness | Plugin or pipeline case-study table |
| **E. Multimodal Reimplementation** | Demonstrate that the schema can support image/multimodal RAG | Full-scale visual/multimodal results, with detailed tables in appendix if needed |

### 2.2 Dataset Selection

Dataset choice must respect metric compatibility. Retrieval-only datasets should not be forced into generation evaluation unless generation ground truth is present.

#### Retrieval Track

| Dataset | Type | Use |
|---------|------|-----|
| **BEIR (scifact)** | scientific text retrieval | Retrieval benchmark and qrels sanity check |
| **MrTyDi (english)** | open-domain text retrieval | Larger retrieval benchmark |
| **BRIGHT (biology)** | reasoning-oriented text retrieval | Harder retrieval benchmark and bridge to generation |

#### Generation Track

| Dataset | Type | Use |
|---------|------|-----|
| **RAGBench (techqa or covidqa)** | RAG QA with answer field | Reference-based generation evaluation |
| **CRAG (dev)** | web/search-grounded QA | Generation evaluation with query-specific search contexts |
| **BRIGHT (biology)** | reasoning QA | End-to-end retrieval+generation evaluation |

#### Multimodal Track

| Dataset | Type | Use |
|---------|------|-----|
| **ViDoRe (arxivqa_test_subsampled)** | visual document retrieval | Full-scale multimodal retrieval evaluation |
| **VisRAG (ChartQA)** | visual/chart QA | Full-scale multimodal generation evaluation |

#### Dataset Subsetting and Corpus Control

Large datasets should be evaluated through deterministic, manifest-backed subsets. A test dataset with roughly 2,000--3,000
queries is sufficient for the paper's reproducibility claim; smaller datasets should be kept intact rather than upsampled or
artificially expanded.

- Use full query splits when they are already below the cap.
- Use `query_limit=3000` for large retrieval-only runs and `query_limit=2000` for expensive generation or multimodal runs.
- Use `min_corpus_cnt` for corpus-size control where the ingestor supports it. This is not a hard `corpus_limit`: gold or
  required corpus items for the selected queries are always included first, then non-gold items are sampled up to the target.
- Record query IDs, gold corpus IDs, sampled non-gold corpus IDs, seed, `query_limit`, `min_corpus_cnt`, ingestor version, and
  dump hash in the dataset ingestion manifest.
- Treat RAGBench, CRAG, and Open-RAGBench as query-centric datasets where corpus-size limiting is ineffective; use
  `query_limit` for those datasets.

For the current MVP matrix, CRAG is the main dataset that needs a query subset. MrTyDi English can keep the full test query
split but may need `min_corpus_cnt` if indexing the full corpus is too expensive.

### 2.3 Pipeline Selection

The full-scale benchmark must run **all implemented pipelines** wherever the dataset modality and metric compatibility make
them valid. A smaller pilot can be used to validate infrastructure, but expensive methods should not be excluded from the
full-scale experiment plan; their detailed tables can move to appendix while the artifacts remain released.

#### Retrieval Pipelines

| Pipeline | Why Included |
|----------|--------------|
| **BM25** | Sparse lexical baseline and deterministic sanity check |
| **Vector Search** | Dense retrieval baseline |
| **Hybrid RRF** | Demonstrates compositional pipeline reuse |
| **Hybrid CC** | Alternative sparse+dense score fusion strategy |
| **HyDE** | Published LLM-assisted retrieval reimplementation |
| **Query Rewrite** | LLM-based query transformation reimplementation |
| **RETRO\*** | High-cost but required LLM reranking case study |
| **Power of Noise** | Retrieval perturbation/diagnostic case |
| **Question Decomposition Retrieval** | Sub-query retrieval case |
| **HEAVEN** | Visual retrieval case for multimodal datasets |

#### Generation Pipelines

| Pipeline | Why Included |
|----------|--------------|
| **BasicRAG** | Baseline retrieve-then-generate method |
| **IRCoT** | Interleaved retrieval/reasoning paper reimplementation |
| **ET2RAG** | Ensemble/subset-selection style generation pipeline |
| **MAIN-RAG** | Multi-agent filtering RAG reimplementation |
| **Question Decomposition** | Decomposition-style generation/retrieval composition |
| **Self-RAG** | Self-reflective generation reimplementation |
| **RAG-Critic** | Critic-guided correction/revision case |
| **SPD-RAG** | Speculative/parallel generation strategy case |
| **AutoThinkRAG** | Adaptive reasoning path case, including multimodal variants where valid |
| **VisRAG Generation** | Vision-language generation case for visual/multimodal datasets |

Cost is handled through `query_limit`, `min_corpus_cnt`, caching, and token/runtime manifests, not by removing implemented
pipelines from the full-scale experiment matrix.

### 2.4 Metrics

#### Retrieval Metrics

- Recall@10
- nDCG@10
- MRR
- MAP
- Precision/F1 where appropriate

#### Generation Metrics

Use reference-based metrics only when `generation_gt` exists:

- ROUGE-L
- BERTScore F1
- Token F1
- Exact Match
- BARTScore F1
- UniEval dimensions where retrieved context/reference requirements are satisfied

LLM-as-judge metrics are optional because API access, pricing, and model drift complicate reproduction. If included, publish judge prompts, model versions, sampling policy, and cached judge outputs.

---

## 3. Reproducibility Protocol

### 3.1 Required Manifests

Every reported run should have:

1. Experiment config file.
2. Pipeline config files.
3. Metric config files.
4. Dataset ingestion manifest, including subset/split, `query_limit`, `min_corpus_cnt`, selected query/corpus IDs, seed, and
   source/dump hash.
5. Git commit hash.
6. Dependency lockfile hash.
7. Database dump hash or source dataset hash.
8. Embedding model identifier and embedding artifact hash.
9. LLM model identifier, temperature, max token settings, and prompt templates.
10. Result artifact hash.

### 3.2 Unique Naming Policy

Pipeline names must encode the method combination. Examples:

- `bm25__scifact`
- `vector_search__bge_large__scifact`
- `hyde__qwen__bge_large__ragbench_techqa`
- `basic_rag__bm25__gpt4omini__ragbench_techqa`
- `ircot__hybrid_rrf__gpt4omini__bright_biology`

This prevents accidental reuse of database rows from a previous config with the same pipeline name.

### 3.3 Fidelity Cards

For each reimplemented paper method, include:

| Field | Required Content |
|-------|------------------|
| Method | Paper name and citation |
| Scope | Which algorithmic components are implemented |
| Shared abstraction | Retrieval pipeline, generation pipeline, metric, ingestor, or plugin |
| Defaults | Hyperparameters used in this paper |
| Deviations | Differences from the original paper/code |
| Tests | Unit/integration/smoke tests supporting the implementation |
| Artifact | Config/result/cache files used in reported experiments |

### 3.4 Reproduction Modes

| Mode | Description | Reviewer Cost |
|------|-------------|---------------|
| **Figure-only** | Regenerate all paper figures/tables from released result artifacts | CPU only, minutes |
| **Small rerun** | Restore one small dataset dump and rerun a subset of pipelines/metrics | PostgreSQL + optional local/mock/API model, under 1 hour target |
| **Full rerun** | Recreate the complete benchmark | Full compute/API/local model resources, days/weeks |

The paper should promise fast figure reproduction, not fast full benchmark reruns.

---

## 4. Analysis Plan

### 4.1 Framework Coverage

Report:

- Number of supported ingestors.
- Number of retrieval pipelines.
- Number of generation pipelines.
- Number of metrics.
- Which components have tests and docs.
- Which components have paper/fidelity cards.

### 4.2 Reimplementation Fidelity

Report a table of representative methods:

| Method | Original Paper | AutoRAG-Research Component | Faithful Parts | Deviations |
|--------|----------------|----------------------------|----------------|------------|
| BM25 | Robertson & Zaragoza | retrieval pipeline | lexical sparse retrieval | DB/index implementation differs |
| HyDE | Gao et al. | retrieval pipeline | hypothetical document generation + dense retrieval | prompt/model may differ |
| IRCoT | Trivedi et al. | generation pipeline | iterative CoT + retrieval | prompt/model/dataset may differ |
| MAIN-RAG | MAIN-RAG paper | generation pipeline | multi-agent filtering | logprob/model support may differ |
| ET2RAG | ET2RAG paper | generation pipeline | context subset voting | model/prompt choices may differ |

### 4.3 Benchmark Results

Report:

1. Retrieval metric table by dataset/pipeline.
2. Generation metric table by dataset/retrieval/generation combination.
3. Runtime and token usage table.
4. Example outputs/failure modes.
5. Optional metric-family comparison as appendix.

The benchmark results validate the framework; they should not overtake the reproducibility thesis.

### 4.4 Reproducibility Results

Report:

1. Figure reproduction time.
2. Small rerun time.
3. Deterministic metric recomputation match.
4. Retrieval rerun exact-match or score-match rate.
5. Artifact completeness checklist.
6. Result export schema and file sizes.

### 4.5 Extensibility Results

If time permits:

- Add one small plugin/pipeline/metric case study.
- Report implementation effort and tests.
- Demonstrate that executor/evaluator/reporting work without additional harness code.

---

## 5. Paper Structure

### Abstract
A concise statement that AutoRAG-Research is a reproducible reimplementation framework for RAG research, validated through representative reimplemented pipeline experiments and released artifacts.

### 1. Introduction
- RAG reproducibility problem.
- Existing ad hoc experiment harnesses.
- Need for a reusable reimplementation framework.
- Summary of contributions.

### 2. Background and Related Work
- RAG pipelines and benchmarks.
- Reproducibility in ML/NLP.
- RAG evaluation frameworks and toolkits.
- Why this paper focuses on reimplementation infrastructure and artifacts.

### 3. AutoRAG-Research Framework
- Architecture.
- Data model.
- Pipeline abstractions.
- Metric/evaluator system.
- Config and plugin system.
- Reporting service.

### 4. Reimplementation and Reproducibility Protocol
- Dataset compatibility.
- Pipeline fidelity cards.
- Config manifests.
- Artifact release.
- Reproduction modes.

### 5. Experiments
- Retrieval track.
- Generation track.
- Optional multimodal track.
- Reproduction measurements.
- Extensibility case study.

### 6. Results
- Framework coverage.
- Reimplementation fidelity summary.
- Benchmark tables.
- Reproduction-time/artifact-completeness results.
- Runtime/cost analysis.

### 7. Discussion
- What reproducibility improves.
- What remains hard: closed models, missing hyperparameters, dataset licensing, model drift.
- How future researchers can extend the framework.

### 8. Conclusion
AutoRAG-Research makes RAG reimplementation studies more systematic, auditable, and reusable.

---

## 6. Success Criteria

The paper succeeds if it demonstrates:

1. **Framework coverage:** A meaningful set of retrieval/generation pipelines, datasets, and metrics are expressible in one framework.
2. **Fidelity transparency:** Each representative paper method has documented implementation choices and deviations.
3. **Reproducible artifacts:** Reviewers can regenerate figures/tables from released result artifacts quickly.
4. **Small rerun path:** Reviewers can restore a small dataset and rerun a subset without rebuilding the harness.
5. **Empirical usefulness:** The framework produces nontrivial RAG benchmark results with runtime/cost diagnostics.
6. **Extensibility:** At least one added method/metric/plugin can reuse the existing execution/evaluation/reporting stack.

---

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Tool-paper perception | High | Anchor paper in reproducible reimplementation and artifact evaluation, not UI/features |
| Exact reproduction challenged | High | Claim faithful documented reimplementation; publish deviation cards |
| Generation metrics used on datasets without answers | High | Split retrieval-only and generation tracks; enforce metric compatibility |
| Pipeline/config contamination | High | Unique pipeline naming and config manifests |
| Closed-model drift | Medium | Cache outputs, record versions, temperature 0, publish generations |
| Full benchmark too costly | Medium | Use deterministic subset caps, cached outputs, staged execution, and token/runtime manifests while still covering all implemented pipelines |
| Artifact package too large | Medium | Separate figure-only artifacts from full DB dumps |
| Insufficient time before 2026 deadline | High | Submit MVP if results already exist; otherwise target TMLR/JMLR/next-cycle venue with stronger artifacts |

---

*Draft corrected for reproducibility-framework positioning: 2026-04-24*
