# AutoRAG-Research NeurIPS 2026 Experimentation Plan
## Evaluations & Datasets Track Submission

**Target:** NeurIPS 2026 Evaluations & Datasets Track  
**Deadlines:** Abstract May 4, 2026 | Full Paper May 6, 2026  
**Branch:** `experiment/paper-publication`  
**Paper Working Title:** *AutoRAG-Research: A Unified, Reproducible Benchmarking Framework for Retrieval-Augmented Generation*

---

## 1. Paper Positioning & Core Contribution

### 1.1 Problem Statement
The RAG research landscape is fragmented:
- Every paper claims SOTA using different datasets, metrics, and experimental setups
- No unified way to fairly compare pipeline designs across diverse retrieval and generation strategies
- Existing benchmarks (CRAG, RAGChecker) evaluate *systems* but lack systematic *pipeline design comparison* infrastructure
- Reproducing and extending prior RAG work requires re-implementing evaluation harnesses from scratch

### 1.2 Core Contribution
AutoRAG-Research is **not** a new RAG pipeline. It is an **evaluation science contribution**:
- A unified benchmarking framework that implements 20+ SOTA RAG pipelines from papers in one reproducible codebase
- Pre-computed embeddings and unified data formats across 9+ diverse datasets
- A plugin architecture enabling extensible, fair comparison of retrieval and generation strategies
- Systematic analysis revealing how pipeline design choices affect performance across dataset types

### 1.3 Scientific Claims
1. **Reproducibility Claim:** AutoRAG-Research enables exact reproduction of 20+ published RAG pipeline results with unified data formats and metrics
2. **Comparative Analysis Claim:** Systematic comparison across pipelines reveals dataset-dependent optimal design patterns (e.g., sparse vs. dense retrieval, single-hop vs. multi-hop generation)
3. **Evaluation Design Claim:** The choice of evaluation metrics significantly affects pipeline rankings, and unified metric suites are necessary for fair comparison
4. **Extensibility Claim:** The plugin system lowers the barrier for researchers to contribute new pipelines/metrics and receive automatic benchmarking

---

## 2. Experimental Design

### 2.1 Datasets (Unified via AutoRAG)

| Dataset | Type | Size | Domains | Pre-computed Embeddings |
|---------|------|------|---------|------------------------|
| BEIR (scifact) | Text | ~5K queries | Scientific | Yes |
| RAGBench (subset) | Text | ~10K queries | Multi-domain | Yes |
| MrTyDi (en) | Text | ~6K queries | Multilingual | Yes |
| BRIGHT | Text | ~2K queries | Reasoning-intensive | Yes |
| CRAG (dev) | Text | ~4K queries | Web/KG simulated | No (public) |
| ViDoRe (pdf) | Image | ~3K queries | Visual documents | Yes |
| VisRAG (ChartQA) | Image | ~2K queries | Charts/figures | Yes |
| Open-RAGBench | Multi-modal | ~5K queries | arXiv PDFs | Yes |

**Justification:** Diverse coverage of text, image, and multimodal RAG scenarios. Mix of public benchmarks and AutoRAG-preprocessed versions.

### 2.2 Pipelines (20+ Implemented)

**Retrieval Pipelines (8):**
1. Vector Search (DPR) - dense single-vector
2. Vector Search (ColBERT) - dense multi-vector MaxSim
3. BM25 - sparse lexical
4. HyDE - hypothetical document embeddings
5. Query Rewrite - query reformulation before retrieval
6. RETRO* - rubric-based LLM reranking
7. Hybrid RRF - reciprocal rank fusion
8. Hybrid CC - convex combination

**Generation Pipelines (8):**
1. BasicRAG - retrieve-then-generate baseline
2. IRCoT - interleaved retrieval + chain-of-thought
3. ET2RAG - majority voting on context subsets
4. MAIN-RAG - multi-agent filtering
5. VisRAG - vision-language generation
6. Self-RAG - self-reflective generation
7. RAG-Critic - critic-guided correction
8. AutoThinkRAG - adaptive reasoning paths

**Justification:** Coverage of major RAG paradigms from 2020-2025, spanning simple to complex strategies.

### 2.3 Metrics

**Retrieval Metrics:**
- Recall@k, Precision@k, F1@k
- nDCG@k, MRR, MAP

**Generation Metrics:**
- N-gram: BLEU, ROUGE, METEOR
- Semantic: BERTScore, BARTScore, SemScore
- LLM-as-Judge: UniEval (coherence, consistency, fluency, relevance)
- Diagnostic: Response Relevancy, Token F1, Exact Match

**Justification:** Both traditional and modern metrics to study metric-dependent ranking effects.

### 2.4 Experimental Protocol

**Phase 1: Reproducibility Verification (Weeks 1-2)**
- Run all retrieval pipelines on all text datasets
- Run all generation pipelines on datasets with generation ground truth
- Verify that results align with published paper claims where available
- Document deviations and implementation choices

**Phase 2: Systematic Comparison (Weeks 3-4)**
- Cross-pipeline comparison tables per dataset
- Analysis of: retrieval type (sparse vs. dense vs. hybrid) vs. dataset characteristics
- Analysis of: generation complexity (single-pass vs. multi-hop vs. agent-based) vs. query complexity
- Correlation analysis between retrieval and generation performance

**Phase 3: Evaluation Design Analysis (Weeks 5-6)**
- Run all metrics on all generation outputs
- Compute metric-to-metric correlations (Kendall tau between pipeline rankings)
- Identify cases where metric choice changes "best" pipeline
- Compare LLM-judge metrics vs. traditional metrics for ranking stability

**Phase 4: Ablation & Failure Mode Analysis (Weeks 7-8)**
- Ablate top-k values (5, 10, 20, 50) across pipelines
- Ablate LLM backends (GPT-4o-mini, Claude 3.5 Haiku, local models)
- Identify failure modes: when do complex pipelines underperform simple baselines?
- Cross-dataset transfer: do top pipelines generalize?

**Phase 5: Plugin Ecosystem Validation (Week 9)**
- Implement 2 new pipelines as plugins (to demonstrate extensibility)
- Run them through the same evaluation harness
- Measure time-to-benchmark for new contributions

---

## 3. Analysis & Figures

### 3.1 Primary Figures

**Figure 1: Framework Overview**
- Architecture diagram showing: unified data ingestion → pipeline execution → metric evaluation → leaderboard/reporting
- Highlight plugin system and cross-database analytics

**Figure 2: Pipeline Comparison Heatmap**
- Rows: datasets | Columns: pipelines
- Color: nDCG@10 (retrieval) or BERTScore F1 (generation)
- Clustered to show dataset-type preferences

**Figure 3: Metric Ranking Correlation**
- Kendall tau correlation matrix between all generation metrics
- Highlight cases where metric choice inverts rankings

**Figure 4: Design Choice Analysis**
- Scatter: retrieval complexity (x) vs. generation complexity (y)
- Points colored by overall performance
- Annotate surprising findings (e.g., simple BM25+BasicRAG competitive on factoid QA)

**Figure 5: Failure Mode Examples**
- Specific query examples where SOTA pipelines fail
- Side-by-side outputs from different pipeline types
- Error taxonomy: hallucination, retrieval miss, reasoning error, etc.

### 3.2 Primary Tables

**Table 1: Dataset Statistics**
- Dataset | Type | #Queries | #Docs | Avg Doc Length | Domain | Ground Truth Types

**Table 2: Retrieval Pipeline Results (nDCG@10)**
- Full comparison matrix across datasets
- Bold best per dataset; underline second best

**Table 3: Generation Pipeline Results (BERTScore F1)**
- Full comparison matrix across datasets
- Include retrieval metric as reference

**Table 4: Metric Ranking Stability**
- Pipeline rankings under different metric sets
- Report how often "best" pipeline changes

**Table 5: Ablation Results**
- Top-k sensitivity for top-3 pipelines
- LLM backend sensitivity

---

## 4. Paper Structure (9 pages)

**Abstract (1 paragraph)**
- Problem: fragmented RAG evaluation
- Solution: unified benchmarking framework with 20+ pipelines, 9 datasets, comprehensive metrics
- Key findings: dataset-dependent optimal designs, metric-choice sensitivity, simple baselines often competitive

**1. Introduction (1.5 pages)**
- RAG proliferation and SOTA claims problem
- Existing benchmarks evaluate systems, not design choices
- AutoRAG-Research: unified, reproducible, extensible
- Main contributions (bulleted)

**2. Related Work (1 page)**
- RAG benchmarks: CRAG, RAGChecker, ClashEval, UDA, HawkBench
- Evaluation frameworks: RAGPerf, UniBench, HEMM
- AutoML for RAG: AutoRAG (the original tool, not this research framework)
- Gap: no unified pipeline benchmarking with systematic design analysis

**3. Framework Design (1.5 pages)**
- Architecture: layered design (repository → UoW → service → pipeline)
- Unified data format and pre-computed embeddings
- Pipeline abstraction: retrieval vs. generation base classes
- Plugin system and extensibility
- Metric system: retrieval + generation with granularity support

**4. Experimental Setup (1 page)**
- Datasets and preprocessing
- Pipeline implementations and faithfulness to papers
- Metrics and evaluation protocol
- Computational environment and reproducibility measures

**5. Results & Analysis (2.5 pages)**
- 5.1 Reproducibility verification
- 5.2 Systematic pipeline comparison (heatmaps, rankings)
- 5.3 Evaluation design analysis (metric correlations, ranking stability)
- 5.4 Ablation studies and failure modes
- 5.5 Plugin ecosystem demonstration

**6. Discussion & Limitations (0.5 page)**
- Coverage limitations (languages, modalities)
- Metric limitations (LLM-judge bias, reference-based metrics)
- Computational cost of comprehensive benchmarking

**7. Conclusion (0.5 page)**
- AutoRAG-Research advances RAG evaluation science
- Future work: real-time leaderboard, community submissions, new modalities

**References**

**Appendix**
- A. Implementation details per pipeline
- B. Full result tables
- C. Metric definitions and prompts
- D. Plugin API documentation
- E. Computational costs

---

## 5. Artifact Preparation

### 5.1 Code Repository
- **GitHub:** anonymized repo under `autorag-research` org (or `anonymous-autorag`)
- **Documentation:** README with quickstart, architecture docs, API reference
- **Tests:** `make test` passes (Docker PostgreSQL required)
- **License:** Apache 2.0

### 5.2 Datasets
- **HuggingFace Datasets:** host pre-processed, unified versions
- **Croissant metadata:** include core + RAI fields for each dataset
- **Sample data:** include small samples (<4GB total) for reviewer inspection

### 5.3 Pre-computed Results
- **HuggingFace or Zenodo:** all pipeline outputs and evaluation scores
- Include: raw retrieval results, generation outputs, metric scores per query
- Enable reproducibility without requiring API keys or GPU access

### 5.4 Demo
- **Gradio UI:** hosted HuggingFace Space with leaderboard
- **CLI demo:** `autorag-research run --db-name=beir_scifact_test` reproduction

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| API costs for 20+ pipelines x 9 datasets | High | Use smaller model variants (GPT-4o-mini); pre-compute and cache all results |
| PostgreSQL setup complexity for reviewers | Medium | Provide Docker Compose one-liner; include SQLite fallback mode |
| Results don't show clear patterns | High | Frame as "negative result" — showing fragmentation IS the finding |
| Similar to RAGPerf (arXiv 2026) | Medium | Emphasize: more pipelines, more datasets, plugin system, pre-computed embeddings |
| Computational time exceeds deadline | High | Parallelize across machines; prioritize key experiments |
| Reviewers question novelty of "just a tool" | High | Frame around evaluation science; emphasize diagnostic insights and systematic analysis |

---

## 8. Success Criteria

The paper succeeds if it demonstrates:
1. **Exact reproducibility** of 15+ published pipeline results
2. **Novel insights** from systematic comparison (≥3 non-obvious findings)
3. **Metric sensitivity analysis** showing ranking instability across metrics
4. **Plugin validation** with 2+ community-contributed pipelines benchmarked
5. **Accessibility:** reviewers can reproduce key results in <30 minutes

---

*Drafted: 2026-04-22*
*Target: NeurIPS 2026 Evaluations & Datasets Track*
