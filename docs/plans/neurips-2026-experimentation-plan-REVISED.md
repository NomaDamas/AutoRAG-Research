# AutoRAG-Research NeurIPS 2026 Experimentation Plan — REVISED
## Evaluations & Datasets Track Submission

**Target:** NeurIPS 2026 Evaluations & Datasets Track  
**Deadlines:** Abstract May 4, 2026 | Full Paper May 6, 2026  
**Branch:** `experiment/paper-publication`  
**Paper Working Title:** *When Evaluation Changes the Winner: A Controlled Study of RAG Pipeline Rankings Under Unified Benchmarking*

---

## 0. CRITIQUE SYNTHESIS & RESPONSE

### What the Critics Said (Fatal Flaws)
1. **Timeline impossible:** 9-week experimental plan with May 6 deadline — most science would not exist at submission
2. **Reads as "framework/tool," not "evaluation science":** ED track requires evaluation itself as the object of study
3. **Novelty gap vs. RAGPerf (arXiv 2026):** "More pipelines, more datasets" is engineering, not science
4. **"Exact reproducibility" claim too strong:** Closed LLMs, prompt drift, version issues make this indefensible
5. **Experimental protocol is confounded:** No strict fairness controls across pipeline comparisons
6. **No statistical rigor:** No confidence intervals, no significance testing, no variance estimates
7. **Scope far too broad:** Text + image + multimodal + ablations + plugins = multiple papers

### What the Supporters Said (Strongest Elements)
1. **ED-track-native framing:** "Metric choice changes pipeline rankings" is exactly evaluation science
2. **Metric-sensitivity study (Phase 3):** The most ED-specific and scientifically valuable part
3. **Composable architecture:** Enables clean retrieval-generation interaction analysis
4. **Negative-result value:** "Simple baselines remain competitive" is a strong scientific finding
5. **Reproducibility-forward:** Pre-computed results, unified schema, fast reviewer reproduction

### Our Response: One Sharp Claim, Drastically Smaller Scope

**Core Thesis:** Under a unified, controlled evaluation protocol, the ranking of RAG pipelines is unstable across metrics, and simple baselines often match or exceed complex SOTA designs — challenging the validity of uncontrolled SOTA claims in the RAG literature.

This is **evaluation science**, not framework engineering. We are not selling AutoRAG-Research as a tool. We are using it as an **instrument** to study how evaluation design alters scientific conclusions in RAG.

---

## 1. SHARPENED SCIENTIFIC QUESTIONS

### Research Question 1 (Primary)
**How unstable are RAG pipeline rankings across different evaluation metrics under a controlled protocol?**
- Hypothesis: Rankings differ significantly (Kendall tau < 0.7) between traditional n-gram metrics, embedding-based metrics, and LLM-judge metrics.
- Falsifiable: If Kendall tau > 0.9 across all metric pairs, the hypothesis is rejected.

### Research Question 2 (Secondary)
**Do complex, paper-proposed RAG pipelines consistently outperform simple baselines when evaluated under identical conditions?**
- Hypothesis: On >30% of datasets, BM25 + BasicRAG matches or outperforms multi-hop/agentic pipelines.
- Falsifiable: If complex pipelines win on >90% of datasets by >5 points on all metrics, the hypothesis is rejected.

### Research Question 3 (Exploratory)
**How does retrieval quality correlate with generation quality across pipeline designs?**
- Hypothesis: Better retrieval does not always translate to better generation (diminishing returns beyond top-5).
- Falsifiable: If Pearson r > 0.9 between retrieval nDCG and generation BERTScore, the hypothesis is rejected.

---

## 2. STRICT EXPERIMENTAL PROTOCOL (Fairness Controls)

To ensure clean science, the following are **held constant** across ALL pipeline comparisons:

| Control | Value | Rationale |
|---------|-------|-----------|
| **Corpus** | Fixed per dataset (no re-chunking) | Chunking is a confounder; we fix it |
| **Embedding model** | `BAAI/bge-large-en-v1.5` for text | Single embedding family for all dense retrieval |
| **Generation model** | `gpt-4o-mini-2024-07-18` | Fixed, reproducible, cost-controlled |
| **Temperature** | 0.0 | Deterministic generation |
| **Max context length** | 4,000 tokens | Fixed context budget for all generation pipelines |
| **Top-k retrieval** | 10 | Fixed unless top-k is the ablation variable |
| **Prompt template** | Standard RAG prompt for all generation pipelines | Only pipeline-internal prompts vary |
| **Evaluation seed** | 42 | Reproducible metric computation |
| **Number of runs** | 3 per generation pipeline | API variance estimation |

**What varies (the experimental factors):**
1. **Retrieval strategy** (sparse, dense, hybrid, rewrite)
2. **Generation strategy** (single-pass, multi-hop, agentic)
3. **Evaluation metric** (n-gram, embedding, LLM-judge)

---

## 3. MINIMUM VIABLE EXPERIMENT MATRIX

### 3.1 Datasets (3 — Tightly Selected)

| Dataset | Type | #Queries | Why Included |
|---------|------|----------|--------------|
| **BEIR (scifact)** | Text (scientific) | ~300 | Factoid QA, short answers, clean retrieval GT |
| **RAGBench (techqa)** | Text (technical) | ~1,000 | Long-form answers, complex reasoning GT |
| **MrTyDi (english)** | Text (open-domain) | ~4,000 | Diverse queries, multilingual corpus (English subset) |

**Why only 3:** Reduces variance, ensures all datasets have generation ground truth, keeps compute manageable. Multimodal deferred to future work.

### 3.2 Retrieval Pipelines (4 — Representative Set)

| Pipeline | Family | Why Included |
|----------|--------|--------------|
| **BM25** | Sparse lexical | Strong baseline, no embeddings needed |
| **Vector Search (DPR)** | Dense single-vector | Standard neural retrieval |
| **Hybrid RRF** | Fusion | Combines sparse + dense |
| **HyDE** | Query expansion | Tests hypothetical doc embeddings |

**Why only 4:** Covers the key design axis (sparse vs. dense vs. fusion vs. expansion). Reranking and rewrite deferred — they add API costs and complexity.

### 3.3 Generation Pipelines (4 — Representative Set)

| Pipeline | Family | Why Included |
|----------|--------|--------------|
| **BasicRAG** | Single-pass | Baseline: retrieve → generate |
| **IRCoT** | Multi-hop | Interleaved retrieval + reasoning |
| **MAIN-RAG** | Agentic | Multi-agent filtering |
| **ET2RAG** | Ensemble | Majority voting on subsets |

**Why only 4:** Covers simple, multi-hop, agentic, and ensemble strategies. Self-RAG and RAG-Critic omitted — they require specialized model checkpoints or API costs that exceed budget.

### 3.4 Metrics (6 — Covering All Families)

**Retrieval (3):**
- Recall@10
- nDCG@10
- MRR

**Generation (6):**
- ROUGE-L (n-gram)
- BERTScore F1 (embedding)
- BARTScore F1 (embedding)
- UniEval Relevance (LLM-judge)
- UniEval Coherence (LLM-judge)
- Response Relevancy (diagnostic)

**Total experiment size:**
- 3 datasets × 4 retrieval = 12 retrieval runs
- 3 datasets × 4 retrieval × 4 generation × 3 runs = 144 generation runs
- All evaluated with 9 metrics
- **Estimated cost:** ~$300-500 (GPT-4o-mini), ~20 hours compute

---

## 4. ANALYSIS PLAN

### 4.1 Primary Analysis: Metric Ranking Stability

For each dataset and each pipeline family (retrieval-only and generation):
1. Compute pipeline rankings under each metric
2. Compute Kendall tau correlation between all metric pairs
3. Identify metric pairs with tau < 0.7 ("ranking instability")
4. Report how often the "best" pipeline changes when the metric changes

**Key Figure:** Kendall tau heatmap (metrics × metrics) with annotations for unstable pairs.

**Key Table:** Pipeline rankings under each metric for one representative dataset, showing rank inversions.

### 4.2 Secondary Analysis: Simple vs. Complex

For each dataset:
1. Compare best simple pipeline (BM25 + BasicRAG) vs. best complex pipeline (HyDE + MAIN-RAG)
2. Report win/loss/tie counts across all metrics
3. Compute effect sizes (Cohen's d) for significant differences

**Key Figure:** Scatter plot — simple pipeline score (x) vs. complex pipeline score (y), with diagonal reference. Points above diagonal = complex wins.

**Key Table:** Win/loss/tie matrix (simple vs. complex) across datasets and metrics.

### 4.3 Exploratory Analysis: Retrieval-Generation Correlation

1. For each generation pipeline, correlate retrieval nDCG@10 with generation BERTScore F1
2. Fit linear regression, report R²
3. Identify "high retrieval, low generation" outlier cases

**Key Figure:** Scatter plot — retrieval nDCG (x) vs. generation BERTScore (y), colored by pipeline. Annotate outliers.

### 4.4 Statistical Rigor

- **Confidence intervals:** Bootstrap 95% CIs for all mean scores (1,000 resamples)
- **Significance testing:** Paired permutation tests for pipeline comparisons
- **Variance reporting:** Standard deviation across 3 generation runs
- **Effect sizes:** Cohen's d for all pairwise comparisons
- **Multiple comparison correction:** Bonferroni for metric-to-metric correlations

---

## 5. PAPER STRUCTURE (9 pages)

**Abstract (1 paragraph)**
- Problem: RAG papers claim SOTA using different metrics and setups; rankings may be artifacts of evaluation design
- Method: Controlled benchmarking of 4 retrieval + 4 generation pipelines across 3 datasets with 9 metrics
- Findings: (1) Metric choice changes "best" pipeline on X% of comparisons; (2) Simple baselines match complex SOTA on Y% of cases; (3) Retrieval quality does not always predict generation quality
- Contribution: Demonstrates that evaluation design itself determines RAG conclusions

**1. Introduction (1.5 pages)**
- RAG proliferation and contradictory SOTA claims
- Existing benchmarks evaluate systems; we evaluate evaluation
- Core question: Are RAG pipeline rankings stable, or are they artifacts of metric choice?
- Contributions: (1) Controlled comparison protocol, (2) Ranking instability quantification, (3) Simple baseline competitiveness, (4) Retrieval-generation decoupling

**2. Related Work (1 page)**
- RAG benchmarks: CRAG (dataset), RAGChecker (diagnostic metrics), HawkBench (resilience)
- RAG evaluation frameworks: RAGPerf (end-to-end benchmarking)
- **Gap:** No prior work systematically studies how evaluation design (metric choice, protocol fairness) alters RAG pipeline rankings under controlled conditions
- Positioning: We are not a new benchmark dataset; we are a study of benchmark behavior

**3. Methodology (1.5 pages)**
- 3.1 Datasets: selection criteria, statistics, preprocessing
- 3.2 Pipelines: selection rationale, implementation fidelity to papers
- 3.3 Metrics: families (n-gram, embedding, LLM-judge), why these 9
- 3.4 Fairness protocol: strict controls (Table), what varies, what is fixed
- 3.5 Analysis: ranking stability, simple vs. complex, retrieval-generation correlation

**4. Results (2.5 pages)**
- 4.1 Metric Ranking Instability (primary result)
  - Kendall tau heatmap
  - Rank inversion examples
  - "Best pipeline" changes per metric
- 4.2 Simple vs. Complex (secondary result)
  - Win/loss/tie matrix
  - Effect sizes
  - Example queries where simple wins
- 4.3 Retrieval-Generation Correlation (exploratory)
  - Scatter plots, R² values
  - Outlier analysis
- 4.4 Statistical Summary
  - Confidence intervals table
  - Significance markers on all comparison tables

**5. Discussion (0.5 page)**
- Implications: RAG SOTA claims require metric specification
- Limitations: 3 datasets, 4 pipelines per family, single generation backend
- Future work: Multimodal, more metrics, human evaluation anchor

**6. Conclusion (0.5 page)**
- Evaluation design determines RAG conclusions
- AutoRAG-Research as a reproducible instrument for future evaluation science

**References**

**Appendix**
- A. Full pipeline implementation details and deviations from papers
- B. Complete result tables with confidence intervals
- C. Prompt templates
- D. Computational costs and API usage
- E. Reproducibility checklist

---

## 6. ARTIFACT PREPARATION (Minimum Viable)

### 6.1 Code Repository (Anonymized)
- **GitHub:** `anonymous-rag-benchmark` (new org, no author names)
- **Contents:** Framework code + experiment scripts + result outputs
- **README:** Quick reproduction instructions (target: 30 min with pre-computed results)
- **License:** Apache 2.0

### 6.2 Pre-computed Results
- **HuggingFace Dataset:** `anonymous-rag-results` (no author info)
- **Contents:** All retrieval outputs, generation outputs, metric scores per query
- **Format:** Parquet files with query_id, pipeline_name, output, scores
- **Purpose:** Reviewers can reproduce figures without running pipelines

### 6.3 Documentation
- **Croissant metadata:** For the result dataset (core fields only; no RAI needed for results)
- **Reproduction script:** `python reproduce_figures.py` → generates all paper figures from pre-computed results
- **Full reproduction:** `python run_experiments.py` → re-runs full matrix (for validation)

### 6.4 Double-Blind Compliance
- **No branded names:** Remove "AutoRAG" from paper; use "unified RAG benchmarking framework"
- **No author names:** Strip from repo, docs, commit history (use `git filter-repo`)
- **No public links to existing project:** No references to AutoRAG GitHub, docs, or prior work
- **Anonymous HF datasets:** Use anonymous accounts
- **Camera-ready only:** Reveal identity and branding after acceptance

---

## 8. RISK MITIGATION (Revised)

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| API costs exceed budget | Medium | High | Use GPT-4o-mini only; cache all API calls; if costs spike, reduce to 1 run per pipeline |
| Pipeline implementation bugs | Medium | High | Validate 1 pipeline end-to-end before batch runs; keep 2-day buffer for debugging |
| Results are noisy / no patterns | Medium | High | **This is still a valid result.** Frame as "even under controlled conditions, RAG evaluation is noisy" — still advances evaluation science |
| RAGPerf reviewers find us similar | High | Medium | Emphasize controlled protocol + ranking stability analysis; RAGPerf does not study metric-induced rank changes |
| Double-blind deanonymization | Medium | High | Strip all branding; use anonymous accounts; no public demos; camera-ready only reveal |
| Dataset license issues | Low | High | Verify BEIR, RAGBench, MrTyDi licenses; all are permissive academic use |
| Compute cluster unavailable | Low | High | Local fallback: MacBook Pro + overnight runs; API calls are the bottleneck, not local compute |

---

## 10. WHAT WAS CUT (And Why)

| Cut Element | Reason |
|-------------|--------|
| **Multimodal (ViDoRe, VisRAG, Open-RAGBench)** | Adds complexity, different modeling assumptions, separate compute stack. Defer to future work. |
| **Plugin validation** | Product feature, not evaluation science. Weak paper contribution. |
| **Gradio demo / HF Spaces** | Branded, deanonymization risk, not needed for review. |
| **Cross-backend ablations (Claude, Gemini, local models)** | Expensive, adds variance, not central to core thesis. |
| **Top-k ablations (5, 20, 50)** | Interesting but secondary; focus on metric stability first. |
| **Self-RAG, RAG-Critic, AutoThinkRAG** | Require specialized checkpoints or expensive API calls. Cut to control costs. |
| **20+ pipelines → 8 pipelines** | Breadth inflation. 8 well-controlled pipelines > 20 loosely controlled ones. |
| **9 datasets → 3 datasets** | Focus on depth and statistical power over breadth. |
| **"Exact reproducibility" claim** | Too strong. Changed to "faithful implementation with documented deviations." |

---

*Revised: 2026-04-22*  
*Target: NeurIPS 2026 Evaluations & Datasets Track*  
*Scope: Minimum Viable Submission — One Sharp Claim, Tight Controls, Statistical Rigor*
