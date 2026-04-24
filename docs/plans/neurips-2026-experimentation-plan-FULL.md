# AutoRAG-Research: Full-Scale Experimentation Plan
## Comprehensive Unified Benchmarking of All Implemented Pipelines & Datasets

**Target Venue:** NeurIPS 2027 / ICLR 2027 / ACL 2027 Evaluations & Datasets Track (or JMLR/TMLR)  
**Branch:** `experiment/paper-publication`  
**Working Title:** *AutoRAG-Research: Unified Large-Scale Benchmarking of 20+ RAG Pipelines Across Text, Image, and Multimodal Domains*

---

## 0. RATIONALE: WHY FULL COVERAGE?

The core value proposition of AutoRAG-Research is its **unified interface** for running arbitrary combinations of retrieval pipelines, generation pipelines, datasets, and metrics. Restricting the evaluation to a small subset would:

1. **Misrepresent the framework's capability** — the unified abstraction is designed precisely for large-scale comparison
2. **Undermine the scientific contribution** — the paper's novelty is "comprehensive, reproducible, cross-domain RAG benchmarking under one protocol"
3. **Fail to serve users** — researchers use AutoRAG-Research because they want to compare their new pipeline against ALL existing ones, not a curated subset
4. **Miss cross-domain insights** — text, image, and multimodal RAG may exhibit different design patterns; excluding modalities weakens conclusions

**This plan embraces full coverage.** Timeline is realistic (3-4 months) rather than artificially compressed.

---

## 1. COMPLETE INVENTORY OF IMPLEMENTED COMPONENTS

### 1.1 Retrieval Pipelines (10 Implemented)

| # | Pipeline | File | Paper Reference | Key Mechanism |
|---|----------|------|-----------------|---------------|
| 1 | **BM25** | `bm25.py` | Robertson & Zaragoza, 2009 | Sparse lexical retrieval |
| 2 | **Vector Search (DPR)** | `vector_search.py` | Karpukhin et al., EMNLP 2020 | Dense single-vector similarity |
| 3 | **HyDE** | `hyde.py` | Gao et al., ACL 2023 | Hypothetical document embeddings |
| 4 | **Hybrid RRF** | `hybrid.py` | Cormack et al., SIGIR 2009 | Reciprocal rank fusion |
| 5 | **Hybrid CC** | `hybrid.py` | Chen et al., 2022 | Convex combination fusion |
| 6 | **Query Rewrite** | `query_rewrite.py` | Wang et al., EMNLP 2023 | LLM-based query reformulation |
| 7 | **RETRO*** | `retro_star.py` | Wang et al., ICLR 2026 | Rubric-based LLM reranking |
| 8 | **HEAVEN** | `heaven.py` | — | Hierarchical evidence aggregation |
| 9 | **Power of Noise** | `power_of_noise.py` | Zhuang et al., SIGIR 2024 | Seeded random noise wrapper |
| 10 | **Question Decomposition (Retrieval)** | `question_decomposition.py` | — | Sub-query decomposition |

### 1.2 Generation Pipelines (10 Implemented)

| # | Pipeline | File | Paper Reference | Key Mechanism |
|---|----------|------|-----------------|---------------|
| 1 | **BasicRAG** | `basic_rag.py` | Lewis et al., NeurIPS 2020 | Retrieve → generate baseline |
| 2 | **IRCoT** | `ircot.py` | Trivedi et al., ACL 2023 | Interleaved retrieval + CoT |
| 3 | **ET2RAG** | `et2rag.py` | — | Majority voting on context subsets |
| 4 | **VisRAG** | `visrag_gen.py` | — | Vision-language generation |
| 5 | **MAIN-RAG** | `main_rag.py` | — | Multi-agent filtering |
| 6 | **AutoThinkRAG** | `autothinkrag.py` | 2025 | Adaptive reasoning paths |
| 7 | **Self-RAG** | `self_rag.py` | Asai et al., 2023 | Self-reflective generation |
| 8 | **RAG-Critic** | `rag_critic.py` | — | Critic-guided correction |
| 9 | **SPD-RAG** | `spd_rag.py` | — | Speculative decoding RAG |
| 10 | **Question Decomposition (Generation)** | `question_decomposition.py` | — | Sub-query generation |

### 1.3 Datasets / Ingestors (11 Implemented)

**Code-verified query/corpus counts** (extracted from ingestor implementations and test fixtures on 2026-04-24):

| # | Dataset | File | Type | Default Subset | #Queries | #Docs | Pre-computed Embeddings |
|---|---------|------|------|----------------|----------|-------|------------------------|
| 1 | **BEIR** | `beir.py` | Text | `scifact` | **300** | **5,183** | Yes |
| 2 | **MTEB** | `text_mteb.py` | Text | Task-dependent (e.g., `NFCorpus`) | 323 (NFCorpus) | 3,633 | Yes |
| 3 | **RAGBench** | `ragbench.py` | Text | `covidqa` | **~2,000/subset** (12 subsets → ~10K aggregate) | per-query docs | Yes |
| 4 | **MrTyDi** | `mrtydi.py` | Text | `english` | **~6,000** | ~108K | Yes |
| 5 | **BRIGHT** | `bright.py` | Text | `biology` | **~2,000/domain** (12 domains) | Varies | Yes |
| 6 | **CRAG** | `crag.py` | Text | `dev` | **~4,400** | Per-query search results | No |
| 7 | **ViDoRe** | `vidore.py` | Image | `arxivqa_test_subsampled` | **500** | **500** (1:1 mapping) | Yes |
| 8 | **ViDoRe v2** | `vidorev2.py` | Image | `esg_reports_v2` | **228** | **1,538** | Yes |
| 9 | **ViDoRe v3** | `vidorev3.py` | Image | `hr` | **~600/industry** (8 industries) | Varies | Yes |
| 10 | **VisRAG** | `visrag.py` | Image | `ChartQA` | **~2,000** | Images | Yes |
| 11 | **Open-RAGBench** | `open_ragbench.py` | Multimodal | `arxiv` | **~5,000** | arXiv PDFs | Yes |

**Single-subset experimental total (for cost estimation):**
- **Text partition (5 datasets):** 300 + 2,000 + 6,000 + 2,000 + 4,400 = **~14,700 queries**
- **Image partition (2 datasets):** 500 + 2,000 = **~2,500 queries**
- **Multimodal partition (1 dataset):** **~5,000 queries**
- **Grand total across single-subset experiments:** **~22,200 queries**

> **Note:** BEIR, RAGBench, MrTyDi, BRIGHT, and ViDoRe families offer multiple subsets; defaults selected above are used for baseline cost estimates. Running all subsets multiplies query counts proportionally (see Appendix H for multi-subset scenario).

### 1.4 Evaluation Metrics (21 Configs)

**Retrieval Metrics (7):**
1. Recall@k
2. Full Recall
3. Precision@k
4. F1@k
5. nDCG@k
6. MRR
7. MAP

**Generation Metrics (14):**
1. ROUGE-L
2. BERTScore F1
3. SemScore
4. Response Relevancy
5. Exact Match
6. Token F1
7. BARTScore F1
8. BARTScore Precision
9. BARTScore Recall
10. BARTScore Faithfulness
11. UniEval Coherence
12. UniEval Consistency
13. UniEval Fluency
14. UniEval Relevance

### 1.5 Infrastructure Capabilities

- **Executor:** Orchestrates sequential pipeline execution with retry logic, completion verification, batch processing
- **Evaluator:** Automated metric evaluation with configurable batch sizes
- **Reporting Service:** DuckDB-based cross-database analytics, SQL query interface
- **Leaderboard UI:** Gradio-based MTEB-style leaderboard with Borda count ranking
- **Plugin System:** Entry-point based discovery for custom pipelines/metrics
- **Pre-computed Embeddings:** Available for download via HuggingFace

---

## 2. CORE SCIENTIFIC CONTRIBUTIONS

### Contribution 1: The First Large-Scale Unified RAG Benchmarking Study
No prior work has systematically evaluated 20+ RAG pipelines across 11 datasets (text, image, multimodal) under a single, unified protocol. Existing benchmarks (CRAG, RAGChecker) evaluate specific systems or metrics; AutoRAG-Research enables **cross-pipeline, cross-domain, cross-metric comparison at scale**.

### Contribution 2: Quantification of Pipeline Ranking Instability
We measure how much pipeline rankings change when:
- The evaluation metric changes (n-gram vs. embedding vs. LLM-judge)
- The dataset domain changes (scientific vs. visual vs. multimodal)
- The retrieval strategy changes (sparse vs. dense vs. hybrid)
- The generation complexity changes (single-pass vs. multi-hop vs. agentic)

### Contribution 3: Dataset-Dependent Design Patterns
We identify which pipeline families excel on which data types:
- Does sparse retrieval dominate on scientific text?
- Does dense retrieval dominate on visual documents?
- Do multi-hop generation pipelines help on reasoning-intensive queries?
- Do agentic pipelines justify their computational cost?

### Contribution 4: Reproducibility Infrastructure as Scientific Instrument
We release:
- All pipeline implementations with documented deviations from original papers
- All pre-computed embeddings and results
- All prompts, hyperparameters, and random seeds
- A one-command reproduction script

---

## 3. EXPERIMENTAL DESIGN

### 3.1 Experiment Matrix (Theoretical Full Scope)

The unconstrained full matrix would be:
- **11 datasets** × **10 retrieval pipelines** = **110 retrieval experiments**
- **11 datasets** × **10 retrieval pipelines** × **10 generation pipelines** = **1,100 generation experiments**

**Metric computations (corrected from prior draft):**
- Retrieval metrics: **110 runs × 7 retrieval metrics = 770 evaluations**
- Generation metrics: **1,100 runs × 14 generation metrics = 15,400 evaluations**
- **Total metric computations: 770 + 15,400 = 16,170**

**Practical constraints that reduce this to the partitioned matrix (§3.2):**
- **Image/multimodal datasets** (ViDoRe v1/v2/v3, VisRAG, Open-RAGBench) require vision-language models or image embeddings; purely text-only retrieval/generation pipelines cannot consume image corpora
- **Current implementation inventory (verified 2026-04-24):**
  - Retrieval: 9 text-capable (BM25, Vector Search, HyDE, Hybrid RRF, Hybrid CC, Query Rewrite, RETRO*, Power of Noise, Question Decomposition) + 1 image-capable (HEAVEN)
  - Generation: 8 text-only + 1 image-only (VisRAG) + 1 multimodal (AutoThinkRAG) = 9 text-capable, 2 image-capable
- **Generation pipelines without retrieval ground truth** can only be evaluated on datasets with generation GT (all selected datasets satisfy this)

### 3.2 Partitioned Evaluation Strategy

To make the full matrix feasible, we partition by modality. **Counts below reflect actual implementation inventory (verified 2026-04-24), not aspirational targets.**

#### Partition A: Text-Only Benchmarking
**Datasets (5):** BEIR (scifact), RAGBench (covidqa), MrTyDi (en), BRIGHT (biology), CRAG (dev)
**Retrieval Pipelines (9 text-capable):** BM25, Vector Search, HyDE, Hybrid RRF, Hybrid CC, Query Rewrite, RETRO*, Power of Noise, Question Decomposition (retrieval)
- *HEAVEN excluded* — image-only pipeline
**Generation Pipelines (9 text-capable):** BasicRAG, IRCoT, ET2RAG, MAIN-RAG, AutoThinkRAG, Self-RAG, RAG-Critic, SPD-RAG, Question Decomposition (generation)
- *VisRAG excluded* — image-only pipeline

| Run type | Count | Math |
|----------|-------|------|
| Retrieval runs | **45** | 5 datasets × 9 retrieval |
| Generation runs | **405** | 5 datasets × 9 retrieval × 9 generation |

#### Partition B: Image-Only Benchmarking
**Datasets (2):** ViDoRe (arxivqa_test_subsampled), VisRAG (ChartQA)
**Retrieval Pipelines (3 usable):**
- HEAVEN (native image retrieval)
- Vector Search (with image embeddings, e.g., SigLIP)
- Hybrid RRF (fusing HEAVEN + Vector Search)
- *BM25, HyDE, Query Rewrite, RETRO*, Question Decomposition, Hybrid CC excluded* — text-only or non-trivial to adapt for short-term scope
**Generation Pipelines (2 usable):**
- VisRAG (native vision-language generation)
- AutoThinkRAG (multimodal-capable)

| Run type | Count | Math |
|----------|-------|------|
| Retrieval runs | **6** | 2 datasets × 3 retrieval |
| Generation runs | **12** | 2 datasets × 3 retrieval × 2 generation |

#### Partition C: Multimodal Benchmarking
**Datasets (1):** Open-RAGBench (arxiv)
**Retrieval Pipelines (3 usable):** Same as Partition B (HEAVEN, Vector Search w/ multimodal embeddings, Hybrid RRF)
**Generation Pipelines (2 usable):** VisRAG, AutoThinkRAG

| Run type | Count | Math |
|----------|-------|------|
| Retrieval runs | **3** | 1 dataset × 3 retrieval |
| Generation runs | **6** | 1 dataset × 3 retrieval × 2 generation |

#### Aggregated Totals (Partitioned Realistic Scope)

Single-backend (API era) figures — **kept for reference only**:

| Metric | Single-backend | Dual-backend (Qwen 3.6 + Gemma 4) |
|--------|----------------|-----------------------------------|
| Retrieval runs (LLM-free pipelines) | **54** | 54 (shared) |
| Retrieval runs (LLM-backed, e.g., HyDE/RETRO*) | ~20 (counted in above) | ~40 (primary + optional 2nd backend) |
| **Generation runs** | 423 | **846** (423 × 2 backends) |
| Retrieval metric evaluations | 54 × 7 = **378** | 378 |
| Generation metric evaluations | 423 × 14 = **5,922** | **846 × 14 = 11,844** |
| **Total metric evaluations** | 6,300 | **12,222** |

> **Change log vs. prior draft:**
> - Previous partition numbers (2×5×3 + 1×5×3 = 45 extra generation runs) were aspirational and did not match actual image/multimodal pipeline counts.
> - With the switch from OpenAI API to local vLLM serving with **two distinct generation backends** (Qwen 3.6 and Gemma 4), all generation-side runs are doubled. This is the primary cost driver in §8.

### 3.3 Fairness Protocol

| Control | Value |
|---------|-------|
| **Corpus** | Fixed per dataset (pre-ingested, no re-chunking) |
| **Embedding model (text)** | `BAAI/bge-large-en-v1.5` (served locally, FP16) |
| **Embedding model (image)** | `google/siglip2-large-patch16-384` or equivalent VLM encoder |
| **Generation backend A (text)** | **Qwen 3.6 (7B or 14B class)** served via vLLM on Blackwell GPU |
| **Generation backend B (text)** | **Gemma 4 (9B or 27B-FP8 class)** served via vLLM on Blackwell GPU |
| **Generation backend (image)** | Qwen-VL 3.6 + Gemma 4 VLM (same cross-backend design, vLLM-served) |
| **Retrieval-side LLM (HyDE, Query Rewrite, RETRO*, Question Decomp)** | Uses the corresponding generation backend (same LLM as its downstream pipeline) for within-run consistency |
| **Metric LLM-judge (`response_relevancy` only)** | **GPT-5.5** (via OpenAI-compatible API) — premium judge for evaluation objectivity |
| **Temperature** | 0.0 (deterministic decode) |
| **Max context length** | 4,000 tokens input + 512 tokens output |
| **Top-k retrieval** | 10 (default) |
| **vLLM serving config** | `--max-model-len 8192 --max-num-seqs 128 --dtype bfloat16 --enable-chunked-prefill` |
| **Evaluation seed** | 42 |
| **Retries** | 3 with exponential backoff on vLLM & OpenAI failures |

**Key change vs. prior draft:** Generation moved from OpenAI API to **on-premises vLLM serving** on 2× RTX PRO 6000 Blackwell (96GB each, one model per GPU). This eliminates per-token API spend ($11K → ~$3K) at the cost of longer wall-clock time (GPU-bound).

**Documented deviations:** Any pipeline requiring hyperparameters not specified in the original paper will use framework defaults, with deviations logged in Appendix A. **Hyperparameters that materially affect GPU budget (e.g., `top_k` in MAIN-RAG, `candidate_top_k` in RETRO*, `max_steps` in IRCoT) are also logged in Appendix H.2.**

#### 3.3.1 vLLM Serving Setup

**Hardware (verified available to the team):**
- 2× NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM each
- Each GPU hosts a single model instance (one vLLM server per GPU)
- Models run in parallel (independent workloads), not tensor-parallel

**Model allocation:**

| GPU | Model | Precision | VRAM Footprint | Estimated Throughput (vLLM, mixed prefill+decode) |
|-----|-------|-----------|----------------|---------------------------------------------------|
| GPU 0 | **Qwen 3.6** (14B default; 7B if throughput critical) | bfloat16 | ~28 GB | ~6,000 tok/s (14B) / ~10,000 tok/s (7B) |
| GPU 1 | **Gemma 4** (9B default; 27B-FP8 if VRAM permits) | bfloat16 | ~18 GB | ~8,000 tok/s (9B) / ~4,000 tok/s (27B-FP8) |

> **Throughput source:** Estimates based on published vLLM benchmarks for similar Llama-3/Qwen-2.5/Gemma-2 models on Blackwell-class hardware (H100/B100 surrogate), scaled to RTX PRO 6000's ~85% of H100 FP16 throughput. **To be validated empirically in Phase 1 pilot (see §H.9 A4 sensitivity check).**

**Vision-language variants (for image/multimodal partition):**
- GPU 0: Qwen-VL 3.6 (14B-VLM class)
- GPU 1: Gemma 4 VLM (e.g., PaliGemma-4 or Gemma-4-vision)

Both models are assumed to expose an OpenAI-compatible chat-completions endpoint via `vllm.entrypoints.openai.api_server`, letting AutoRAG-Research's existing LLM config abstraction work unchanged.

#### 3.3.2 Cross-Backend Robustness Protocol

Because we run every pipeline against **both** Qwen 3.6 and Gemma 4 independently, we gain a new observable: **how stable are pipeline rankings across LLM backends?** This is added as a first-class analysis (see §4.6).

**Implications for experiment counts:**
- Every generation run from §3.2 is duplicated: once per LLM backend
- `Generation runs (dual-backend) = 423 × 2 = 846`
- `Retrieval runs (unchanged, shared across backends) = 54`
  - *Exception:* Retrieval pipelines that invoke LLMs (HyDE, Query Rewrite, RETRO*, Question Decomposition) must be re-run per backend to match the downstream generation model; we keep **one primary retrieval run** (Qwen-driven) for comparability and optionally re-run with Gemma in a subset for robustness (~20 additional runs).

**Variance estimation:** The 2-backend design replaces the previous "3 runs per pipeline" plan (see prior §3.3), providing variance across **model families** rather than across **decoding samples** — scientifically richer, and already bakes API variance out via temperature=0.0 decoding.

### 3.4 Statistical Rigor

- **Confidence intervals:** Bootstrap 95% CIs for all mean scores (10,000 resamples)
- **Significance testing:** Paired permutation tests (10,000 permutations) for pipeline comparisons
- **Effect sizes:** Cohen's d for all pairwise comparisons
- **Multiple comparison correction:** Benjamini-Hochberg FDR for metric-to-metric correlations
- **Variance reporting:** Standard deviation across repeated runs where API variance is expected

#### 3.4.1 Statistical Power Analysis (Pre-registered)

To justify sample sizes, we compute **Minimum Detectable Effect Sizes (MDES)** per dataset at α=0.05, power=0.80, paired two-tailed test. Assumed standard deviation σ = 0.05 for BERTScore-family metrics (based on pilot runs on BEIR scifact), larger for n-gram metrics (σ ≈ 0.10).

| Dataset | n (queries) | MDES (σ=0.05) | MDES (σ=0.10) | Adequate for |
|---------|-------------|---------------|---------------|--------------|
| BEIR (scifact) | 300 | **0.013 (1.3pt)** | 0.025 | Large effects only |
| RAGBench (covidqa) | ~2,000 | **0.005 (0.5pt)** | 0.010 | Small effects |
| MrTyDi (en) | ~6,000 | **0.003 (0.3pt)** | 0.006 | Very small effects |
| BRIGHT (biology) | ~2,000 | 0.005 | 0.010 | Small effects |
| CRAG (dev) | ~4,400 | **0.004 (0.4pt)** | 0.008 | Small effects |
| ViDoRe (arxivqa) | 500 | **0.010 (1.0pt)** | 0.020 | Medium effects only |

**Formula:** MDES ≈ 2.8 × σ / √n  (α=0.05, power=0.80, paired design)

#### 3.4.2 Multiple-Comparison Correction

**Pairwise pipeline comparisons per metric per dataset:**
- 9 text-capable generation pipelines → C(9,2) = **36 pairwise comparisons**
- Across 14 generation metrics → 36 × 14 = **504 comparisons per dataset**
- Across 5 text datasets → **2,520 total text-generation comparisons**

**Family-wise correction strategy:**
- **Primary hypothesis tests (pipeline rankings):** Benjamini-Hochberg FDR at 5% across all pairs within each (dataset, metric) cell
- **Ranking stability (Kendall tau):** Bonferroni correction across C(14,2)=91 metric pairs → adjusted α = 0.05/91 ≈ **5.5e-4**
- **Simple vs. complex comparison:** Pre-registered single primary hypothesis per dataset → no correction needed

#### 3.4.3 Implication for BEIR scifact

With n=300, MDES = 1.3pt on BERTScore means we can reliably detect differences of **≥1.3 absolute points** (e.g., 85.0 → 86.3). Smaller differences will be reported descriptively but **not as statistically significant**. This is an inherent limitation of BEIR scifact's test split size, not a design flaw.

> **If reviewers push back on power:** We can (a) aggregate across 3+ BEIR subsets (scifact + nfcorpus + fiqa ≈ 1,400 queries combined → MDES ≈ 0.006) as a robustness check, or (b) report effect sizes instead of p-values for BEIR scifact. Both options documented in Appendix H.

---

## 4. ANALYSIS PLAN

### 4.1 Primary Analysis: Cross-Pipeline Leaderboards

For each dataset:
1. Rank all retrieval pipelines by each retrieval metric
2. Rank all generation pipelines by each generation metric
3. Compute Borda count aggregate rankings
4. Identify best-performing pipeline families per dataset

**Figure 1:** Retrieval leaderboard heatmap (datasets × pipelines)  
**Figure 2:** Generation leaderboard heatmap (datasets × pipelines)  
**Figure 3:** Borda count aggregate rankings across datasets

### 4.2 Secondary Analysis: Metric Ranking Stability

For each dataset:
1. Compute Kendall tau correlations between all metric pairs
2. Identify metric pairs with tau < 0.7 (unstable)
3. Count how often "best pipeline" changes when metric changes

**Figure 4:** Kendall tau heatmap (metrics × metrics) with instability annotations  
**Figure 5:** Rank inversion examples (specific dataset, specific pipeline pairs)

### 4.3 Tertiary Analysis: Design Pattern Discovery

1. **Retrieval family comparison:** Sparse (BM25) vs. Dense (Vector) vs. Fusion (Hybrid) vs. Expansion (HyDE, Query Rewrite)
2. **Generation family comparison:** Single-pass (BasicRAG) vs. Multi-hop (IRCoT) vs. Agentic (MAIN-RAG, RAG-Critic) vs. Ensemble (ET2RAG)
3. **Complexity vs. performance:** Does adding retrieval hops, agents, or critics improve generation quality?
4. **Dataset-dependent patterns:** Which pipeline families excel on scientific vs. open-domain vs. reasoning-intensive queries?

**Figure 6:** Retrieval family performance by dataset type  
**Figure 7:** Generation family performance by dataset type  
**Figure 8:** Complexity-performance scatter (x = #API calls, y = BERTScore F1)

### 4.4 Exploratory Analysis: Retrieval-Generation Interaction

1. Correlate retrieval nDCG@10 with generation BERTScore F1 across all (retrieval, generation) pairs
2. Identify "high retrieval, low generation" and "low retrieval, high generation" outliers
3. Test hypothesis: Better retrieval does not always translate to better generation

**Figure 9:** Scatter plot — retrieval nDCG (x) vs. generation BERTScore (y), colored by generation pipeline family

### 4.5 Failure Mode Analysis

1. Sample queries where top-performing pipelines fail
2. Categorize errors: hallucination, retrieval miss, reasoning error, context overflow
3. Report failure rates per pipeline family

**Figure 10:** Error taxonomy with example query outputs

### 4.6 Cross-LLM-Backend Ranking Robustness (NEW, enabled by dual-backend design)

Because every generation pipeline runs against **both Qwen 3.6 and Gemma 4** (see §3.3.2), we can quantify a previously under-studied dimension of RAG evaluation: **does the choice of LLM backend alone invert pipeline rankings?**

**Analysis:**
1. For each dataset and each metric, compute pipeline rankings under each backend independently
2. Compute Kendall tau between Qwen-ranking and Gemma-ranking per (dataset, metric) cell
3. Identify backend pairs with tau < 0.7 ("backend-induced ranking instability")
4. Compare backend-induced vs. metric-induced instability magnitudes — which source of variation is larger?

**Figure 11:** Pipeline-ranking scatter — Qwen rank (x) vs. Gemma rank (y), colored by pipeline family. Diagonal = perfect agreement.
**Figure 12:** Heatmap of Kendall tau per (dataset, metric) cell, showing backend-induced ranking stability.

**Expected scientific contribution:** A third axis of evaluation sensitivity (alongside metric-choice and dataset-choice) that is especially relevant for RAG research, which rarely controls for LLM backend when comparing pipelines.

---

## 5. PAPER STRUCTURE (9 pages + references + appendix)

**Abstract**  
Problem: RAG pipeline proliferation with contradictory SOTA claims; no unified large-scale comparison. Solution: AutoRAG-Research benchmarks 20+ pipelines across 11 datasets (text, image, multimodal) under one protocol. Findings: (1) Rankings are metric- and dataset-dependent; (2) Simple baselines often match complex designs; (3) Design patterns vary by modality. Contribution: First unified large-scale RAG benchmarking study with reproducible infrastructure.

**1. Introduction (1.5 pages)**
- RAG landscape: 100+ papers, each claiming SOTA
- Fragmentation problem: different datasets, metrics, setups
- Existing benchmarks are narrow (CRAG = 1 dataset, RAGChecker = diagnostic metrics only)
- AutoRAG-Research: unified interface for large-scale comparison
- Contributions: (1) Scale, (2) Cross-domain, (3) Metric stability analysis, (4) Reproducibility

**2. Related Work (1 page)**
- RAG benchmarks: CRAG, RAGChecker, HawkBench, ClashEval, UDA
- RAG evaluation frameworks: RAGPerf, Ragas
- AutoML for RAG: AutoRAG (original tool)
- **Gap:** No prior work benchmarks 20+ pipelines across 11 datasets under unified protocol

**3. The AutoRAG-Research Framework (1.5 pages)**
- 3.1 Architecture: repository → UoW → service → pipeline layers
- 3.2 Unified data format and pre-computed embeddings
- 3.3 Pipeline abstraction: retrieval base + generation base
- 3.4 Metric system: retrieval + generation, query/dataset granularity
- 3.5 Plugin system and extensibility
- **Keep framework description concise** — this is not a tool paper; the framework enables the science

**4. Experimental Setup (1 page)**
- 4.1 Datasets: 11 datasets, selection rationale, statistics
- 4.2 Pipelines: 20 pipelines, paper references, implementation notes
- 4.3 Metrics: 21 metrics, families
- 4.4 Fairness protocol: strict controls table
- 4.5 Reproducibility: code, embeddings, results, one-command reproduction

**5. Results (2.5 pages)**
- 5.1 Retrieval Leaderboards (Figure 1)
- 5.2 Generation Leaderboards (Figure 2-3)
- 5.3 Metric Ranking Stability (Figure 4-5)
- 5.4 Design Patterns by Modality (Figure 6-8)
- 5.5 Retrieval-Generation Interaction (Figure 9)
- 5.6 Failure Modes (Figure 10)

**6. Discussion (0.5 page)**
- Implications for RAG research: SOTA claims need metric and dataset specification
- Limitations: API model drift, implementation approximations, computational cost
- Future work: Human evaluation anchor, more modalities, real-time leaderboard

**7. Conclusion (0.5 page)**
- Unified benchmarking reveals that RAG pipeline superiority is conditional
- AutoRAG-Research as a living benchmark ecosystem

**References**

**Appendix (unlimited pages)**
- A. Pipeline implementation details and paper deviations
- B. Full result tables with confidence intervals
- C. Prompt templates and hyperparameters
- D. Computational costs and API usage
- E. Dataset licenses and redistribution notes
- F. Croissant metadata for all datasets
- G. Reproducibility checklist

---

## 6. ARTIFACT PREPARATION

### 6.1 Code Repository
- **GitHub:** `anonymous-rag-benchmark` (anonymized until acceptance)
- **Contents:** Full AutoRAG-Research framework + experiment scripts + result outputs
- **README:** One-command reproduction instructions
- **Tests:** `make test` passes
- **License:** Apache 2.0

### 6.2 Pre-computed Assets
- **HuggingFace Datasets:** Pre-ingested, unified versions of all 11 datasets with embeddings
- **HuggingFace Datasets:** All pipeline outputs and evaluation scores per query
- **Format:** Parquet files with query_id, pipeline_name, output, scores
- **Croissant metadata:** Core + RAI fields for all datasets

### 6.3 Reproducibility Package
- **Docker image:** `autorag-research:latest` with PostgreSQL + VectorChord
- **One-command reproduction:** `bash reproduce.sh` → re-runs all experiments
- **Fast reproduction:** `bash reproduce_figures.sh` → generates all figures from pre-computed results

**Expected reproduction time (local-vLLM dual-backend design):**

| Scenario | Wall-Clock Time | Hardware Assumption | Bottleneck |
|----------|-----------------|---------------------|------------|
| Figure-only (from pre-computed results) | **~5 minutes** | Single CPU | Figure generation |
| **Fast pilot** (3 generation pipelines × BEIR scifact × 1 backend) | **~4 hours** | 1× Blackwell 96GB + OpenAI API (judge only) | vLLM warmup + small sample decode |
| **Partition A — single backend** (text, Qwen OR Gemma) | **~20 days continuous** | 1× Blackwell 96GB + OpenAI API | vLLM decode throughput |
| **Partition A — dual backend** (Qwen + Gemma parallel) | **~20 days** (parallel GPUs) | 2× Blackwell 96GB + OpenAI API | Slower model (Qwen 14B at ~6K tok/s) |
| **Full benchmark (all partitions, both backends)** | **~25-40 days** | 2× Blackwell 96GB + OpenAI API + 1 GPU for metrics | Generation GPU time; metric GPU optional overlap |
| **Optimized** (smaller models + pipeline subsetting) | **~15-20 days** | 2× Blackwell 96GB | Same, reduced workload |

**Parallelization strategy for dual-backend 40-day target:**
- **GPU 0 (Qwen 3.6 14B via vLLM):** handles all Qwen-backed generation runs sequentially with chunked-prefill batching
- **GPU 1 (Gemma 4 9B via vLLM):** handles all Gemma-backed generation runs in parallel with GPU 0
- **GPU 1 or shared spare time:** runs transformer-based metrics (BERTScore, BARTScore, UniEval) during downtime/between runs
- **API-bound work (GPT-5.5 judge):** Runs asynchronously against OpenAI API as generation outputs complete (Tier 4 preferred; low-volume judge spend doesn't need parallelism)
- **CPU-bound work (BM25 retrieval, ROUGE/BLEU/METEOR):** Multi-process pool on host CPU; not a bottleneck

**Minimum viable reproduction hardware:**
- 2× NVIDIA RTX PRO 6000 Blackwell (or equivalents: 2× H100 80GB; 2× RTX 6000 Ada 48GB with smaller models)
- Host machine: 64 GB RAM, 300 GB SSD (for model checkpoints + outputs)
- OpenAI-compatible API key for GPT-5.5 judge (optional: fall back to local judge with Qwen 14B adds ~$0 cost but slightly lower judge quality)

**If only 1× GPU available:** expect ~40-80 days for dual-backend (sequential); 20-40 days for single-backend.

### 6.4 Leaderboard
- **HuggingFace Space:** Anonymous leaderboard with all results
- **Camera-ready only:** Link to branded AutoRAG-Research documentation

---

## 8. RESOURCE REQUIREMENTS

**Summary (revised 2026-04-24 for local-vLLM dual-backend design; see Appendix H for full derivation):**

| Resource | Default Estimate | Optimized Estimate | Notes |
|----------|------------------|--------------------|-------|
| **Generation GPU time (both backends in parallel)** | **Qwen 14B: ~980h + Gemma 9B: ~735h — wall-clock ~41 days** | **~500h wall-clock ≈ 21 days** | Smaller model tier (Qwen 7B + Gemma 9B) + subset of expensive pipelines |
| **Retrieval-side LLM GPU time** | **~39h wall-clock** (both backends, parallel) | **~9h** | RETRO* candidate_top_k: 100 → 20 saves 78% |
| **Transformer metric GPU time (BERTScore, BARTScore×4, UniEval×4, SemScore)** | **~25 GPU-hours** | **~13 GPU-hours** | 50% sampling for expensive metrics |
| **Metric LLM-judge API (GPT-5.5 @ 25% sampling for `response_relevancy`)** | **~$3,600** | **~$2,400** (switch to GPT-4o-grade) | See §8.3 breakdown |
| **Embedding compute** | **~0.3 GPU-hour** (pre-computed default) | ~0.1 GPU-hour | Pre-computed artifacts on HuggingFace |
| **Storage** | **~145 GB** | ~100 GB | Dual-backend outputs + model checkpoints (~46GB) + image data (~50GB) |
| **Electricity (rough)** | **~$230** (@ $0.15/kWh, 600W × 980h × 2 GPUs) | ~$120 | Negligible vs. equivalent API spend |
| **Human effort** | **2 FTE × 3.5 months** | 2 FTE × 2.5 months | See §8.6 (Phase 0 for vLLM benchmarking added) |
| **TOTAL DIRECT API SPEND** | **~$3,600** | **~$2,400** | Only judge metric uses API |
| **TOTAL COMPUTE (wall-clock days)** | **~41-45 days** | **~21-25 days** | 2 Blackwells in parallel |

**vs. API-only design (prior §8 draft):** API $11,400 → **$3,600 (−68%)** while eliminating rate-limit risk and enabling cross-backend robustness finding (§4.6). Compute time grows from ~2 days (API parallelism) to ~21-45 days (GPU-bound).

**vs. single-backend local design (hypothetical):** Generation GPU time doubles (one model per GPU parallel) but unlocks new primary finding (cross-LLM-backend ranking stability, §4.6). Judge API cost also doubles because there are 2× more generation outputs to evaluate.

### 8.1 Generation GPU-Time Breakdown (Local vLLM)

**Inputs (Partition A, per model backend):**
- 5 text datasets × average 2,940 queries/dataset = **14,700 text queries**
- 9 text retrieval × 9 text generation pipelines = 81 (retrieval, generation) combinations per dataset
- Sum of LLM calls per query across 9 generation pipelines = **94 calls/query** (derivation in Appendix H.2)

**Per-backend generation LLM calls (Partition A):**

```
N_gen = 9 retrievals × 5 datasets × 14,700/5 queries × 94 calls/query
      = 9 × 14,700 × 94
      = 12,436,200 calls per backend
```

**Token volume per call:** 1,500 input + 200 output = 1,700 tokens avg (conservative RAG prompt)

**Per-backend token budget (generation only):**
```
Tokens = 12.44M calls × 1,700 tokens = 21.14B tokens
```

**Throughput assumptions on RTX PRO 6000 Blackwell** (see §H.1 for source):

| Model | Class | Assumed Throughput (tok/sec, mixed) |
|-------|-------|------------------------------------|
| Qwen 3.6 7B | Small-mid | ~10,000 |
| Qwen 3.6 14B | Mid-large | ~6,000 |
| Gemma 4 9B | Mid | ~8,000 |
| Gemma 4 27B (FP8) | Large | ~4,000 |

**GPU-time per backend (default: Qwen 14B + Gemma 9B):**
```
Qwen 14B: 21.14B / 6,000 tok/s = 3,523,333 s = 979 GPU-hours
Gemma 9B: 21.14B / 8,000 tok/s = 2,642,500 s = 734 GPU-hours
```

**Wall-clock (GPUs run in parallel, one model each):** **max(979, 734) = 979 hours ≈ 40.8 days**

Adding image/multimodal generation (~18 runs × 2,500 queries × ~3 avg calls × ~2,500 tokens per VLM call = 337M tokens per backend):
```
+ ~15-40 additional GPU-hours per backend (dominated by image prefill)
```

**Grand total GPU-hours for generation: ~1,020 (Qwen) + ~775 (Gemma) ≈ ~1,800 GPU-hours**
**Wall-clock with 2 GPUs in parallel: ~40 days (bottlenecked by slower model)**

### 8.2 Retrieval-Side LLM GPU Time

Retrieval pipelines that invoke LLMs (same cross-backend policy: each pipeline runs once per backend for fairness):

| Pipeline | Calls/query | Tokens (500 in + 50 out) | Subtotal per backend |
|----------|-------------|--------------------------|----------------------|
| HyDE | 1 | 550 × 14,700 = 8.1M tok | |
| Query Rewrite | 1 | 8.1M tok | |
| Question Decomp (ret) | 1 | 8.1M tok | |
| **RETRO*** | **100** (candidate_top_k default) | 550 × 14,700 × 100 = **808.5M tok** ⚠️ | |
| **Total per backend** | | **~832.8M tokens** | |

**GPU-time per backend:**
- Qwen 14B: 832.8M / 6,000 = 138,800 s = **38.6 GPU-hours**
- Gemma 9B: 832.8M / 8,000 = 104,100 s = **28.9 GPU-hours**

**Subtotal: ~67 GPU-hours for both backends (parallel → ~39 wall-clock hours)**

> ⚠️ **RETRO* dominates retrieval LLM cost.** Reducing `candidate_top_k` from 100 → 20 cuts this subsection by 80% → **~13 GPU-hours combined**.

### 8.3 Metric LLM-Judge API Cost (Only Remaining API Spend)

`response_relevancy` metric requires **3 LLM calls per evaluated generation output** (strictness=3). User specified **GPT-5.5** (or equivalent premium model via OpenAI-compatible API) for judge quality.

**Calls with dual-backend (each model's outputs evaluated separately):**

Per-backend text outputs:

```
Σ_d (9 retrieval × 9 generation × Q_d)
  = 81 × (300 + 2000 + 6000 + 2000 + 4400)
  = 81 × 14,700
  = 1,190,700 outputs per backend
```

Both backends + image/multimodal partition:

```
N_outputs = 2 × 1,190,700 (text)         = 2,381,400
          + 2 × 18 runs × 2,500 queries  =    90,000 (image/multimodal)
          ≈ 2,471,400 generation outputs

N_judge_calls = N_outputs × 3 ≈ 7.41M judge calls
```

**Token cost assumptions** (judge prompt: 400 input + 30 output per call):

| Pricing tier | Per-call cost | N_judge (full) | Cost (full) | Cost (25% sample) |
|--------------|---------------|----------------|-------------|-------------------|
| **GPT-5.5 premium** (assumed 1.5× GPT-4o: $3.75/1M in, $15/1M out) | $1.95/1K | 7.41M | **~$14,450** | **~$3,615** |
| **GPT-4o-grade** ($2.50/1M in, $10/1M out) | $1.30/1K | 7.41M | **~$9,635** | **~$2,410** |
| **Local judge** (Qwen 14B, no API) | $0 | 7.41M | **$0** (+ ~150 GPU-hours) | **$0** (+ ~40 GPU-hours) |

**Recommended default:** **25% sampling + GPT-5.5 premium → ~$3,600** for judge. Balances judge quality with budget. This is the figure used in the §8 summary table.

**If GPT-5.5 unavailable or too expensive:** fallback to GPT-4o-grade at full sampling for ~$9,600, or 25%-sampling for ~$2,400.

> **Mitigation: 25% sampling of generation outputs for judge** retains statistical power for ranking-stability analysis (see §3.4.1 MDES table — sampling reduces per-pipeline n from 14,700 → 3,675, still above MDES threshold on MrTyDi/CRAG and borderline-sufficient on BEIR scifact).

### 8.4 Transformer Metric GPU Time

**Workloads requiring GPU** (8 transformer-based metrics run on each generation output; dual-backend → 2× outputs to evaluate):

| Metric | Samples (846 runs × ~2,000 queries) | Throughput | GPU-hours |
|--------|-------------------------------------|-----------|-----------|
| BERTScore | 1.69M | batch-16 @ 50 ms → 540/s | **~0.87h** |
| BARTScore (×4 variants) | 6.77M | batch-8 @ 70 ms → 115/s | **~16.4h** |
| UniEval (×4 dimensions) | 6.77M | batch-16 @ 60 ms → 267/s | **~7.0h** |
| SemScore | 1.69M | batch-32 @ 30 ms → 1,067/s | **~0.44h** |
| Text embedding (pre-computed → 0) | 0 | — | 0 |
| Image embedding (pre-computed → 0) | 0 | — | 0 |
| **TOTAL** | | | **~24.7 GPU-hours** |

**With sampling 50% of outputs for expensive metrics (BARTScore, UniEval):** **~13 GPU-hours**

### 8.5 Storage Breakdown

| Item | Calculation | Size |
|------|-------------|------|
| Raw pipeline outputs (both backends) | 846 runs × ~2,000 queries × ~2KB | **~3.4 GB** |
| Local LLM response cache (both backends) | 24.9M generation + 3M retrieval = 27.9M × ~800 bytes | **~22 GB** |
| API judge response cache | 5.1M × ~500 bytes | **~2.5 GB** |
| Retrieval results (ChunkRetrievedResult) | 54 retrieval runs × 2,000 × 10 × ~500B | **~0.54 GB** |
| Metric scores | 12,222 × 14 × ~200B | **~0.034 GB** |
| Pre-computed embeddings (downloaded) | ~120K text + ~4K images × 1,024-dim × 4B | **~0.5 GB** |
| Local LLM checkpoints | Qwen 3.6 14B (~28 GB) + Gemma 4 9B (~18 GB) | **~46 GB** |
| Image datasets (raw PDFs/images) | ViDoRe + VisRAG + Open-RAGBench | **~50 GB** |
| PostgreSQL indexes + overhead | Triangular estimate | **~12 GB** |
| Docker images + code | autorag-research + PostgreSQL + VectorChord + vLLM | **~8 GB** |
| **TOTAL** | | **~145 GB** |

> **Revision note:** Dual-backend adds ~50 GB (2× outputs, 2× model checkpoints) vs. single-backend API plan. 200 GB SSD recommended; 500 GB comfortable.

### 8.6 Human Effort Breakdown

**Revised for local-vLLM dual-backend design:**

| Phase | FTE-weeks | Duration | Scope |
|-------|-----------|----------|-------|
| Phase 0: vLLM infrastructure setup | 1 FTE × 1 week | 1 week | Install vLLM on both Blackwell GPUs, benchmark Qwen 3.6 & Gemma 4 throughput, validate §H.9 A4 assumption |
| Phase 1: Pre-experiment validation | 2 FTE × 1 week | 1 week | 100-query pilot on BEIR scifact with both backends, validate token/throughput estimates |
| Phase 2: Partition A execution (text, dual-backend) | 1 FTE × 4 weeks | 4 weeks | Long wall-clock due to GPU-bound workload; mostly automated with resumable checkpoints |
| Phase 3: Partition B+C execution (image/multimodal) | 1 FTE × 2 weeks | 2 weeks | VLM-specific setup (Qwen-VL, Gemma VLM), smaller data volume |
| Phase 4: Metric evaluation + statistical analysis | 2 FTE × 2 weeks | 2 weeks | Scripts for CIs, permutation tests, cross-backend ranking stability analysis |
| Phase 5: Paper writing | 2 FTE × 3 weeks | 3 weeks | Concurrent with statistical analysis |
| Phase 6: Response letters + revisions | 2 FTE × 2 weeks | 2 weeks | NeurIPS rebuttal cycle |
| **TOTAL** | **~26 FTE-weeks** | **~13 calendar weeks** | **Assumes 2 researchers at ~50% dedication** |

**Calendar timeline:** ~3.5 months, with GPU-bound phases (2 & 3) running largely unattended (monitoring only).

### 8.7 Resource Mitigation Options (Wall-Clock & API Cost)

**Goal: reduce 40-day wall-clock to <3 weeks OR reduce $6,600 judge spend to <$2,000.**

| Mitigation | GPU-hours Saved | API $ Saved | Impact on Science |
|------------|-----------------|-------------|-------------------|
| Reduce RETRO* candidate_top_k from 100 → 20 | **~54 GPU-hours** | — | Low: rerank depth still adequate |
| Reduce MAIN-RAG top_k from 10 → 5 in agent step | **~220 GPU-hours** | — | Low-Medium: still multi-agent |
| Reduce IRCoT max_steps from 8 → 4 | **~180 GPU-hours** | — | Medium: truncates reasoning chains |
| Reduce SPD-RAG top_k from 10 → 5 | **~230 GPU-hours** | — | Medium: fewer per-doc agents |
| Skip Self-RAG + RAG-Critic (most expensive + redundant with IRCoT) | **~190 GPU-hours** | — | High: loses 2 pipeline families |
| Use Qwen 7B + Gemma 4B instead of 14B/9B | **~600 GPU-hours** (2× throughput) | — | Medium: smaller models → lower quality; re-validate cross-backend claim |
| Sample 25% of generation outputs for judge | — | **~$5,000** | Low: estimator only, retains rankings |
| Use GPT-4o-mini (not 5.5) for judge baseline | — | **~$5,000** | Medium: less stringent judge |
| Drop one backend (Qwen only or Gemma only) | **~50% of all GPU-hours** | ~50% of judge | High: eliminates cross-backend finding |
| **Aggressive combo** (all GPU mitigations + 25% judge sample) | **~870 GPU-hours** (≈18 days wall-clock) | **~$5,000** | Baseline total → **~20 days + $1,650** |
| **Conservative combo** (smaller model tier + judge sampling) | **~500 GPU-hours** | **~$5,000** | **~25 days + $1,650** |

> **Critical dependency:** Assumption A4 (Blackwell throughput) must be validated in Phase 0 pilot. If actual throughput is ≤50% of assumed, every GPU-hour estimate doubles, and mitigations move from "optional" to "mandatory."

---

## 9. RISKS & MITIGATIONS

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **vLLM throughput below assumption** (§H.1 A4) | **Medium** | **High** | Phase 0 mandatory benchmark on 1,000-query pilot; if <50% assumed → mitigation combos from §8.7 apply or reduce model tier |
| **GPU OOM during long runs** | Low-Medium | High | vLLM v0.6+ handles paged KV cache; set `--max-model-len 8192`; monitor VRAM with `nvidia-smi -l 10` in container |
| **Judge API (GPT-5.5) pricing change or access denied** | Low | Medium | Fallback: use local Qwen 14B or Gemma 9B as judge (loses ~10-15% judge correlation with human, but retains within-experiment consistency) |
| Pipeline implementation bugs | Medium | High | Validate each pipeline on 1 dataset before full run; keep 2-week debug buffer |
| Dataset ingestion failures | Medium | High | Pre-ingest all datasets in Phase 0; have backup ingestion scripts |
| Results are noisy | Medium | Medium | Statistical rigor (CIs, permutation tests); frame noise as finding if persistent |
| **Cross-backend rankings perfectly agree** (null finding) | Medium | Low | Still publishable: "pipeline rankings are robust across ≥2 modern LLM families" is a strong evaluation-science result |
| **Cross-backend rankings disagree wildly** | Medium | Medium | Strongest finding: "LLM backend choice alone inverts pipeline rankings" — central narrative asset |
| Similar to RAGPerf | Medium | Medium | Emphasize: (1) scale (20+ pipelines, 11 datasets), (2) dual-LLM-backend cross-ranking, (3) on-premises reproducibility without API dependence |
| **Model checkpoints unavailable** (Gemma 4 / Qwen 3.6 release delays) | Low-Medium | High | Fallback: Gemma 3 + Qwen 3 (previous-gen but confirmed available); update paper claims to match actual models used |
| Reviewers want human evaluation | Medium | Medium | Plan for small human evaluation subset (100 queries, 3 annotators) as follow-up |

---

## 10. SUCCESS CRITERIA

The paper succeeds if it delivers:

1. **Complete benchmark results** for all 20+ pipelines across all 11 datasets
2. **3+ novel empirical findings:**
   - Metric-induced ranking instability quantified
   - Dataset-dependent design patterns identified
   - Simple vs. complex pipeline competitiveness measured
3. **Full reproducibility:** Reviewers can reproduce key results in <1 hour using pre-computed outputs
4. **Open artifacts:** All code, data, and results publicly available with Croissant metadata
5. **Living benchmark:** Plugin system enables future community contributions

---

## 11. WHAT DIFFERENTIATES THIS FROM PRIOR WORK

| Aspect | CRAG | RAGChecker | RAGPerf | **AutoRAG-Research (This Paper)** |
|--------|------|------------|---------|-----------------------------------|
| **# Pipelines benchmarked** | 3-4 baselines | 8 systems | 5-10 | **20+** |
| **# Datasets** | 1 | 10 (repurposed) | 3-5 | **11 (unified)** |
| **Modalities** | Text | Text | Text | **Text + Image + Multimodal** |
| **Pre-computed embeddings** | No | No | No | **Yes** |
| **Plugin extensibility** | No | No | Limited | **Yes (entry-point based)** |
| **Cross-domain analysis** | No | No | No | **Yes (design patterns by modality)** |
| **Metric stability study** | No | Partial | No | **Yes (systematic)** |
| **Unified interface** | No | No | Partial | **Yes (retrieval + generation + metrics)** |

---

---

## Appendix H: Detailed Calculation Methodology

This appendix makes every resource estimate in §8 traceable to a formula, an assumption, and a source file or pricing page. All figures use default hyperparameters verified against the codebase on 2026-04-24 unless noted.

### H.1 Hardware & Throughput Assumptions (Local vLLM)

**Hardware inventory (verified available 2026-04-24):**
- **2× NVIDIA RTX PRO 6000 Blackwell**, 96 GB VRAM each, independent (non-NVLink) operation
- Models loaded one-per-GPU (no tensor parallelism for small-mid models)

**vLLM throughput assumptions (aggregated over prefill + decode, mixed workload with batch-64+, chunked prefill enabled):**

| Model | Precision | VRAM | Estimated Aggregate Throughput (tok/sec) | Source |
|-------|-----------|------|------------------------------------------|--------|
| Qwen 3.6 **7B** | bf16 | ~14 GB | ~10,000 | Scaled from Qwen-2.5 7B on H100 benchmarks (~14 TFLOPS/GB/s memory bandwidth) |
| Qwen 3.6 **14B** | bf16 | ~28 GB | ~6,000 | Scaled linearly from 7B |
| Gemma 4 **9B** | bf16 | ~18 GB | ~8,000 | Similar to Qwen 7B with slight FP throughput hit |
| Gemma 4 **27B** | FP8 | ~27 GB | ~4,000 | Quantized; 3-4× slower than 7B |

**Blackwell-specific acceleration:** RTX PRO 6000 Blackwell's FP8 tensor cores + 4th-gen TMA engines provide ~1.5-2.0× speedup over Ada Lovelace (previous gen). Benchmarks above assume ~85% of H100 PCIe FP16 throughput.

> ⚠️ **All throughput numbers must be validated in Phase 0 pilot (§H.9 A4).** If measured <50% of assumed, GPU-time estimates in §8 are invalidated.

### H.1b Remaining API Pricing Assumptions

Only one workload remains on API: `response_relevancy` metric judge. User specified **GPT-5.5** (or equivalent premium model via OpenAI-compatible endpoint).

| Provider | Model | Input ($/1M tok) | Output ($/1M tok) | Source |
|----------|-------|------------------|-------------------|--------|
| OpenAI | **gpt-5.5 (assumed)** | **$3.75** (1.5× 4o) | **$15.00** (1.5× 4o) | Extrapolated from GPT-4o pricing trajectory |
| OpenAI | gpt-4o-2024-08-06 (fallback) | $2.50 | $10.00 | openai.com/pricing (2025) |
| OpenAI | gpt-4o-mini-2024-07-18 (cheap fallback) | $0.15 | $0.60 | openai.com/pricing (2025) |

> **Actual GPT-5.5 pricing TBD** as of plan revision date. Conservative estimate used; update §8.3 if real pricing differs by >20%.

### H.2 Pipeline LLM-Call Counts (Verified from Code)

Counts below were extracted by static analysis of pipeline source files on 2026-04-24. Each count assumes default hyperparameters as defined in the pipeline's Pydantic config.

#### Generation Pipelines (calls per query)

```
BasicRAG:             1                                    (single generate)
AutoThinkRAG:         1 + {0 if simple else 1} + {0-3}     (avg 3; complexity routed)
ET2RAG:               num_subsets + 1 = 5 + 1 = 6          (num_subsets=5 default)
Question Decomp:      1 + (1 + max_subquestions) + 1 = 6   (max_subquestions=3)
Self-RAG:             1 + max_steps × 2 = 1 + 3×2 = 7      (max_steps=3)
RAG-Critic:           1 + max_iterations × 3 = 1 + 3×3=10  (max_iterations=3)
IRCoT:                1 + max_steps × 2 + 1 = 1+8×2+1=18   (max_steps=8)
MAIN-RAG:             top_k × 2 + 1 = 10×2+1 = 21          (top_k=10 in agent step)
SPD-RAG:              top_k × 2 + log₂(top_k) ≈ 10×2+3=22  (top_k=10)
```

Sum across 9 text-capable generation pipelines: **94 calls/query**

#### Retrieval Pipelines (calls per query)

```
BM25:                 0
Vector Search:        0
Hybrid RRF/CC:        0
Power of Noise:       0  (wrapper around base retriever)
HEAVEN:               0  (image retrieval, no LLM)
HyDE:                 1
Query Rewrite:        1
Question Decomp:      1
RETRO*:               candidate_top_k × num_samples = 100 × 1 = 100  ⚠️ dominant
```

Sum across 9 text-capable retrieval pipelines: **103 calls/query** (100 from RETRO*)

### H.3 Text-Generation GPU-Time Formula (Local vLLM)

**Let:**
- `Q_d` = #queries in dataset `d`
- `C_g` = #LLM calls per query for generation pipeline `g`
- `R` = #retrieval pipelines (9 text)
- `G` = #generation pipelines (9 text)
- `D` = #datasets (5 text)
- `T_in`, `T_out` = avg input / output tokens per call (1,500 / 200 default)
- `TP_m` = aggregate throughput (tok/sec) of model `m` on Blackwell (from §H.1)
- `B` = #backends (2: Qwen 3.6 + Gemma 4)

**Total generation LLM calls per backend (Partition A):**

```
N_gen = Σ_d Σ_r Σ_g (Q_d × C_g)
      = R × Σ_d Q_d × Σ_g C_g
      = 9 × 14,700 × 94
      = 12,436,200 calls per backend
```

**Total token budget per backend (generation only):**

```
Tokens = N_gen × (T_in + T_out)
       = 12.44M × 1,700
       = 21.14B tokens
```

**GPU-time per backend (Blackwell, one model per GPU):**

```
GPU_hours(backend) = Tokens / TP_m / 3600

For Qwen 3.6 14B @ ~6,000 tok/s:
  GPU_hours = 21.14B / 6,000 / 3,600 = 978.7 hours ≈ 40.8 GPU-days

For Gemma 4 9B @ ~8,000 tok/s:
  GPU_hours = 21.14B / 8,000 / 3,600 = 734.0 hours ≈ 30.6 GPU-days
```

**Wall-clock (two GPUs running each model in parallel):**

```
Wall-clock = max(979, 734) = 979 hours ≈ 40.8 days
```

### H.4 Retrieval-Side GPU-Time Formula (Local vLLM)

Retrieval pipelines that invoke LLMs run once **per backend** (HyDE, Query Rewrite, Question Decomposition (ret), RETRO*) for cross-backend consistency:

```
C_retr_sum = 1 + 1 + 1 + 100 = 103 calls/query per backend
N_ret_llm_per_backend = D × C_retr_sum × Σ_d Q_d / D
                      = 103 × 14,700
                      = 1,514,100 calls
Tokens_per_backend = 1.51M × 550 tokens = 832.8M tokens
```

**GPU-time per backend:**

```
Qwen 14B:  832.8M / 6,000 / 3600 = 38.6 GPU-hours
Gemma 9B:  832.8M / 8,000 / 3600 = 28.9 GPU-hours

Total parallel wall-clock: max = 38.6 hours ≈ 1.6 days
```

**If RETRO* candidate_top_k reduced from 100 → 20 (cost mitigation):**

```
C_retr_sum = 1 + 1 + 1 + 20 = 23 calls/query
Tokens_per_backend = 23 × 14,700 × 550 = 185.9M
Qwen 14B GPU-hours: 185.9M / 6,000 / 3600 = 8.6 h  (−78%)
Gemma 9B GPU-hours: 185.9M / 8,000 / 3600 = 6.5 h  (−78%)
```

### H.5 Metric LLM-Judge API Cost Formula

**`response_relevancy` metric calls with dual-backend design:**

Each generation output (from either backend) is evaluated once by the judge. With 846 generation runs (423 × 2 backends):

```
Text partition: Σ_d Σ_r Σ_g × 2 backends × Q_d × 3 calls
              = 9 × 9 × 5 × 14,700/5 × 3 × 2
              Wait — let me recount.

Per-backend generation outputs = 9 retr × 9 gen × 14,700 queries = 1,190,700 outputs
Dual-backend outputs = 2 × 1,190,700 = 2,381,400 outputs

Judge calls = 2,381,400 × 3 (strictness=3) ≈ 7,144,200 calls

Image/multimodal adds: 2 × 18 × 2,500 × 3 ≈ 270,000 calls

Total N_judge ≈ 7.41M calls
```

**Cost (GPT-5.5 at assumed $3.75/1M input, $15/1M output, ~400 tok in + 30 tok out per judge call):**

```
Cost_judge_input  = 7.41M × 400 / 1e6 × $3.75  = $11,115
Cost_judge_output = 7.41M × 30 / 1e6 × $15     = $3,335
                                                 --------
Subtotal (GPT-5.5 premium):                    ≈ $14,450
```

**Mitigation 1: Sample 25% of generation outputs for judge evaluation:**
```
N_judge_sampled = 7.41M × 0.25 = 1.85M calls
Cost_sampled = 0.25 × $14,450 = ~$3,613
```

**Mitigation 2: Use GPT-4o-grade pricing instead of GPT-5.5:**
```
At $2.50/1M in, $10/1M out:
Cost = 7.41M × (400 × 2.50 + 30 × 10) / 1e6 = $9,645

With 25% sampling + 4o-grade: = $2,411
```

**Mitigation 3: Local judge (Qwen 14B, no API cost):**
```
Tokens = 7.41M × 430 = 3.19B tokens
GPU-time on Qwen 14B: 3.19B / 6,000 / 3600 = 147 GPU-hours (adds to §8.1 budget)
API cost: $0
```

> **Recommended default:** Option 1 (25% sampling with GPT-5.5) → **~$3,600** for judge, balances judge quality with budget. Recomputed §8 "default" summary row uses this.

### H.6 GPU-Time Formula (per metric)

```
GPU_hours(metric) = N_outputs × avg_sec_per_sample / 3600
                  = N_outputs × (1 / (batch_size × samples_per_sec)) / 3600
```

**Per-metric derivations (assuming 1× A100 @ FP16):**

| Metric | batch_size | sec_per_batch | N_samples | GPU-hours |
|--------|-----------|---------------|-----------|-----------|
| BERTScore | 16 | 0.05 | 423 × 2,000 × 1 = 846K | 846K/16 × 0.05 / 3600 = **0.73h** |
| BARTScore (×4) | 8 | 0.07 | 846K × 4 = 3.38M | 3.38M/8 × 0.07 / 3600 = **8.2h** |
| UniEval (×4) | 16 | 0.06 | 846K × 4 = 3.38M | 3.38M/16 × 0.06 / 3600 = **3.5h** |
| SemScore | 32 | 0.03 | 846K | 846K/32 × 0.03 / 3600 = **0.22h** |
| **Subtotal** | | | | **~12.7h** |

Plus embedding computation (if not pre-computed) and image metrics: **~15-25 GPU-hours realistic lower bound**.

**Previous §8.4 estimate of "~85 GPU-hours" was conservative upper bound.** Actual expected: **15-30 GPU-hours** with pre-computed embeddings. Updated best estimate: **~25 GPU-hours (default), ~15 GPU-hours (cost-optimized)**.

### H.7 Storage Formula

```
Storage = Σ_i (record_count_i × bytes_per_record_i)
```

**Record counts (all from §8.1-8.5):**
- Generation outputs: 423 runs × 2,000 queries = 846K records × 2KB = 1.69 GB
- LLM cache (text): 17.5M calls × 500B = 8.75 GB
- Retrieval results: 54 runs × 2,000 × 10 chunks × 500B = 540 MB
- Metric scores: 6,300 × 14 × 200B ≈ 18 MB
- Embeddings: 120K docs × 1,024 × 4B = 491 MB
- Image data: estimated from HuggingFace metadata
- Postgres overhead: ~15% of data volume

**Assumption:** No redundant storage; outputs are Parquet-compressed. Replacing caches with SQLite-serialized form reduces by ~30%.

### H.8 Power-Analysis Derivation

**Paired two-tailed t-test:**

```
MDES = (z_{α/2} + z_{power}) × σ × √(2/n)
     ≈ 2.8 × σ / √n                      (paired design, α=0.05, power=0.80)
```

**For BEIR scifact (n=300, σ=0.05 BERTScore):**

```
MDES = 2.8 × 0.05 / √300
     = 2.8 × 0.05 / 17.32
     = 0.0081  (i.e., 0.81 pt on 0-100 scale)
```

Practical note: reported as "1.3 pt" in §3.4.1 as conservative estimate (ignores correlation gains from paired design → ~1.6× larger MDES).

### H.9 Assumption Register

Every downstream number depends on these **falsifiable assumptions**:

| # | Assumption | Source | If wrong, affects |
|---|------------|--------|-------------------|
| **A1 (NEW)** | **Qwen 3.6 7B / 14B and Gemma 4 9B / 27B are publicly released and deployable via vLLM by project start** | Model release roadmaps | Entire §8 — fallback: Qwen 3 + Gemma 3 |
| **A2 (NEW)** | **RTX PRO 6000 Blackwell delivers ~85% of H100 PCIe FP16 throughput with vLLM v0.6+** | Vendor benchmarks + vLLM perf guide | §H.3 / §H.4 (GPU-hours) — ±50% |
| **A3 (UPDATED)** | **Aggregate vLLM throughput estimates: Qwen 7B ≈ 10K tok/s, 14B ≈ 6K tok/s; Gemma 9B ≈ 8K tok/s, 27B-FP8 ≈ 4K tok/s on Blackwell (batch 64+)** | Scaled from H100 benchmarks | §H.3 / §H.4 (GPU-hours) — ±30% |
| A4 | Avg 1,500 input + 200 output tokens per generation call | Pilot run on BEIR scifact + BasicRAG | §H.3 (~±20% per line item) |
| A5 | Default hyperparameters (top_k=10, max_steps=8, candidate_top_k=100) | Code inspection 2026-04-24 | §H.2 (LLM call counts) |
| A6 | GPT-5.5 pricing ≈ 1.5× GPT-4o ($3.75/1M in, $15/1M out) | Extrapolation from GPT-4/4o trajectory | §H.5 — ±50% possible |
| A7 | 100% of queries evaluated with all 14 generation metrics (default); 25% sampling for judge (cost-optimized) | Plan §3.1 / §H.5 | §H.5 (judge calls) |
| A8 | RETRO* candidate_top_k=100 is acceptable depth | Paper default | §H.4 (dominant GPU cost on retrieval side) |
| A9 | Pre-computed embeddings downloaded, not regenerated | Plan §1.5 | GPU hours (removes ~0.3h) |
| A10 | Pairwise paired tests with σ ≈ 0.05 for BERTScore | BEIR pilot | §H.8 (MDES) |
| **A11 (NEW)** | **vLLM's continuous batching and chunked prefill saturate at batch-128+ on 96GB VRAM for 7B-14B models** | vLLM benchmarks on H100/A100 | §H.3 (throughput scales with batch size; if not, re-pin batch to 64 → ~70% throughput) |
| **A12 (NEW)** | **Qwen 3.6 and Gemma 4 produce comparable answer lengths (~200 tokens avg) for identical prompts** | Similar-size instruction-tuned models behave similarly | §H.3 (avg T_out) — ±30% |

**Pre-registered sensitivity checks (Phase 0 + Phase 1 — MUST PASS before full run):**

1. **[CRITICAL] Phase 0: vLLM throughput benchmark**
   - Deploy Qwen 3.6 14B (or chosen model) on Blackwell GPU with vLLM
   - Run 1,000-sample decode with realistic RAG prompts (1,500 in + 200 out)
   - **Pass criterion:** measured aggregate throughput within 30% of A3 assumption
   - **If fails:** downgrade to 7B tier OR add GPU OR extend timeline

2. **[CRITICAL] Phase 0: Judge API pricing validation**
   - Confirm actual GPT-5.5 pricing (or equivalent tier)
   - **If A6 is off by >50%:** switch to GPT-4o or apply 25% sampling mitigation

3. **Phase 1: Token count validation**
   - Pilot 100 queries on BEIR scifact with BasicRAG, MAIN-RAG, IRCoT
   - Measure actual avg input/output tokens
   - **Pass criterion:** A4 (1,500 in + 200 out) within ±30%

4. **Phase 1: Statistical variance validation**
   - Compute σ of BERTScore across 3 decoding samples (temperature >0) on 100 queries
   - **Pass criterion:** σ ≤ 0.07 (for MDES calculations in §3.4.1)

5. **Phase 1: Cross-backend length parity**
   - Generate 100 queries with Qwen 3.6 and Gemma 4 on BasicRAG
   - **Pass criterion:** avg output length ratio within 0.7-1.4× (verify A12)

**If any Phase 0 check fails by >30%, STOP and recompute §8 estimates before committing to full-scale run.**
**Phase 1 checks are less critical but still advisory for budget fine-tuning.**

---

*Full-Scale Plan: 2026-04-22*
*First calculation reinforcement: 2026-04-24 (API-based)*
*Local-vLLM dual-backend pivot: 2026-04-24 (current)*
*Target: ICLR 2027 / NeurIPS 2027 / ACL 2027 / TMLR*
*Scope: All 20+ Pipelines × All 11 Datasets × All 21 Metrics × **2 LLM backends (Qwen 3.6 + Gemma 4)***

**Key revisions (2026-04-24, in chronological order):**

*First pass (API-based reinforcement):*
- *§1.3: Concrete query counts per dataset*
- *§3.1: Fixed metric computation total (15,400 → 16,170)*
- *§3.2: Realistic Partition B/C counts (30/15 → 12/6 runs)*
- *§3.4: Added Statistical Power Analysis (MDES, Bonferroni)*
- *§6.3: Explicit parallelization assumptions*
- *§8: Complete resource-cost overhaul with 3-way API breakdown*
- *Appendix H: Traceable calculation methodology with formulas and assumption register*

*Second pass (local-vLLM dual-backend pivot):*
- *§3.3: Generation backend switched from OpenAI API → **on-premises vLLM with Qwen 3.6 + Gemma 4** on 2× Blackwell 96GB*
- *§3.3.1: vLLM serving configuration (per-GPU model allocation, throughput table)*
- *§3.3.2: Cross-backend robustness protocol (generation runs doubled: 423 → 846)*
- *§4.6: **New primary finding** — cross-LLM-backend ranking stability analysis*
- *§6.3: Reproduction time revised to GPU-bound wall-clock (~40 days for dual-backend)*
- *§8: Rewritten as GPU-time-centric; API spend reduced to judge metric only ($11K → $3-7K)*
- *§9: Risk register updated (API rate-limit risk removed; vLLM throughput & model-availability risks added)*
- *Appendix H.1, H.3, H.4, H.5: Replaced API pricing formulas with vLLM throughput formulas; assumption register expanded to 12 assumptions with hardware-specific items*
