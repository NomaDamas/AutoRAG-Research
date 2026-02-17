# AutoRAG-Research Paper Planning Document

> **Last Updated:** 2025-02-17

---

## Table of Contents

1. [Literature Review](#part-1-literature-review)
   - [Reproducibility in RAG Research](#11-reproducibility-in-rag-research)
   - [Framework Papers from Major AI Conferences](#12-framework-papers-from-major-ai-conferences)
   - [Competitive Landscape](#13-competitive-landscape)
2. [Paper Positioning & Structure](#part-2-paper-positioning--structure)
   - [Positioning Statement](#21-positioning-statement)
   - [Target Venue](#22-target-venue)
   - [Proposed Title & Outline](#23-proposed-title--outline)
3. [SWOT Analysis](#part-3-swot-analysis)
4. [Features to Add & Experiments](#part-4-features-to-add--experiments)
   - [Must-Add Features (Priority 1)](#41-must-add-features-priority-1)
   - [Should-Add Features (Priority 2)](#42-should-add-features-priority-2)
   - [Experiment Designs](#43-experiment-designs)
   - [Feature Comparison Table](#44-feature-comparison-table)
5. [Timeline](#part-5-timeline)
6. [References](#references)

---

## Part 1: Literature Review

### 1.1 Reproducibility in RAG Research

The reproducibility crisis in RAG research is well-documented: studies that appear to compare RAG methods often confound algorithmic differences with infrastructure differences (embedding models, chunk sizes, retrieval backends). This makes it nearly impossible to attribute performance gains to the actual method under study.

**ReproRAG: Benchmarking Reproducibility in Retrieval-Augmented Generation**
- *Authors:* Meng et al.
- *Year:* 2025
- *Link:* [arXiv:2509.18869](https://arxiv.org/abs/2509.18869)
- *Key Findings:*
  - Conducted systematic reproducibility benchmarking across RAG systems.
  - Identified **embedding model choice** as the #1 source of variance in RAG pipeline performance — often larger than the algorithmic difference between retrieval methods.
  - Showed that preprocessing decisions (chunking, tokenization) create compounding variance.
- *Relevance to AutoRAG-Research:* Our database-backed persistence of embeddings and retrieval results directly addresses ReproRAG's findings. By storing pre-computed embeddings and allowing pipelines to share the same underlying data, we eliminate the primary source of non-reproducibility.

**TREC 2024 RAG Track**
- *Organizers:* NIST / TREC
- *Year:* 2024
- *Link:* [trec.nist.gov/pubs/trec33/papers/Overview-rag.pdf](https://trec.nist.gov/pubs/trec33/papers/Overview-rag.pdf)
- *Key Contributions:*
  - Established a standardized RAG evaluation protocol using MS MARCO v2.1 corpus.
  - Introduced the Ragnarök framework for systematized end-to-end RAG evaluation.
  - Defined shared retrieval and generation sub-tasks to isolate component contributions.
- *Relevance:* Validates the need for standardized evaluation infrastructure. AutoRAG-Research's executor framework already implements this pattern — pipelines share the same queries, corpora, and ground truth through database-backed storage.

**Systematic Surveys on RAG**
- *"A Survey on Retrieval-Augmented Generation"* — Gao et al. (arXiv:2507.18910, 2025). Comprehensive taxonomy of RAG paradigms (Naive, Advanced, Modular, Agentic). Identifies evaluation standardization as a key open challenge.
- *"Benchmarking Retrieval-Augmented Generation: A Survey"* — Wang et al. (arXiv:2506.00054, 2025). Focuses on RAG benchmarking methodology. Highlights the lack of controlled experimentation environments where only one variable changes at a time.

### 1.2 Framework Papers from Major AI Conferences

| Paper | Venue | Year | Key Contribution | Citation Count* |
|-------|-------|------|-----------------|----------------|
| **FlashRAG** (Jin et al.) | WWW 2025 | 2025 | Modular RAG toolkit with 16 methods and 38 datasets. Extensive benchmark comparing RAG methods under unified preprocessing. | ~200+ |
| **BERGEN** (Rau et al.) | EMNLP 2024 Findings | 2024 | RAG benchmarking library. Key finding: preprocessing choices (chunking, embedding) impact performance more than algorithm choice. | ~50+ |
| **RAGAS** (Es et al.) | EACL 2024 Demo | 2024 | Reference-free RAG evaluation framework. Four core metrics: faithfulness, answer relevancy, context precision, context recall. | ~500+ |
| **RAGChecker** (Ye et al.) | NeurIPS 2024 D&B | 2024 | Claim-level entailment checking for fine-grained RAG diagnostics. Breaks answers into claims and traces to source chunks. | ~100+ |
| **ARES** (Saad-Falcon et al.) | NAACL 2024 | 2024 | Automated RAG evaluation with prediction-powered inference (PPI). Statistically outperforms RAGAS on correlation with human judgments. | ~100+ |
| **AutoRAG** (Kim et al.) | arXiv 2024 | 2024 | AutoML-style RAG optimization. Searches pipeline configurations to find optimal combinations. | ~50+ |
| **RGB** (Chen et al.) | AAAI 2024 | 2024 | Benchmarks 4 fundamental RAG abilities: noise robustness, negative rejection, information integration, counterfactual robustness. | ~200+ |
| **RAGBench** (Friel & Sanchez) | 2024 | 2024 | 100k industry-domain examples with TRACe framework for fine-grained annotation. Covers 5 industry domains. | ~30+ |

*\* Approximate citation counts as of early 2025.*

**Detailed Analysis of Key Competitors:**

**FlashRAG** is the closest competitor in scope. It provides:
- 16 RAG methods (Naive, Self-RAG, CRAG, IRCoT, Self-Ask, ITER-RETGEN, etc.)
- 38 benchmark datasets
- Modular component architecture (retriever, reranker, refiner, generator)
- File-based experiment configuration

*Limitations:* No database persistence, no multi-hop ground truth semantics, no multimodal support, no pre-computed embedding distribution. All results are ephemeral (file-based).

**BERGEN** focuses specifically on controlled benchmarking:
- Finds that preprocessing > algorithm in most cases
- Provides standardized evaluation harness
- Limited to text-only, evaluation-only (no production deployment)

*Limitations:* No pipeline composition, no multimodal, no persistence layer.

**RAGAS** dominates the evaluation-only space:
- Reference-free metrics (no ground truth needed for some metrics)
- LLM-as-judge approach for faithfulness and relevancy
- Strong community adoption and integration ecosystem

*Limitations:* Evaluation only — not an experimentation framework. Does not manage data, pipelines, or execution.

### 1.3 Competitive Landscape

**Full-Stack RAG Frameworks:**

| Tool | Type | Strengths | Weaknesses vs AutoRAG-Research |
|------|------|-----------|-------------------------------|
| **LangChain** | SDK | Massive ecosystem, 500+ integrations | No built-in evaluation, no experiment tracking, no DB persistence for results |
| **LlamaIndex** | SDK | Strong data connectors, tree-based indexing | No standardized evaluation, no controlled comparison infrastructure |
| **Haystack** | SDK | Pipeline-first design, production focus | Limited evaluation, no ground truth management |
| **DSPy** | Compiler | Automatic prompt optimization | Research-focused, steep learning curve, no persistent evaluation |
| **DeepEval** | Evaluation | 14+ metrics, LLM-based evaluation | Evaluation only, no pipeline management |
| **TruLens** | Evaluation | Feedback functions, tracing | Evaluation overlay, not a full framework |
| **MLflow** | Tracking | Experiment tracking, model registry | General ML — no RAG-specific features, no retrieval evaluation |

**Key Insight:** No existing tool combines (1) database-backed data/result persistence, (2) pipeline execution, (3) evaluation, and (4) reproducibility guarantees in a single framework. AutoRAG-Research occupies a unique position at this intersection.

---

## Part 2: Paper Positioning & Structure

### 2.1 Positioning Statement

> **AutoRAG-Research is a database-backed platform for reproducible RAG experimentation** — distinct from benchmarking-only tools (FlashRAG, BERGEN) that lack persistence, evaluation-only tools (RAGAS, RAGChecker) that lack pipeline management, and SDK toolkits (LangChain, LlamaIndex) that lack experimental rigor.

The core argument: **Reproducibility in RAG requires persistent infrastructure.** File-based approaches lose provenance when configurations change. Database-backed storage ensures that every embedding, retrieval result, generation output, and evaluation score is permanently linked to its exact configuration — enabling true apples-to-apples comparisons.

### 2.2 Target Venue

**Primary: ACL 2025 System Demonstrations** (4 pages + references)
- Best fit for software tool papers with working systems
- Emphasis on novelty of the system, not just results
- Allows demonstration of the tool's capabilities
- Deadline: Rolling / TBD for ACL 2025

**Alternative Venues:**
- **EMNLP 2025 Demo Track** — Similar format, slightly later deadline
- **NeurIPS 2025 Datasets & Benchmarks** — If we emphasize the benchmark/dataset contribution
- **SIGIR 2025 Resource Track** — Strong IR community, retrieval focus

### 2.3 Proposed Title & Outline

**Title:** *"AutoRAG-Research: A Database-Backed Platform for Reproducible RAG Experimentation"*

**Format:** ACL System Demonstrations (4 pages + unlimited references)

---

#### Section 1: Abstract (~150 words)

- **Problem:** RAG research suffers from a reproducibility gap. Studies comparing RAG methods confound algorithmic differences with infrastructure differences (embedding models, chunking, retrieval backends), making it impossible to attribute performance gains.
- **Solution:** AutoRAG-Research, a database-backed framework that ensures reproducibility by persisting all experimental artifacts (embeddings, retrieval results, generation outputs, evaluations) in PostgreSQL with pgvector. Features dynamic schema configuration, composable pipeline architecture, and pre-computed embedding distribution.
- **Key Results:** Demonstrate zero-variance reproduction across machines, show that pipeline rankings change when controlling for preprocessing, and present the first unified text+vision RAG evaluation.

#### Section 2: Introduction (~0.75 page)

- Motivate the reproducibility problem with concrete examples from ReproRAG and BERGEN findings.
- Position AutoRAG-Research in the landscape (Figure 1: positioning diagram).
- Contribution list:
  1. Database-backed persistence layer for complete experimental provenance
  2. Dynamic schema factory supporting arbitrary embedding dimensions
  3. Composition-based generation pipeline architecture
  4. AND/OR multi-hop ground truth with group-aware nDCG
  5. Unified text+vision RAG evaluation pipeline
  6. Pre-computed embeddings on HuggingFace Hub for instant reproducibility

#### Section 3: System Architecture (~1 page)

- **Figure 2:** Layered architecture diagram showing:
  ```
  CLI / Configuration (YAML)
       ↓
  Executor / Evaluator
       ↓
  Pipeline Layer (Retrieval / Generation)
       ↓
  Service Layer (Business Logic)
       ↓
  Unit of Work (Transaction Management)
       ↓
  Repository Layer (GenericRepository[T])
       ↓
  PostgreSQL + pgvector + VectorChord
  ```
- **Schema Factory:** `create_schema(embedding_dim)` generates ORM classes at runtime, enabling the same codebase to work with any embedding model dimension.
- **Pipeline Composition:** Generation pipelines compose with retrieval pipelines. Any generation strategy (BasicRAG, IRCoT, ET2RAG, MAINRAG, VisRAG) can be combined with any retrieval method (Vector, BM25, HyDE, Hybrid).
- **Data Persistence:** Every intermediate result is stored: embeddings → retrieval scores → generation text → evaluation metrics. Full provenance chain from raw data to final score.
- **Plugin System:** External packages register pipelines and metrics via Python entry points. CLI scaffolding auto-generates commands from plugin configurations.

#### Section 4: Key Features (~1 page)

- **Pre-computed Embeddings:** Upload to HuggingFace Hub. Users download and immediately run pipelines — no expensive embedding computation required. This is critical for reproducibility (eliminates embedding variance).
- **Multi-hop Ground Truth:** AND/OR semantics via `RetrievalGT` API:
  ```python
  gt = (TextId(1) | TextId(2)) & TextId(4)  # (A OR B) AND C
  ```
  Group-aware nDCG correctly evaluates multi-hop queries where multiple evidence pieces must be retrieved.
- **Multimodal RAG:** Unified pipeline for text and vision:
  - `ImageChunk` model with single/multi-vector embeddings
  - ColPali embeddings (5 model variants)
  - VisRAG generation pipeline processes document images via VLMs
  - Supports ViDoRe v1/v2/v3 benchmarks
- **Comprehensive Reranking:** 15+ reranker implementations (SentenceTransformer, ColBERT, FlashRank, MonoT5, TART, FlagEmbedding, Jina, Cohere, VoyageAI, MixedBreadAI, OpenVINO, etc.)
- **Feature Comparison Table** (Table 1 — see Section 4.4 below)

#### Section 5: Experiments (~0.75 page)

- Experiment 1: Retrieval pipeline comparison on BEIR under identical preprocessing
- Experiment 2: Generation pipeline comparison on RAGBench with controlled retrieval
- Experiment 3: Reproducibility audit — identical results across runs and machines
- Experiment 4: Multi-hop evaluation — group-aware nDCG vs flat nDCG changes rankings
- Experiment 5: Multimodal RAG on ViDoRe/VisRAG benchmarks

#### Section 6: Related Work (~0.5 page)

- RAG frameworks (FlashRAG, BERGEN, AutoRAG)
- Evaluation tools (RAGAS, RAGChecker, ARES)
- Reproducibility in NLP (ReproRAG, TREC RAG Track, ML reproducibility initiatives)

#### Section 7: Conclusion

- Summary of contributions
- Future work: LLM-as-judge metrics, agentic RAG pipelines, public leaderboard

---

## Part 3: SWOT Analysis

### Strengths (to emphasize in the paper)

| # | Strength | Evidence in Codebase | Why It Matters |
|---|----------|---------------------|----------------|
| S1 | **Database-backed persistence** (PostgreSQL + pgvector + VectorChord) | `orm/models/`, `orm/repository/`, `orm/uow/`, `orm/service/` — full layered architecture | No competitor persists all experimental artifacts in a database. Enables complete provenance, cross-experiment queries, and guaranteed reproducibility. |
| S2 | **Dynamic schema factory** with runtime-configurable embedding dimensions | `orm/schema_factory.py` — `create_schema(embedding_dim, primary_key_type)` with `@lru_cache(maxsize=16)` | Same codebase supports any embedding model without code changes. Competitors hardcode dimensions or require config migration. |
| S3 | **Composition-based generation** (any generation × any retrieval) | `pipelines/generation/` — BasicRAG, IRCoT, ET2RAG, MAINRAG, VisRAG all compose with any retrieval pipeline via `BaseGenerationPipeline` | Enables factorial experiment designs. E.g., test 5 generation strategies × 5 retrieval methods = 25 configurations with minimal code. |
| S4 | **Pre-computed embeddings on HuggingFace Hub** | `data/registry.py` — `@register_ingestor(hf_repo=...)`, ingestors for BEIR, RAGBench, ViDoRe, VisRAG, MTEB, etc. | Eliminates the #1 source of variance (embedding computation). Users get identical embeddings, ensuring reproducible baselines. |
| S5 | **AND/OR multi-hop ground truth** with group-aware nDCG | `orm/models/retrieval_gt.py` — `RetrievalGT`, `OrGroup`, `AndChain`, `TextId`, `ImageId`; `evaluation/metrics/retrieval.py` — `retrieval_ndcg()` with group semantics | Unique capability. Multi-hop queries require multiple evidence pieces (AND) with alternatives (OR). Standard nDCG ignores this structure. |
| S6 | **Unified text + vision RAG** | `orm/models/` — `ImageChunk`; `embeddings/` — `ColPaliEmbeddings` (5 variants), `BiPaliEmbeddings`; `pipelines/generation/visrag_gen.py`; ingestors for ViDoRe v1/v2/v3, VisRAG | FlashRAG, BERGEN, and AutoRAG are text-only. Multimodal RAG is an underserved and growing area. |
| S7 | **15+ reranker implementations** | `rerankers/` — SentenceTransformer, ColBERT, FlashRank, MonoT5, TART, FlagEmbedding, FlagEmbeddingLLM, KoReranker, UPR, Jina, Cohere, VoyageAI, MixedBreadAI, OpenVINO | Most comprehensive reranker collection. FlashRAG has ~3, BERGEN has ~5. |
| S8 | **Plugin system with CLI scaffolding** | `plugin_registry.py`, `data/registry.py` — entry-point-based discovery, `cli/commands/plugin.py` | Extensible without forking. External packages register via `pyproject.toml` entry points. |

### Weaknesses (gaps to address before paper)

| # | Weakness | Impact on Paper | Mitigation Strategy |
|---|----------|----------------|---------------------|
| W1 | **No LLM-as-judge / reference-free metrics** | RAGAS has faithfulness, answer relevancy without ground truth. We require reference answers for generation evaluation. | **Priority 1:** Implement LLM-as-judge metrics (faithfulness, answer relevancy, context relevance). |
| W2 | **No chunking/parsing pipeline** | Users must provide pre-chunked data. Competitors handle end-to-end from raw documents. | Frame as a design choice: focus on evaluation, not preprocessing. Preprocessing variance is the problem (per BERGEN/ReproRAG). |
| W3 | **Limited agentic RAG** | Only IRCoT for multi-step reasoning. No Self-RAG, CRAG, or adaptive retrieval. FlashRAG has more methods. | **Priority 2:** Implement 1-2 additional agentic pipelines (Self-RAG or CRAG). |
| W4 | **PostgreSQL dependency** increases setup friction | Competitors are pip-install-and-run. We require Docker/PostgreSQL. | Docker Compose is included. Pre-computed HF datasets reduce need for local DB during evaluation. Emphasize that DB is the feature, not a limitation. |
| W5 | **No statistical significance testing** | Cannot claim methods are significantly different. Reviewers will ask for confidence intervals. | **Priority 1:** Add bootstrap confidence intervals for metric comparisons. |
| W6 | **No public hosted leaderboard** | Competitors (e.g., MTEB) attract community contributions through leaderboards. | **Priority 1:** Create HuggingFace Space with interactive results dashboard. |
| W7 | **Fewer RAG methods than FlashRAG** | 5 retrieval + 5 generation vs FlashRAG's 16. May appear less comprehensive. | Emphasize composition (5×5=25 combinations) and quality over quantity. Prioritize adding 1-2 high-impact methods. |

### Opportunities

1. **Reproducibility narrative is timely.** ReproRAG (2025) and TREC RAG Track (2024) have put RAG reproducibility in the spotlight. A tool paper that directly addresses this need is well-positioned.
2. **Multimodal RAG is underserved.** FlashRAG, BERGEN, and AutoRAG are all text-only. AutoRAG-Research's unified text+vision pipeline is a differentiator that no competitor matches.
3. **Adding LLM-as-judge fills the biggest evaluation gap.** One feature addition would neutralize our main weakness versus RAGAS.
4. **BEIR/RAGBench are widely used benchmarks.** Supporting them out-of-the-box with pre-computed embeddings lowers the barrier for adoption and comparison.
5. **ACL Demo Track values working systems.** Our fully implemented system (not a prototype) is a strong fit.

### Threats

1. **FlashRAG has more methods.** If reviewers prioritize method count, we need to emphasize the composition advantage and unique features.
2. **RAGAS has strong mindshare.** The name recognition for evaluation could overshadow our evaluation capabilities unless we clearly position as a *platform* rather than an evaluation tool.
3. **PostgreSQL requirement creates friction.** In a world of `pip install` tools, requiring Docker/PostgreSQL may deter some users. Must be framed as an intentional design choice for reproducibility.
4. **Fast-moving field.** New RAG methods and frameworks appear monthly. Paper must be submitted before the landscape shifts further.

---

## Part 4: Features to Add & Experiments

### 4.1 Must-Add Features (Priority 1)

These features should be implemented **before paper submission** as they address critical reviewer concerns.

#### Feature 1: LLM-as-Judge Generation Metrics

**What:** Reference-free evaluation metrics using LLM-based judgment.

**Metrics to implement:**
- **Faithfulness** — Does the generated answer contain only information supported by the retrieved context? (Claim decomposition + entailment checking)
- **Answer Relevancy** — Does the generated answer address the original query? (LLM scoring)
- **Context Relevance** — Are the retrieved passages relevant to the query? (LLM scoring)

**Implementation approach:**
- Add new metric type `LLM_JUDGE` alongside existing `RETRIEVAL` and `GENERATION` types.
- Use LangChain's LLM abstraction for model-agnostic evaluation.
- Support configurable judge models (GPT-4, Claude, open-source).
- Store judge outputs in `EvaluationResult` with metadata containing reasoning.

**Why critical:** RAGAS's main advantage over us. Reviewers familiar with RAGAS will expect reference-free evaluation. Without this, our generation evaluation story is incomplete.

#### Feature 2: Statistical Significance Testing

**What:** Bootstrap confidence intervals and statistical tests for metric comparisons.

**Implementation approach:**
- Add `bootstrap_ci(scores, n_bootstrap=1000, alpha=0.05)` utility function.
- Add paired bootstrap test for comparing two pipelines.
- Integrate into executor reporting — every metric comparison includes 95% CI.
- Store CIs alongside aggregate metrics in `Summary` table.

**Why critical:** Any experimental paper claiming "method A outperforms method B" must include statistical significance. Reviewers at ACL/EMNLP will require this.

#### Feature 3: Hosted Public Leaderboard

**What:** Interactive HuggingFace Space showing pipeline performance across datasets.

**Implementation approach:**
- Export `Summary` table data to standardized JSON format.
- Build Gradio dashboard (we already have `reporting` extra with Gradio).
- Deploy on HuggingFace Spaces.
- Allow community submissions with reproducibility verification.

**Why critical:** Demonstrates community value beyond the paper. Leaderboards attract users and citations.

### 4.2 Should-Add Features (Priority 2)

These features would strengthen the paper but are not blocking.

#### Feature 4: Self-RAG or CRAG Pipeline

**What:** At least one modern agentic RAG pipeline beyond IRCoT.

**Candidates:**
- **Self-RAG** (Asai et al., 2023) — Self-reflective retrieval with critique tokens
- **CRAG** (Yan et al., 2024) — Corrective RAG with confidence-based retrieval adaptation

**Why valuable:** Closes the method gap with FlashRAG. Shows the framework can support modern agentic patterns.

#### Feature 5: Per-Component Latency Tracking

**What:** Fine-grained timing for each pipeline stage (embedding, retrieval, reranking, generation).

**Implementation:** Extend `execution_time` tracking to report per-stage latencies in `result_metadata`.

**Why valuable:** Enables cost-performance trade-off analysis. Reviewers appreciate efficiency analysis.

#### Feature 6: Cross-Framework Reproducibility Comparison

**What:** Run the same experiment in AutoRAG-Research and FlashRAG, show result differences.

**Why valuable:** Directly demonstrates the reproducibility problem and our solution. High-impact experiment.

### 4.3 Experiment Designs

#### Experiment 1: Retrieval Pipeline Comparison

**Objective:** Demonstrate fair comparison under identical preprocessing.

| Parameter | Value |
|-----------|-------|
| **Datasets** | BEIR subset: SciFact, NFCorpus, FiQA, HotpotQA |
| **Pipelines** | VectorSearch, BM25, HyDE, HybridRRF, HybridCC |
| **Embedding Models** | all-MiniLM-L6-v2 (384d), BGE-base (768d), E5-large (1024d) |
| **Rerankers** | None, ColBERT, FlashRank, Cohere |
| **Metrics** | nDCG@10, Recall@10, Precision@10, F1 |
| **Key Hypothesis** | Rankings change when controlling for preprocessing (confirms BERGEN/ReproRAG findings). DB persistence ensures exact reproduction. |

**Expected Outcome:** Show that pipeline rankings are sensitive to embedding model choice, but our framework makes this explicit and reproducible.

#### Experiment 2: Generation Pipeline Comparison

**Objective:** Evaluate generation strategies with controlled retrieval.

| Parameter | Value |
|-----------|-------|
| **Datasets** | RAGBench (3-4 domains: e.g., tech, finance, biomedical) |
| **Retrieval** | Fixed: VectorSearch + ColBERT reranker (top-5) |
| **Generation Pipelines** | BasicRAG, IRCoT, ET2RAG, MAINRAG |
| **LLMs** | GPT-4o-mini, Llama-3.1-8B, Gemma-2-9B |
| **Metrics** | ROUGE-L, BERTScore, SemScore, Faithfulness (LLM-judge), Answer Relevancy (LLM-judge) |
| **Key Hypothesis** | Composition architecture enables fair generation comparison by fixing retrieval quality. |

**Expected Outcome:** Clear generation pipeline rankings that would be impossible without controlled retrieval. IRCoT and MAINRAG expected to outperform BasicRAG on complex queries.

#### Experiment 3: Reproducibility Audit

**Objective:** Demonstrate zero-variance reproduction.

| Parameter | Value |
|-----------|-------|
| **Setup** | Same as Experiment 1, but run 3× on 2 different machines |
| **Machines** | Machine A (Linux, CUDA), Machine B (macOS, CPU) |
| **Comparison** | Bit-for-bit comparison of retrieval results and scores |
| **Key Hypothesis** | Pre-computed embeddings + DB persistence → identical results across runs and machines. |

**Expected Outcome:** Zero variance in retrieval results. Generation results may vary slightly due to LLM non-determinism (document this). Statistical metrics (nDCG, ROUGE) should be identical for retrieval evaluation.

#### Experiment 4: Multi-Hop Evaluation

**Objective:** Show that group-aware nDCG changes pipeline rankings vs flat nDCG.

| Parameter | Value |
|-----------|-------|
| **Datasets** | HotpotQA (multi-hop subset), BRIGHT |
| **Ground Truth** | AND/OR annotated using `RetrievalGT` API |
| **Pipelines** | VectorSearch, HyDE, IRCoT-retrieval |
| **Metrics** | Flat nDCG@10, Group-aware nDCG@10, Recall with group semantics |
| **Key Hypothesis** | Pipelines that retrieve both evidence pieces (AND) rank differently under group-aware nDCG than under flat nDCG. |

**Expected Outcome:** IRCoT-style iterative retrieval ranks higher under group-aware nDCG because it's more likely to retrieve all required evidence pieces. Standard nDCG misses this distinction.

#### Experiment 5: Multimodal RAG Evaluation

**Objective:** Demonstrate unified text+vision RAG.

| Parameter | Value |
|-----------|-------|
| **Datasets** | ViDoRe v1 (subset), VisRAG-Ret-Test |
| **Retrieval** | ColPali (multi-vector), BiPali (single-vector) |
| **Generation** | VisRAG (VLM-based) |
| **Metrics** | nDCG@5, Recall@5 (retrieval); Answer accuracy (generation) |
| **Key Hypothesis** | Unified pipeline handles visual documents without separate OCR/text-extraction stage. |

**Expected Outcome:** Competitive results on visual QA benchmarks using end-to-end vision pipeline. No competitor framework supports this evaluation.

### 4.4 Feature Comparison Table

*Table for inclusion in the paper (Section 4):*

| Feature | AutoRAG-Research | FlashRAG | BERGEN | AutoRAG | RAGAS |
|---------|:---:|:---:|:---:|:---:|:---:|
| DB-backed persistence | **Yes** | No | No | No | No |
| Pre-computed embeddings (HF Hub) | **Yes** | No | No | No | N/A |
| Multimodal (text + vision) | **Yes** | No | No | No | No |
| Multi-hop ground truth (AND/OR) | **Yes** | No | No | No | No |
| Composable generation × retrieval | **Yes** | Partial | No | No | N/A |
| Plugin system (entry points) | **Yes** | No | No | No | No |
| Reference-free metrics (LLM judge) | Planned* | No | No | No | **Yes** |
| Statistical significance testing | Planned* | No | No | No | No |
| Reranker implementations | **15+** | ~3 | ~5 | Varies | N/A |
| RAG methods (retrieval + generation) | 5 + 5 | **16** | 5 | 7 | N/A |
| Benchmark datasets supported | 8+ | **38** | 6 | 4 | N/A |
| Public leaderboard | Planned* | No | No | No | No |

*\* "Planned" features to be implemented before paper submission.*

---

## Part 5: Timeline

| Phase | Duration | Key Tasks | Deliverables |
|-------|----------|-----------|-------------|
| **Feature Development** | 3-4 weeks | LLM-as-judge metrics, statistical significance testing, per-component latency tracking | New metric implementations, bootstrap CI utility, updated executor |
| **Dataset Preparation** | 1-2 weeks | Pre-compute embeddings for all experiment datasets, upload to HuggingFace Hub | HF Hub datasets with embeddings for BEIR, RAGBench, HotpotQA, ViDoRe |
| **Experiment Execution** | 2-3 weeks | Run all 5 experiments, iterate on configurations, collect results | Raw results in database, exported tables and figures |
| **Paper Writing** | 2-3 weeks | Write all sections, create architecture diagrams, format tables, revise | Draft paper (4 pages + references, ACL Demo format) |
| **Review Buffer** | 1-2 weeks | Internal review, address feedback, final polishing | Camera-ready paper |
| **Total** | **9-14 weeks** | | Submitted paper + public leaderboard |

**Milestones:**
1. **Week 4:** LLM-as-judge metrics merged, statistical testing ready
2. **Week 6:** All datasets pre-computed and uploaded to HF Hub
3. **Week 9:** All experiments complete, results tables finalized
4. **Week 12:** Paper draft complete, under internal review
5. **Week 14:** Paper submitted

---

## References

1. Jin, J., et al. "FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research." *WWW 2025*. [arXiv:2405.13576](https://arxiv.org/abs/2405.13576)

2. Rau, D., et al. "BERGEN: A Benchmarking Library for Retrieval-Augmented Generation." *EMNLP 2024 Findings*. [arXiv:2407.01102](https://arxiv.org/abs/2407.01102)

3. Es, S., et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation." *EACL 2024 System Demonstrations*. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)

4. Ye, D., et al. "RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation." *NeurIPS 2024 Datasets and Benchmarks*. [arXiv:2408.08067](https://arxiv.org/abs/2408.08067)

5. Saad-Falcon, J., et al. "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems." *NAACL 2024*. [arXiv:2311.09476](https://arxiv.org/abs/2311.09476)

6. Kim, D., et al. "AutoRAG: Automated Framework for Optimization of Retrieval Augmented Generation Pipeline." *arXiv 2024*. [arXiv:2410.20878](https://arxiv.org/abs/2410.20878)

7. Chen, J., et al. "Benchmarking Large Language Models in Retrieval-Augmented Generation." *AAAI 2024*. [arXiv:2309.01431](https://arxiv.org/abs/2309.01431)

8. Friel, R. & Sanchez, A. "RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems." *2024*. [arXiv:2407.11005](https://arxiv.org/abs/2407.11005)

9. Meng, Y., et al. "ReproRAG: Benchmarking Reproducibility in Retrieval-Augmented Generation." *2025*. [arXiv:2509.18869](https://arxiv.org/abs/2509.18869)

10. Gao, Y., et al. "Retrieval-Augmented Generation for Large Language Models: A Survey." *2025*. [arXiv:2507.18910](https://arxiv.org/abs/2507.18910)

11. Wang, S., et al. "Benchmarking Retrieval-Augmented Generation: A Survey." *2025*. [arXiv:2506.00054](https://arxiv.org/abs/2506.00054)

12. TREC 2024 RAG Track. *NIST*. [trec.nist.gov](https://trec.nist.gov/pubs/trec33/papers/Overview-rag.pdf)

13. Asai, A., et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." *ICLR 2024*. [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)

14. Yan, S., et al. "Corrective Retrieval Augmented Generation." *2024*. [arXiv:2401.15884](https://arxiv.org/abs/2401.15884)
