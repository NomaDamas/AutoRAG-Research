# Paper-Fidelity Audit Report — dev-branch pipelines (2026-06-10)

Scope: all retrieval/generation pipelines **added** on `dev` vs `origin/main`.
Verdict classes: CORE-MISMATCH (fix required), ALLOWED-1 (model brand/size), ALLOWED-2 (AutoRAG-Research
architecture workaround), ALLOWED-3 (info unavailable via abstracted LLM provider API / training out of scope),
COSMETIC.

---

## 1. adaptive_rag.py — Adaptive-RAG (Jeong et al., NAACL 2024, arXiv:2403.14403)

Paper core: 3-way complexity routing (A=no retrieval, B=single-step, C=multi-step). Route C is the
**multi-step approach = IRCoT** (interleaved CoT reasoning + retrieval, per-step intermediate reasoning,
termination when the reasoning chain derives the answer).

| Finding | Verdict |
|---|---|
| LLM-as-classifier instead of trained T5 classifier | ALLOWED-3 (no trained classifier weights via provider API) |
| Unknown classifier output defaults to "moderate" | ALLOWED-3 adaptation |
| Multi route = follow-up-query generation loop, no interleaved CoT/intermediate answers | **CORE-MISMATCH** |
| Multi route merges all retrievals by score and globally truncates (no ordered reasoning state) | **CORE-MISMATCH** |
| Multi route terminates on a STOP query signal instead of answer-derivation in the reasoning trace | **CORE-MISMATCH** |
| gpt-4o-mini instead of FLAN-T5/GPT-3.5 | ALLOWED-1 |

**Fix**: replace the multi route with an IRCoT-style loop (mirroring the repo's existing faithful
`IRCoTGenerationPipeline`): initial retrieval → iterate {generate next CoT thought from (query, ordered
paragraphs, thought history); terminate on answer marker; retrieve with the thought; append paragraphs FIFO} →
final QA over accumulated paragraphs + CoT.

## 2. deep_rag.py — DeepRAG (Guan et al., arXiv:2502.01142)

Paper core: MDP over subqueries. Each step: termination decision σ∈{continue, terminate}; if continue,
generate subquery q_t, atomic decision δ∈{retrieve, parametric}, then ALWAYS generate intermediate answer r_t
(from retrieved docs if δ=retrieve, parametrically otherwise). State = ordered (q_i, r_i) trajectory.

| Finding | Verdict |
|---|---|
| BTS data synthesis / imitation learning / Chain-of-Calibration training omitted | ALLOWED-3 |
| Controller actions retrieve/reason/answer don't encode subquery-level σ/δ decisions | **CORE-MISMATCH** |
| Retrieve branch never generates an intermediate answer for the subquery | **CORE-MISMATCH** |
| Parametric steps are free-form "reason" text, not subquery + intermediate answer | **CORE-MISMATCH** |
| State is flat lists, not ordered (q_i, r_i) trajectory | **CORE-MISMATCH** |
| Exact "Follow up:/Intermediate answer:" narrative wording | COSMETIC (structured-equivalent accepted) |

**Fix**: restructure the loop: each step the controller emits `<retrieve>subquery</retrieve>` (δ=retrieve),
`<parametric>subquery</parametric>` (δ=parametric), or `<answer>final</answer>` (σ=terminate). After
retrieve/parametric, a second LLM call produces the intermediate answer for that subquery (with retrieved docs
when δ=retrieve). State kept and prompted as an ordered trajectory of
{subquery, decision, retrieved docs, intermediate answer}. Fallback final prompt retained (untrained model may
never terminate; eval harness must produce an answer) but operates on the trajectory — documented deviation.

## 3. dynamic_rag.py + rerankers/dynamic_rag.py — DynamicRAG (Sun et al., arXiv:2505.07233)

Paper core (inference): an **LLM reranker agent** receives query + top-N docs and directly **generates an
ordered subset of document IDs** — order AND k decided jointly by the LLM; may output `None` (0 docs).
Generator answers from exactly that subset; zero-doc case answers from model knowledge.

| Finding | Verdict |
|---|---|
| SFT + DPO/RL training with generator feedback omitted | ALLOWED-3 |
| Reranker is a heuristic score-gap/min-score cut over an arbitrary scorer, no LLM call | **CORE-MISMATCH** |
| Dynamic k via score-drop threshold instead of generated subset length | **CORE-MISMATCH** |
| Cannot return 0 documents (min 1 forced) | **CORE-MISMATCH** |
| No-base fallback returns original-order prefix (not reranking at all) | **CORE-MISMATCH** |
| Generator prompt forbids zero-evidence answering | **CORE-MISMATCH** |
| RerankResult(score) adapter shape | ALLOWED-2 (scores synthetic from output rank) |

**Fix**: `DynamicRAGReranker` becomes a prompted LLM listwise reranker (paper Appendix C.3 Table 11
semantics): numbered docs + query → ordered ID list or `None`; parsed list length = dynamic k (0 allowed);
requested `top_k` acts only as a safety cap. Heuristic cut policy and base_reranker wrapping removed. Pipeline
injects its generator LLM into the reranker when unset (paper §4.5.1 allows shared parameters) and the
generator prompt handles the zero-document case by answering from model knowledge (Table 12).

## 4. hybrid_deep_searcher.py — HybridDeepSearcher (Ko et al., ICLR 2026, arXiv:2508.19113)

Paper core: alternate <think> reasoning with parallel query blocks
`<|begin search queries|> q1; q2; ... <|end search queries|>`; results returned query-labelled inside
`<|begin search results|> qi: si ... <|end search results|>` appended to the rolling context; budgets = max
turns (MT) AND max search calls (MC) with fan-out truncation to remaining MC; must answer at budget exhaustion.

| Finding | Verdict |
|---|---|
| Qwen3-8B HDS-QA fine-tuning omitted | ALLOWED-3 |
| Retrieval pipeline instead of web search + external summarizer | ALLOWED-2 (Appendix D.2 no-summarizer ablation) |
| `<queries>`/`<answer>` tags instead of the paper's search-query token protocol | **CORE-MISMATCH** |
| Evidence merged globally by doc score, losing query↔result pairing and result-block serialization | **CORE-MISMATCH** |
| No max-search-calls budget / fan-out truncation / call-budget termination | **CORE-MISMATCH** |
| Final-prompt fallback at budget exhaustion | FAITHFUL (paper requires answering once budgets exhaust) |

**Fix**: adopt the paper protocol (parse `<|begin search queries|>…<|end search queries|>` with
semicolon/newline splitting, keep `<think>` text in the rolling trace, paper answer marker with tolerant
fallbacks), serialize per-query labelled result blocks into the rolling context, add `max_search_calls` with
remaining-budget truncation and forced finalization.

## 5. search_r1.py — Search-R1 (Jin et al., COLM 2025, arXiv:2503.09516)

Paper core (Algorithm 1 rollout): template mandates <think> before every action; actions <search>q</search> /
<answer>a</answer>; retrieved content appended to the rolling sequence as <information>…</information>;
malformed action → append "My action is not correct. Let me rethink." and continue; action budget B; at budget
exhaustion return rollout (no extra out-of-budget LLM call).

| Finding | Verdict |
|---|---|
| PPO/GRPO training, retrieved-token masking | ALLOWED-3 |
| No <think>/<information> rolling rollout; synthetic scratchpad rebuild | **CORE-MISMATCH** |
| Parser prefers <answer> globally instead of earliest completed action | **CORE-MISMATCH** |
| Malformed/untagged output treated as final answer instead of rethink path | **CORE-MISMATCH** |
| Extra out-of-budget fallback final LLM call by default | **CORE-MISMATCH** |
| max_searches=3, k=5 vs B=4, top-3 | ALLOWED-1/config |

**Fix**: maintain rolling rollout string with environment-inserted `<information>` blocks; paper-style step
template requiring `<think>`; dispatch on earliest completed action tag; malformed → rethink message,
budget-counted; `fallback_to_final_prompt` now defaults to **False** (paper mode) and remains available as an
explicitly-documented non-paper compatibility option.

## 6. interact_rag.py — Interact-RAG (Hui et al., ICLR 2026, arXiv:2510.27566)

Paper core: Corpus Interaction Engine with real primitives — semantic_search (dense), exact_search
(sparse/BM25/FTS), weighted_fusion (normalize both strategies' top-20, weighted-sum), entity_match (FTS term
matching, most query-related snippets), include/exclude_docs, adjust_scale; agent emits reasoning + a suite of
concurrent actions per step; engine returns one consolidated response.

| Finding | Verdict |
|---|---|
| SFT/RL-trained end-to-end agent omitted (prompted LLM instead) | ALLOWED-3 |
| Global-Planner/Adaptive-Reasoner/Executor 3-module training-free workflow absent | FIXED — default workflow now runs one planner call, per-iteration reasoner directives, and executor action emission with plan/directives metadata. |
| exact_search prompt-simulated by prefixing "exact:" into the same retriever | **CORE-MISMATCH** (repo has BM25 — not an architecture limit) |
| weighted_fusion only embeds weights into a query string; no dual retrieval + score fusion | **CORE-MISMATCH** |
| entity_match prompt-simulated via "entity:" prefix | **CORE-MISMATCH** |
| semantic_search bound to arbitrary configured retriever (config wires hybrid_rrf) | **CORE-MISMATCH** (config-level: wire a dense retriever) |
| Single action per step, no concurrent action suite | **CORE-MISMATCH** |
| include/exclude/adjust_scale semantics | COSMETIC (safety-refined, faithful) |

**Fix**: add an optional second `exact_retrieval_pipeline` (BM25) — exact_search and entity_match route
through it; weighted_fusion runs both strategies, normalizes scores, and fuses by weighted sum; multi-action
parsing executes every action emitted in a step (state actions first, then retrievals) with one consolidated
evidence update; default config wires vector search (semantic) + bm25 (exact). When no exact pipeline is
configured the previous prompt-simulated behavior remains as an explicitly-flagged degraded mode (ALLOWED-2
fallback only).

## 7. ras.py — RAS (Jiang et al., arXiv:2502.10996, latest revision)

Paper core (latest arXiv revision — standard text retrieval; the v1-only ThemeScope stage was dropped by the
authors, so plain retrieval here is FAITHFUL — decision recorded): initial planning BEFORE any retrieval with
[NO_RETRIEVAL] direct-answer branch; iterative [SUBQ] retrieval → text-to-triples structuring → evolving
query-specific graph G_Q with per-iteration (q_i, g_i) history → planner reassesses until [SUFFICIENT];
answering conditioned on graph + history + question.

| Finding | Verdict |
|---|---|
| Trained f_t2t triple extractor and GraphLLM/GNN encoding replaced by prompted LLM / text serialization | ALLOWED-3 |
| ThemeScope retrieval absent | FAITHFUL vs latest revision (decision: target latest paper version) |
| Unconditional initial retrieval before any planning; no [NO_RETRIEVAL] branch | **CORE-MISMATCH** |
| Flat triple list without per-subquery (q_i, g_i) grouping/graph state | **CORE-MISMATCH** |
| Final answer conditioned on raw passages + flat triples instead of graph + history + question | **CORE-MISMATCH** |
| Loop breaks on no-new-passages without a sufficiency decision | **CORE-MISMATCH** |
| Malformed plan silently retried as original-query retrieval | **CORE-MISMATCH** |

**Fix**: paper action protocol ([NO_RETRIEVAL] / [SUBQ] q / [SUFFICIENT]); plan first, retrieve only when
planned; maintain evolving triple graph plus ordered (subquery → triples) iteration history serialized into
planner and answer prompts; answer from graph + history + question (passages retained in metadata);
no-new-passages records a note but planning decides termination; malformed plan gets one bounded retry then is
treated as sufficiency (never silent original-query retrieval).

## 8. gqr_hybrid.py — Guided Query Refinement (Uzan et al., ICLR 2026, arXiv:2510.05038)

Paper core (Algorithm 1): candidate pool = union of both retrievers' top-K; p_j = softmax over each
retriever's native scores; z(0)=primary query embedding; per step t recompute p1(z(t)), form consensus
p_avg(t)=½[p1(z(t))+p2], minimize KL(p_avg(t)‖p1(z(t))) via gradient steps; final ranking by s1(z(T),d).

| Finding | Verdict |
|---|---|
| Fixed pre-loop target distribution instead of per-step p_avg(t) recomputation | **CORE-MISMATCH** |
| Z-score normalization of scores before softmax (paper: raw scores + temperature) | **CORE-MISMATCH** |
| Single-vector cosine only; no MaxSim/multi-vector native-scorer support | ALLOWED-2 (documented scope restriction; multi-vector primaries use the score-space fallback) |
| Floor score for union candidates missing from one retriever's list | ALLOWED-2 (no score-by-candidate API) |
| Score-space fallback when embeddings unavailable | ALLOWED-2 (documented degraded path) |
| candidate_pool_mode primary/union, fetch_k_multiplier | ALLOWED-2 (union default matches paper; primary = Appendix E variant) |

**Fix**: recompute the consensus target from the current primary distribution every optimization step (both
embedding and score-space paths); drop z-score normalization in favor of raw-score softmax with temperature
(missing-candidate floor retained); add behavioral tests for the per-step objective.

## 9. rerank.py — generic retrieve-then-rerank wrapper

No source paper (generic two-stage retrieval). Internal consistency verified (candidate fetch ≥ top_k,
content backfill, contract-ordered reranker output). **FAITHFUL — no changes.**
