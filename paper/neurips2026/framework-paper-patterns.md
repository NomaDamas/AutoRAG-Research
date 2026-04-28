# Framework-paper patterns used to restructure the AutoRAG-Research manuscript

This note summarizes writing patterns extracted from major framework / open-source system papers and records how they should shape the AutoRAG-Research NeurIPS manuscript.

## Source papers reviewed

| Paper | Venue/source | Relevant paper move |
| --- | --- | --- |
| DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines | ICLR 2024 | Defines a programming model, names its abstractions, then evaluates case studies that the abstractions make possible. |
| PyTorch: An Imperative Style, High-Performance Deep Learning Library | NeurIPS 2019 | Starts from a framework tension, usability vs. speed, then explains design principles and architecture before performance evidence. |
| RLlib: Abstractions for Distributed Reinforcement Learning | ICML 2018 | Centers the paper around composable abstractions, scalable primitives, code reuse, and implementation of many algorithms. |
| Dopamine: A Research Framework for Deep Reinforcement Learning | ICLR 2019 | Positions itself as a research framework for diverse goals, emphasizes compact reliable implementations, and adds a taxonomy of research needs. |
| Habitat: A Platform for Embodied AI Research | ICCV 2019 | Presents a stack diagram / layer decomposition, generic dataset support, high-performance platform pieces, and scientific experiments enabled by the platform. |
| CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms | JMLR 2022 | Uses transparency and compactness as framework virtues; benchmarks against reputable sources to prove usability does not sacrifice quality. |
| MMDetection: Open MMLab Detection Toolbox and Benchmark | arXiv technical report | Combines toolbox coverage, modular design, model zoo / artifact release, and a benchmark study over methods/components. |

## Reusable structure observed

1. **Problem framing as a field bottleneck, not just a local implementation gap.**
   Good framework papers begin by naming the repeated pain in the field: brittle prompts in DSPy, usability/speed tradeoffs in PyTorch, distributed irregularity in RLlib, fragmented simulators/tasks in Habitat, or opaque RL implementations in CleanRL/Dopamine.

2. **Explicit design requirements before implementation details.**
   The strongest papers do not jump directly into files/classes. They define requirements such as composability, controllability, scalability, transparency, and reproducibility, then show how the framework satisfies them.

3. **A small set of named abstractions.**
   DSPy has signatures/modules/teleprompters; RLlib has scalable primitives; Habitat has Habitat-Sim and Habitat-API; PyTorch emphasizes define-by-run Python programs and runtime components. AutoRAG-Research should name its own abstractions: ingestors, embedded corpus dumps, retrieval pipelines, generation pipelines, metrics, manifests, and fidelity cards.

4. **Layered stack / architecture figure or table.**
   Framework papers show where the contribution sits in the stack. AutoRAG-Research uses a clear stack from raw datasets to normalized database, embedded corpus, pipelines, metrics, artifacts, and reproduction modes.

5. **Coverage evidence, not only performance numbers.**
   Framework value is demonstrated by breadth: number of algorithms, datasets, components, tasks, or supported backends. AutoRAG-Research should report coverage of ingestors, pipelines, metrics, dumps, and configs as a primary result.

6. **Enabled experiments or case studies.**
   DSPy and Habitat show scientific findings enabled by the framework. AutoRAG-Research should frame the benchmark as a reimplementation study enabled by the common substrate, not merely as a leaderboard.

7. **Artifact and reproduction story in the main text.**
   Good framework papers make code/data availability and reusable artifacts central. For AutoRAG-Research, pre-ingested and pre-embedded PostgreSQL dumps must be a main contribution, not an appendix detail.

8. **Acknowledge what the framework cannot solve.**
   The limitations should be explicit: unavailable proprietary model snapshots, license-constrained datasets, hosted LLM drift, full-rerun cost, and inability to guarantee exact historical score equality.

## Target AutoRAG-Research paper shape

1. Introduction: RAG reproducibility as a field bottleneck; thesis; contributions.
2. Design requirements from framework papers: composability, dataset normalization, embedded-corpus reproducibility, transparent reimplementation, artifact-backed evaluation.
3. Framework: layered stack, ingestors, embedded dumps, pipeline/metric abstractions.
4. Reimplementation protocol: fidelity cards, manifests, caching, query/corpus controls, reproduction modes.
5. Evaluation: coverage, dump reproducibility, full pipeline benchmark, cost/runtime/artifact evidence.
6. Discussion/limitations/broader impact.
7. Conclusion.

## Concrete rewrite directives

- Replace dataset tables that look like an experiment subset plan with tables showing **framework coverage**.
- Keep all implemented pipelines in the full-scale matrix; cost control should be described as a reproducibility mechanism, not a reason to omit methods.
- Treat numeric cells as final-result slots (`XX`, `0.XXX`) without using a proposal-style status sentence.
- Move the pre-embedded corpus dump feature into the abstract, introduction, framework section, evaluation section, and artifact appendix.
- Include citations to framework papers to make the manuscript read like a framework contribution, not a narrow experiment plan.
