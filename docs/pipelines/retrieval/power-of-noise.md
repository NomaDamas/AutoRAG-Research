# Power of Noise

A retrieval wrapper inspired by [The Power of Noise: Redefining Retrieval for RAG Systems](https://arxiv.org/abs/2401.14887).
Instead of only returning the base retriever's highest-ranked documents, this pipeline mixes in seeded random corpus documents so you can evaluate how noisy context changes downstream RAG quality.

## When to use it

- Reproduce paper-style retrieval-noise ablations
- Compare retrieval metrics vs generation quality under noisy context
- Test whether a generator is robust to distractor documents

## Key features

- Wraps any existing retrieval pipeline
- Deterministic sampling with `seed`
- Supports fixed `noise_count` or ratio-based `noise_ratio`
- Supports `retrieved_first`, `noise_first`, or `interleave` ordering
- Includes an evaluation-only `answer_aware_random` mode that excludes known positives and answer-containing chunks when query metadata is available

## Config

```yaml
_target_: autorag_research.pipelines.retrieval.power_of_noise.PowerOfNoiseRetrievalPipelineConfig
name: power_of_noise
base_retrieval_pipeline_name: vector_search
noise_count: 2
noise_order: interleave
noise_mode: random
seed: 42
top_k: 10
```

## Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_retrieval_pipeline_name` | `str` | required | Existing retrieval pipeline to wrap |
| `noise_count` | `int` | `0` | Fixed number of noisy documents to inject |
| `noise_ratio` | `float \| null` | `null` | Fraction of `top_k` reserved for noise when `noise_count == 0` |
| `noise_order` | `str` | `retrieved_first` | Final ordering: `retrieved_first`, `noise_first`, or `interleave` |
| `noise_mode` | `str` | `random` | `random` for deployable seeded noise, `answer_aware_random` for evaluation-only exclusion of positives / answer-containing chunks |
| `seed` | `int` | `0` | Deterministic sampling seed |

## Notes

- `noise_count` takes precedence over `noise_ratio`.
- `answer_aware_random` needs a DB-backed query with `generation_gt` and/or retrieval relations. For ad-hoc raw text queries, the wrapper falls back to standard random noise sampling.
- The wrapper preserves the base retriever's scores and assigns noise documents a score of `0.0`.
- `noise_count` and `noise_ratio` define the noise budget. If the base retriever under-fills its share, the wrapper does **not** silently add extra noise beyond that configured budget.
