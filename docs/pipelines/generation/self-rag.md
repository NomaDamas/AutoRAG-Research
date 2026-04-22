# SelfRAG

SelfRAG is implemented here as a **generation pipeline** that composes an existing retrieval pipeline. This repo does **not** reproduce the paper's fine-tuned reflection-token model. Instead, it uses a prompt-based approximation that:

1. drafts an answer without retrieval,
2. reflects on whether evidence is needed,
3. retrieves only when reflection requests it,
4. revises the answer against retrieved evidence, and
5. stops once the answer is judged supported or the reflection budget is exhausted.

## Classification

SelfRAG is **not a pure retrieval pipeline** in this repository's architecture. It belongs in generation because retrieval is only one tool inside the answer-generation loop.

## Configuration

```yaml
_target_: autorag_research.pipelines.generation.self_rag.SelfRAGPipelineConfig
name: self_rag
retrieval_pipeline_name: bm25
llm: gpt-4o-mini
max_reflection_steps: 3
top_k: 4
```

## Main Parameters

- `retrieval_pipeline_name`: retrieval pipeline composed into SelfRAG
- `llm`: model used for drafting, reflection, and revision
- `max_reflection_steps`: maximum number of reflection iterations
- `initial_prompt_template`: initial answer prompt
- `reflection_prompt_template`: structured reflection prompt
- `revision_prompt_template`: answer revision prompt

## Notes

- This implementation is a **prompt-based approximation** suitable for benchmarking within the existing pipeline abstractions.
- The reflection prompt defaults to JSON output (`should_retrieve`, `is_supported`, `follow_up_query`, `critique`) and also tolerates simple `KEY: value` lines for experimentation.
