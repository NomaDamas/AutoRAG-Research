# Overview

- Gather responses from configured member CLIs.
- Let the chairman synthesize the final response (default: `role: auto`, current agent).
- Configure members in `council.config.yaml`; exclude the chairman from members by default.
- Reference [Karpathy's LLM Council](https://github.com/karpathy/llm-council) for inspiration.

## Workflow (3 stages)

1. Send the same prompt to each member.
2. Collect and surface member responses.
3. Synthesize the final answer as chairman; optionally run the chairman inside `council.sh` via `chairman.command`.
