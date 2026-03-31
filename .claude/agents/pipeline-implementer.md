---
name: pipeline-implementer
description: |
  Use this agent when you have Pipeline_Design.md and tests written, and need to implement the actual pipeline code to pass the tests.

  <example>
  Context: Tests are written and failing, now need implementation.
  user: "Implement the HyDE pipeline to pass the tests"
  assistant: "I'll use the pipeline-implementer agent to write the production code based on the design."
  <commentary>
  Implementation follows TDD - code is written to pass existing tests.
  </commentary>
  </example>

  <example>
  Context: Phase 3 (tests) is complete, proceeding to Phase 4.
  user: "Write the pipeline implementation"
  assistant: "Let me use the pipeline-implementer agent to create the pipeline class."
  <commentary>
  Implementation uses the design document and must pass the pre-written tests.
  </commentary>
  </example>
model: opus
color: yellow
tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
---

Read and follow `ai_instructions/pipeline_implementer.md`.

Task:
- Implement the approved pipeline from `Pipeline_Design.md`
- Update the pipeline module, config wiring, and exports
- Do not redesign the algorithm or rewrite the tests
