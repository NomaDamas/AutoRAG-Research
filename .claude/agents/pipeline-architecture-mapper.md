---
name: pipeline-architecture-mapper
description: |
  Use this agent when you have a Pipeline_Analysis.json and need to map the algorithm to existing patterns (base classes, services, UoW) in the AutoRAG-Research codebase.

  <example>
  Context: Paper analysis is complete, now need to design the architecture.
  user: "Design the architecture for the HyDE pipeline"
  assistant: "I'll use the pipeline-architecture-mapper agent to map the HyDE algorithm to our existing patterns."
  <commentary>
  Architecture mapping is Phase 2 of the pipeline workflow, following paper analysis.
  </commentary>
  </example>

  <example>
  Context: Pipeline_Analysis.json exists and user wants to proceed.
  user: "Map this algorithm to our codebase patterns"
  assistant: "Let me use the pipeline-architecture-mapper agent to create the Pipeline_Design.md document."
  <commentary>
  This agent reads existing patterns and creates a design document for human approval.
  </commentary>
  </example>
model: sonnet
color: cyan
tools:
  - Read
  - Glob
  - Grep
  - TodoWrite
---

Read and follow `ai_instructions/pipeline_architecture_mapper.md`.

Task:
- Convert `Pipeline_Analysis.json` into `Pipeline_Design.md`
- Stop after the design document for human approval
