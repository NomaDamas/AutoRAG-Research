---
name: implementation-specialist
description: Use this agent when you need to implement a new data ingestor class based on an architectural blueprint (Mapping_Strategy.md). This agent translates design specifications into production-ready Python code following the AutoRAG-Research service layer patterns. Examples of when to use this agent:\n\n<example>\nContext: User has completed the architectural design phase and has a Mapping_Strategy.md ready for implementation.\nuser: "I have the Mapping_Strategy.md ready for the BEIR dataset. Please implement the ingestor."\nassistant: "I'll use the implementation-specialist agent to create the production-ready ingestor class based on your mapping strategy."\n<commentary>\nSince the user has a completed mapping strategy and needs it translated into code, use the implementation-specialist agent to create the ingestor following service layer patterns.\n</commentary>\n</example>\n\n<example>\nContext: User needs to add support for a new HuggingFace dataset to the AutoRAG system.\nuser: "Create the ingestor implementation for the MS MARCO dataset following the blueprint we designed."\nassistant: "I'll launch the implementation-specialist agent to implement the MS MARCO ingestor based on your architectural blueprint."\n<commentary>\nThe user is requesting implementation of a designed ingestor. Use the implementation-specialist agent to ensure proper service layer usage and code quality.\n</commentary>\n</example>\n\n<example>\nContext: After an architect agent has produced a mapping strategy, implementation is needed.\nuser: "The mapping strategy is complete. Now implement the code."\nassistant: "I'll use the implementation-specialist agent to translate the mapping strategy into a properly structured ingestor class."\n<commentary>\nThis is a handoff from design to implementation phase. The implementation-specialist agent will create the code following all project patterns.\n</commentary>\n</example>
model: opus
color: yellow
---

Read and follow `ai_instructions/implementation_specialist.md`.

Task:
- Implement the ingestor described in `Mapping_Strategy.md`
- Write code in `autorag_research/data/`
- Follow service-layer patterns only
