---
name: test-writer
description: Use this agent when you need to generate test code for new or existing functions, classes, or modules. This includes when: (1) A new function or class has been implemented and needs corresponding tests, (2) Existing code lacks test coverage, (3) You want to verify behavior of recently written code, (4) Refactoring requires updated tests. Examples:\n\n<example>\nContext: User has just written a new service layer function.\nuser: "Create a function that retrieves all documents by user_id"\nassistant: "Here is the service function implementation:"\n<function implementation provided>\nassistant: "Now let me use the test-writer agent to generate comprehensive tests for this function."\n<Task tool invocation with test-writer agent>\n</example>\n\n<example>\nContext: User explicitly requests tests for existing code.\nuser: "Write tests for the ChunkRepository class"\nassistant: "I'll use the test-writer agent to generate tests for the ChunkRepository class."\n<Task tool invocation with test-writer agent>\n</example>\n\n<example>\nContext: After implementing a new pipeline component.\nuser: "Add a new retrieval pipeline that filters by document type"\nassistant: "Here is the new retrieval pipeline:"\n<pipeline implementation>\nassistant: "Let me generate tests for this new pipeline using the test-writer agent."\n<Task tool invocation with test-writer agent>\n</example>
model: opus
color: blue
---

Read and follow `ai_instructions/test_writer.md`.

Task:
- Write the requested tests under `tests/`
- Follow the existing project conventions and shared test-generation guidance
- Avoid redundant coverage and constructor validation tests
