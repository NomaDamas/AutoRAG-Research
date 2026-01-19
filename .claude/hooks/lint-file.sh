#!/bin/bash
# Code quality check for individual Python files (PostToolUse hook)
# Runs after Edit/Write operations to provide immediate feedback
# Checks: ruff (lint/format) + ty (type check)
# Note: deptry requires project-level analysis, runs only in Stop hook via make check

# Read tool input JSON from stdin
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Skip if not a Python file
if [[ ! "$FILE_PATH" == *.py ]]; then
  exit 0
fi

# Skip if file doesn't exist
if [[ ! -f "$FILE_PATH" ]]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

echo "🔍 Checking: $FILE_PATH"

HAS_ERROR=0

# Ruff check (linting)
RUFF_OUTPUT=$(uv run ruff check "$FILE_PATH" 2>&1)
if [[ $? -ne 0 ]]; then
  echo "❌ Ruff lint errors:"
  echo "$RUFF_OUTPUT"
  HAS_ERROR=1
fi

# Ruff format check
FORMAT_OUTPUT=$(uv run ruff format --check "$FILE_PATH" 2>&1)
if [[ $? -ne 0 ]]; then
  echo "❌ Ruff format errors:"
  echo "$FORMAT_OUTPUT"
  HAS_ERROR=1
fi

# Type check (ty)
TY_OUTPUT=$(uv run ty check "$FILE_PATH" 2>&1)
if [[ $? -ne 0 ]]; then
  echo "❌ Type errors:"
  echo "$TY_OUTPUT"
  HAS_ERROR=1
fi

if [[ $HAS_ERROR -eq 0 ]]; then
  echo "✅ All checks passed"
fi

# Always exit 0 to allow Claude to continue (errors are reported via stdout)
exit 0
