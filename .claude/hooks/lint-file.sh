#!/bin/bash
# Lint check for individual Python files (PostToolUse hook)
# Runs after Edit/Write operations to provide immediate feedback

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

# Ruff check (linting)
RUFF_OUTPUT=$(uv run ruff check "$FILE_PATH" 2>&1)
RUFF_EXIT=$?

# Ruff format check
FORMAT_OUTPUT=$(uv run ruff format --check "$FILE_PATH" 2>&1)
FORMAT_EXIT=$?

# Report results to Claude
if [[ $RUFF_EXIT -ne 0 ]]; then
  echo "❌ Ruff lint errors:"
  echo "$RUFF_OUTPUT"
fi

if [[ $FORMAT_EXIT -ne 0 ]]; then
  echo "❌ Ruff format errors:"
  echo "$FORMAT_OUTPUT"
fi

if [[ $RUFF_EXIT -eq 0 && $FORMAT_EXIT -eq 0 ]]; then
  echo "✅ Lint/format check passed"
fi

# Always exit 0 to allow Claude to continue (errors are reported via stdout)
exit 0
