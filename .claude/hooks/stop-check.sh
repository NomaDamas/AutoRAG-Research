#!/bin/bash
# Stop hook: runs make check with automatic dependency recovery
# If make check fails due to missing deps, runs uv sync and retries once
# Always exits 0 (errors reported via stdout, following lint-file.sh pattern)

cd "$CLAUDE_PROJECT_DIR"

echo "🔍 Running make check..."

OUTPUT=$(make check 2>&1)
if [[ $? -eq 0 ]]; then
  echo "✅ make check passed"
  exit 0
fi

# First attempt failed — try syncing dependencies and retrying
echo "⚠️ make check failed. Running uv sync --all-extras --all-groups..."

SYNC_OUTPUT=$(uv sync --all-extras --all-groups 2>&1)
if [[ $? -ne 0 ]]; then
  echo "❌ uv sync failed:"
  echo "$SYNC_OUTPUT"
  echo ""
  echo "❌ Original make check output:"
  echo "$OUTPUT"
  exit 0
fi

echo "🔄 Retrying make check..."

RETRY_OUTPUT=$(make check 2>&1)
if [[ $? -eq 0 ]]; then
  echo "✅ make check passed after uv sync"
  exit 0
fi

echo "❌ make check still failing after uv sync:"
echo "$RETRY_OUTPUT"

exit 0
