#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v node >/dev/null 2>&1; then
  echo "Error: Node.js is required to run Agent Council job mode." >&2
  echo "Install Node.js and try again (plugin installs cannot bundle Node)." >&2
  echo "" >&2
  echo "macOS (Homebrew): brew install node" >&2
  echo "Or download from: https://nodejs.org/" >&2
  exit 127
fi

exec node "$SCRIPT_DIR/council-job.js" "$@"
