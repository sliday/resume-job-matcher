#!/usr/bin/env bash
set -euo pipefail

# Run the node's test suite; exit 0 on success, non-zero on failure
# -----------------------------------------------------------------

# No-op by default; extend per node with the project's test command.

# Loop signal: typecheck + eval harness (added by orchestrator)
npx tsc --noEmit
npm run eval
