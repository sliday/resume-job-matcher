#!/usr/bin/env bash
set -euo pipefail

# Environment setup -- runs at the start of every iteration
# ---------------------------------------------------------
#
# Must be idempotent. Update this script instead of installing
# packages inline, so the environment stays reproducible across
# iterations. The loop activates the repo venv automatically.

# Idempotent dependency install for the TS build (added by orchestrator)
npm install --no-audit --no-fund
