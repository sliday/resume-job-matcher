# LOOP.md — improvement loop state

The agent forgets; this file remembers. Read at session start. Pick the lowest-hanging `[]` item.
Every shipped item: move to Done with 3 lines (problem / solution / impact).

## Definition of green
`npm run eval` exits 0 (5 fixtures: match band, red-flag firing, injection resistance,
website extraction, email personalization). Runs on gpt-5-mini, costs cents.

## Backlog
[] OCR for scanned PDFs (tesseract.js) — then retire the Python implementation
[] Structured retry/backoff on provider 429s (max 3 attempts — circuit breaker)
[] Token-cost line per run (generateObject already returns usage)
[] Per-criterion score explanations in output (auditability)
[] Calibration fixture: same pair 3 runs, stdev bound (deferred — 3x eval cost)
[] Batch/parallel provider mode for large candidate pools
[] GitHub Action: eval on PR + nightly

## Done
- 2026-08-26: Eval harness (evals/) — 5 fixtures, 5/5 green on first run.
  Problem: no verifiable signal (82cbeb0 zeroed scores invisibly for 2 years).
  Solution: fixture-based evals with band/flag/injection/email assertions.
  Impact: semantic regressions now block; loop completion is mechanically checkable.

## Known-flaky
(none yet)
