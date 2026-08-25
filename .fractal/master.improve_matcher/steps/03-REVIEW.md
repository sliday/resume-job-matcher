---
requires_approval: false
---

## Review

When the work product carries claims of record (theorems, tables, measured
results), verify the PRINTED claims themselves -- re-derive tables from their
stated rule rather than trusting the machine artifacts that generated them, and
re-run a verification whose subject changed since it last ran: a review that
predates the newest claim covers nothing.

Review the diff (`git status`, `git diff`) for mistakes, missed edge cases, and
style violations; fix and re-validate.

**Update memory** per the memory skill: fold this iteration's findings into
child pages and split any bloated index into child pages. Then
`wiki lint --path=$MEMORY_DIR` and repair what it flags; iterate until the only
remaining lines are `Requires update` diffs (the indexes refresh mechanically at
commit).

**Project-wide learnings** (architecture, conventions, patterns useful to other
nodes) go in `$WIKI_DIR`, not node memory. After editing, run
`wiki lint --path=$WIKI_DIR`.

Append a `## Post-Mortem` section (accomplishments, deviations, next-iteration
notes) to each plan you wrote this iteration -- list them with
`fractal plan list`. If you adopted a plan from an interrupted earlier
iteration, append to that plan instead and note it; if there is no plan this
iteration, skip.
