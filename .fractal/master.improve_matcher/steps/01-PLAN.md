---
requires_approval: false
---

## Plan

Read the Instructions and Completion Requirements above for goals and direction.

**Check for interrupted work.** If a prior iteration was interrupted there may
be uncommitted changes (`git diff`, `git status`), or the last commit may be a
backstop save -- check `git log -1` for a `(failed on <step>)` label, whose body
names the failure and the steps that never ran. Adopt an existing plan and
continue partial work rather than starting fresh -- but still go through every
remaining step in order (don't skip EXECUTE/REVIEW because a plan already
exists). After a backstop save, treat `$NODE_DIR/tmp/` scratch as stale until
you regenerate it -- leftover artifacts there can impersonate completed work.

**Orient.** Before planning, read each `$NODE_DIR/skills/*/SKILL.md` for
available capabilities, then survey memory (`wiki map --path=$MEMORY_DIR`, then
the pages relevant to this iteration) and the shared project wiki
(`wiki map --path=$WIKI_DIR`) for conventions and patterns from other nodes.
Explore the project to understand its current state.

**Decompose before doing.** If `$MAX_DEPTH`, `$MAX_CHILDREN`, and
`$MAX_DESCENDANTS` are not `0`, evaluate whether the work ahead has separable
parts that child nodes should own. For complex goals, plan as a manager -- your
plan should include spawn commands, not deferred TODOs. A pure-management plan
is valid: "spawn N children to cover X, Y, Z; monitor and steer; merge results."
The `fractal` skill documents mechanics, configuration, and guardrails.

Decide what to do next from the goals and accumulated context, then create the
plan file with `fractal plan init --name=<short_descriptive_name>` (snake_case)
-- it seeds the file with a `# $ITER_REF <title>` heading ($ITER_REF is this
iteration's run.iter reference; pass `--title` to set the title) and prints the
path -- then write the plan below the heading. Use a separate
`fractal plan init` per concern when the work splits cleanly.

Re-read the plan and check feasibility against the budget (recheck remaining
time/cost -- see Context) and any blockers noted in memory. Trim the plan to fit
and update the file -- but a budget of "no limit" is not a reason to scope down.
