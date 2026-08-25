---
requires_approval: false
---

## Commit

1. If the Completion Requirements section is non-empty and every requirement is
   met, verify all children are finished or killed (`fractal node list`) -- do
   not self-complete with active children. Then signal completion (if empty,
   never self-complete): `fractal node finish --reason="<reason>"`.

   After finishing, if your parent is the user (root) node, post a brief
   sign-off to them -- what you accomplished, the final state, and any decision
   the user owns:
   `fractal radio send "<summary>" --parent --subject="<subject>" --priority=<0-10>`.
   Fire-and-forget; do not wait for a reply.

2. Commit: `fractal commit "<short lowercase summary>"` -- this checks scope,
   lints, stages, commits, and pushes (unless `--local` was passed to
   initialization). Fix and retry on lint failure. Hook reformats of project
   files are auto-retried once -- review what they changed (`git diff HEAD~`).
   Hook rewrites of wiki pages (`wiki/` and your memory wiki) are auto-retried
   too when they preserve wiki structure (a breaking rewrite fails with the fix
   in the error); any hook rewrite of other `.fractal/` pages is refused -- and
   never run project format hooks over those paths yourself: damage applied
   before staging bypasses the commit-time guard.

   The summary is wrapped as `<branch>: iteration <run>.<iter> (<summary>)` --
   pass only the bare summary. A message containing the branch name or the word
   "iteration" is rejected; re-commit with a plain summary.

   If it rejects **out-of-scope changes** (e.g. ancestor `_index.md` files
   touched by `wiki update`), revert them and retry:

   ```bash
   git checkout HEAD -- <out-of-scope files listed in the error>
   fractal commit "<summary>"
   ```

   When the out-of-scope changes are genuinely intentional, commit them with
   `--ignore-scope` (it still lints):

   ```bash
   fractal commit "<summary>" --ignore-scope
   ```

   Reserve `--force` for a true last resort -- it bypasses the scope check *and*
   lint.
