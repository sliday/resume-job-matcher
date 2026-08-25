---
requires_approval: false
---

## Execute

Execute the plan. If running low on time, finish the current sub-task cleanly
rather than starting a new one.

If your plan lists children to spawn and you have not yet spawned them, do so
now before any leaf work. Managing running children (checking status, editing
their NODE.md, sending directives, merging completed work) is also execution.

Verify with `bash $NODE_DIR/scripts/test.sh` if configured (exit 0 or no-op =
proceed) -- the loop never runs test.sh for you, and it runs in your ambient
CWD, so invoke it from the worktree root. Run `bash $NODE_DIR/scripts/lint.sh`
as you go to catch issues early and fix what you introduce; `fractal commit`
enforces it at COMMIT. Under fleet load a test suite can run far slower than its
solo baseline (concurrent nodes' test workers compound) -- budget step time for
it, and read slowness *with progressing output* as load; slowness with no new
output is a hang, not load.

The full memory update happens in REVIEW. But if you discover something that
would be lost if the session ended (a finding, blocker, or convention), write it
to memory now -- a new page needs `desc:` frontmatter and
`wiki update --path=$MEMORY_DIR` to be findable later. Your first memory write
creates the topical layout (one page per topic, stubs fine -- see the memory
skill), not a grab-bag page.

If you hit a blocker someone else owns, raise it on the radio now rather than
waiting for the next sync -- send it to your parent's inbox
(`fractal radio send "<note>" --parent --subject="<subject>" --priority=<0-10>`)
or a sibling's (swap `--parent` for `--node=<branch>`). `--subject` and
`--priority` are required. It is fire-and-forget, so do not pause execution
waiting on a reply.
