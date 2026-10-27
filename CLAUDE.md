## Agent skills

### Issue tracker

Issues live in GitHub Issues on gsajith/record-scan, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

A ticket is unblocked when every issue in its native `blocked_by` list is closed. Read that list (`gh api repos/gsajith/record-scan/issues/<n>/dependencies/blocked_by`) before calling a ticket unblocked.

A ticket being worked has an assignee and the `in-progress-by-agent` label, and a ticket with either is taken. Claim one before starting it: `gh issue edit <n> --add-assignee @me --add-label in-progress-by-agent`.

Phase A of `docs/superpowers/plans/2026-10-24-identifier-plan.md` is tracked as issues #7 to #33, each citing its plan task. Keep the two in sync: when a plan task lands, close its issues, tick the plan's checkboxes and update its ledger in the same session, whichever one you started from.

### Triage labels

The five default labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one root `GLOSSARY.md` plus `docs/adr/`. See `docs/agents/domain.md`.

### Code review

`REVIEW.md` says what reviews check here and which paths need Gautham's approval. Every review reads it, whichever skill or subagent runs it, so every reviewer gets its path as a standards source.

- A re-review covers the commits since the last review, plus whether each earlier finding is fixed.
- When a second review round still has findings, stop patching and reconsider the approach before the third.
- Before merge, a PR carries its latest review as a PR comment.
- A PR whose diff touches a path under "Human approval" in `REVIEW.md` merges only on Gautham's go-ahead.
- PR bodies fill in `.github/pull_request_template.md`.

## Running the pipeline

The Python venv exists only in the main checkout, `~/Projects/record-scan/pipeline/.venv`, and every worktree uses it. Run pytest from `pipeline/`, and scripts with `PYTHONPATH` set to the worktree root.
