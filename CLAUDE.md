## Agent skills

### Issue tracker

Issues live in GitHub Issues on gsajith/record-scan, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

A ticket is unblocked when every issue in its native `blocked_by` list is closed. Read that list (`gh api repos/gsajith/record-scan/issues/<n>/dependencies/blocked_by`) before calling a ticket unblocked.

Phase A of `docs/superpowers/plans/2026-10-24-identifier-plan.md` is tracked as issues #7 to #33, each citing its plan task. Keep the two in sync: when a plan task lands, close its issues, tick the plan's checkboxes and update its ledger in the same session, whichever one you started from.

### Triage labels

The five default labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one root `GLOSSARY.md` plus `docs/adr/`. See `docs/agents/domain.md`.

### Code review

`REVIEW.md` says what reviews check here and which paths need Gautham's approval. Any review reads it: `/work-issue`'s reviewer, `/mattpocock-skills:code-review` (which `/mattpocock-skills:implement` ends with) and ad hoc subagents. Pass its path to every reviewer subagent, and to both of code-review's sub-agents as a standards source.

- A re-review after fixes covers the commits since the last review, plus whether each earlier finding is fixed. With code-review, pass the last reviewed SHA as the fixed point.
- After a third round of findings, stop patching and reconsider the approach.
- When a diff touches a path under "Human approval" in `REVIEW.md`, don't merge it. Say so in the final report.
- PR bodies fill in `.github/pull_request_template.md`.
