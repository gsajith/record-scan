## Agent skills

### Issue tracker

Issues live in GitHub Issues on gsajith/record-scan, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

A ticket is unblocked when every issue in its native `blocked_by` list is closed. Read that list (`gh api repos/gsajith/record-scan/issues/<n>/dependencies/blocked_by`) before calling a ticket unblocked.

Phase A of `docs/superpowers/plans/2026-10-24-identifier-plan.md` is tracked as issues #7 to #33, each citing its plan task. Keep the two in sync: when a plan task lands, close its issues, tick the plan's checkboxes and update its ledger in the same session, whichever one you started from.

### Triage labels

The five default labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one root `GLOSSARY.md` plus `docs/adr/`. See `docs/agents/domain.md`.
