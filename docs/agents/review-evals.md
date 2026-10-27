# Review evals

Past diffs with a known answer, for scoring the reviewer. Rescore after changing the reviewer's prompt (in `/work-issue` or the `CLAUDE.md` review rules), its model or `REVIEW.md`. A change that loses a bug case, or adds false alarms on a clean one, doesn't ship.

## Cases

| id | diff | known answer |
|---|---|---|
| bug-43 | `0418370^..0418370` (PR #43 as merged) | Shipped bug, fixed by #45. Shards, the eval gallery cache and the index aren't keyed by an embedder's preprocessing, so vectors embedded the old way stay in use after it changes. PE-Core's changed during #43's review. Review focus 7, high or medium, in `pipeline/retrieval/` (`embed.py`, `eval.py`, `build_index.py`, `identify.py`). Pre-existing or new both count. |
| bug-45 | `0418370..6e02bfd` (PR #45's first push, in `refs/pr/45`) | Caught in round 1, fixed in a05b191. The `.done` marker ignores the signature (`pipeline/retrieval/embed.py`), so `embed` skips a prefix whose shard has the old signature and the re-embed the error messages prescribe never happens. Review focus 7, medium. |
| bug-42 | `f1e2dcb^..f1e2dcb` (the last of PR #42's three rebase-merged commits) | Shipped bug, #53. `same_album` stops calling `album_of` and tests for "no release group" differently, and `album_of`'s docstring still says eval uses it (`pipeline/metadata/catalog.py`). Low. The PR description covers all three commits. |
| clean-44 | `0208fec^..0208fec` (PR #44) | No known bug. The labeler carries crate and rack tags between frames of one video. |
| bug-46 | `c266aa6^..fada1c3` (PR #46, two rebase-merged commits) | Shipped bugs, #55, both low. A photo first dropped without a frame takes a clip's group when the same bytes later arrive as a frame (`annotator/lib/store.ts`), and the TypeScript and Python label checks disagree on a blank group (`annotator/lib/frames.ts`, `annotator/lib/labels.ts`). Focus 1 and 2. |

A bug case passes when a finding at the known severity or above names the file and the class of bug. A clean case passes with no finding of medium or above that holds up when checked. A finding that holds up on a clean case is a new bug: file it and turn the case into a bug case.

## Run

1. `git fetch origin pull/45/head:refs/pr/45` (bug-45's head isn't on any branch).
2. For each case, `git worktree add --detach <scratchpad>/eval-<id> <head>` and copy the current `REVIEW.md` into it. The old commits predate it.
3. Spawn one fresh reviewer per case, in parallel, with `/work-issue`'s reviewer prompt. Give it the case's diff command, the PR description and the issue it closed, and tell it to return the review instead of posting it. Don't tell it the known answer. Point it at the main checkout's venv (`~/Projects/record-scan/pipeline/.venv`) for the fast tests.
4. Score each case, add a row below, and remove the worktrees.

## Scores

Only one clean case is left. Add the next PR that ships with no known bug.

| date | reviewer | bug-43 | bug-45 | bug-42 | bug-46 | clean-44 | notes |
|---|---|---|---|---|---|---|---|
| 2026-10-27 | `/work-issue` prompt with `REVIEW.md`, Opus 5.5 | pass (high) | pass (medium) | pass (low) | not run | pass (none) | Invalid for bug-43 and bug-45: focus 7 then named #43 and #45 as its examples. bug-42 ran as a clean case and found its bug, which became #53. bug-43 ran on `f1e2dcb..0418370`, which also holds the docs-only b7192ee. |
| 2026-10-27 | same, focus 7 without PR numbers | pass (high) | pass (medium) | not rerun | pass (2 low) | not rerun | Rerun of the contaminated cases. bug-46 ran as clean-46 and found two bugs, which became #55. |
