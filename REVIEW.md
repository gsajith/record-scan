# Review

What a reviewer of this repo checks, on top of general correctness. Every review reads this file, whichever skill or agent runs it.

## How to report

- Open with the number of findings, or "No issues found."
- Cite `file:line` for every finding. Don't guess file names.
- For each focus item below that the diff touches, say whether it holds.
- Name the class of bug, not one line, so the fix covers every instance.
- Mark a bug that's already on `main` as pre-existing.
- Skip style nits. CI runs the labeler's lint, `tsc` and tests, and the pipeline's fast tests. Slow and MPS tests run only on the Mac, and nothing in `web/` runs in CI.
- On a re-review, review the commits since the last review and check each earlier finding. Don't re-report a finding already raised.

## Ignore

- Hashes inside `pipeline/requirements*.lock` and every `package-lock.json`. CI checks that the two Python locks agree. A new or upgraded dependency still deserves a look.

## Human approval

An agent may not merge a PR whose diff touches any of these. It stops and hands the PR to Gautham, whatever the reviewer's verdict. A bug here is expensive or hard to notice. The list guards behavior, not file names: when that behavior moves to a new file, add the file.

- `.github/` (CI and the PR template)
- `pipeline/requirements*.txt`, `pipeline/requirements*.lock`, every `package.json` and `package-lock.json` (dependencies)
- `pipeline/config.py`, `pipeline/metadata/fetch_new_covers.py`, `pipeline/legacy/` (paths and writes to the drive, focus 5)
- `pipeline/retrieval/eval.py`, `pipeline/metadata/catalog.py` (`same_album`), `pipeline/gold/labels.py`, `annotator/lib/labels.ts`, `annotator/lib/frames.ts`, `annotator/lib/store.ts` (hit rule, split, split groups and label format, focus 2 and 6)
- `REVIEW.md`, `docs/agents/review-evals.md` (the reviewer's own rules)

## Checking the reviewer

`docs/agents/review-evals.md` lists past diffs with known bugs and known clean diffs. Rescore the reviewer on them after changing its prompt, its model or this file.

## Focus

The numbers are stable: the plan and commit messages cite them as "Review focus N". Each item names the plan task that owns its test (`docs/superpowers/plans/2026-10-24-identifier-plan.md`). When a shipped bug shows a class missing here, add it as a new number and add the bug to the evals. Describe the class, not the PRs that shipped it: the evals copy this file into old commits, and a PR number would hand the reviewer the answer. CI fails on a `#<number>` in this section.

1. Two labels for one image: the labeler appends a new line and the loader keeps the latest line per `sha1`; a re-label never silently keeps the old answer. Test in Task 2.
2. Split drift: the Python and TypeScript `split_for(sha1, group)` return the same answer for the same input; the same five literal sha1 values and the same three published FNV-1a vectors are asserted in both suites. Tests in Tasks 2 and 3.
3. Catalog search misses on accents and case: "bjork" finds Björk and "the beatles" finds "Beatles, The"; FTS5 is built with `unicode61 remove_diacritics 2`. Test in Task 1.
4. A corrupt JPEG or an unmounted drive mid-run: the gallery pass logs and skips the file, finishes the prefix, and resumes per prefix on rerun; it never creates folders under `/Volumes`. Test in Task 5.
5. Lost or clobbered covers in a catalog update: the cover fetcher never overwrites or deletes a file on the drive, keeps an old cover while it is still upstream, backs off on 503, and resumes from its lookup log after a crash without repeating a request. Test in Task 6.
6. Metric gaming: a model task that changes the hit rule, the split, seeds or test assertions so that a number improves without the model improving. Reviewers check for it in every task from Task 4 on; the eval-integrity constraint forbids it in the same commit.
7. Stale derived artifacts: anything cached or marked done (shards, `.done` markers, indexes, eval gallery caches, resume logs) is keyed by everything that produced it, so changing the producer (preprocessing, weights, input size, the cover list) makes the old artifact rebuild or get refused. Reviewers check every diff that changes a producer for a cache that would still be reused.
