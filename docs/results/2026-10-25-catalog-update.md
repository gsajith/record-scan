# Catalog update from the 2026-10-24 MusicBrainz dump

Plan Task 6, issue #21, decision D11 as revised on 2026-10-25. Code at the commit that adds this
file; base `039305c`. All commands ran from the repo root with `pipeline/.venv`.

## Inputs

| Input | Source | Size |
|---|---|---|
| `release.tar.xz` | `json-dumps/20261024-001001/` on data.metabrainz.org | 21.9 GiB, 363.8 GiB unpacked |
| `mbdump-cover-art-archive.tar.bz2` | `fullexport/20261024-102842/` | 162 MB, 7,633,851 images |
| previous catalog | built from the 2025-08-13 dump | 3,129,812 releases |

Internet Archive search, the source the plan first named, lists 230 Cover Art Archive items in
total, because the rest are `noindex`. Checked with the scrape API and `advancedsearch.php` on
`collection:coverartarchive`.

## Slim

`python -m pipeline.metadata.slim_file $D/release.tar.xz --date 2026-10-24` took 67 min.

- 5,874,691 releases read, 3,966,332 with cover art, 815,954 artists
- every kept release has `release_group`

## Backlog

`python -m pipeline.metadata.fetch_new_covers --date 2026-10-24 --since 2025-08-13 --dry-run`
took 41 s:

| | Count |
|---|---|
| 2025 covers kept, still on the Cover Art Archive | 3,110,764 |
| 2025 covers dropped, deleted upstream or release gone from MusicBrainz | 17,998 |
| new fronts on releases we already had | 14,849 |
| releases to look up, new or with no cover left | 855,568 |
| **backlog** | **870,417** |

After the PR review, the fetcher reads its starting covers from a fixed snapshot,
`covers-2025-08-13.jsonl` (`python -m pipeline.legacy.seed_covers_2025`), instead of the live
catalog. The seed holds 3,128,310 covers. It leaves out 453 of the 688 failed 2025 downloads,
the ones whose file really is missing, so they get looked up again. The dry run then reported
3,110,638 kept, 17,672 dropped, 14,846 new fronts and 855,694 lookups, with 861,251 left to fetch
after about 9,000 had landed.

## Live runs

| Run | Items | Fetched | Failed | Rate |
|---|---|---|---|---|
| `--limit 500`, 16 workers | 500 | 499 | 1 | 1.9/s |
| `--limit 1000 --workers 48` | 1,000 | 989 | 11 | 2.7/s |
| full run, `--workers 48`, first 3,000 | 3,000 | 2,989 | 11 | 6.6/s |

The short runs understate the rate. Each one ends waiting on its slowest item, a release whose
retries take about three minutes. The full run keeps every worker fed and holds 6.5 to 6.6 items/s,
so it should take about 36 h. It was restarted on the reviewed code after about 9,000 items. One request at a time, a lookup and its image take about 2.2 s. At
32 parallel lookups, coverartarchive.org answered 12.9 requests/s with no 429 or 503. Most
failures are fresh uploads that have no 500 px thumbnail yet (404). A rerun retries every
failure.

## Catalog

After the first 2,000 or so covers, `python -m pipeline.metadata.build_catalog_db` took 144 s:

- 3,966,332 releases, all with `release_group`
- 3,111,857 with a cover, 2.3 GB
- covers table: 3,112,939 rows, 2,175 of them new `<mbid>-<cover_id>.jpg` files

Rebuild again when the full run ends.
