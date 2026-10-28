# Gallery pass: original covers in shards, with perceptual hashes and sizes

Plan Task 5, issue #19, decision D10 as revised by Gautham on 2026-10-27. Code at `4d1b7b3`,
recipe `original-phash8-pillow11.3.0-scipy1.18.1-numpy2.2.6`. Commands ran from the repo root
with `pipeline/.venv`, the Seagate drive mounted over USB, and the catalog built from
`covers-2026-10-24.jsonl` (3,977,195 covers).

## Prefix 00

The full run below started with prefix 00, which took 122 s after a 10 s listing of every
prefix's covers from the catalog.

| | Value |
|---|---|
| covers in the catalog for prefix 00 | 15,375, of which 3,360 are extra fronts |
| written | 15,375 |
| skipped as unreadable | 0 |
| rate | 126 covers/s; the drive's read speed sets the pace |
| `00.tar` | 1,084 MB |
| `00.covers.npy` | 4.2 MB |
| byte-identical to the drive | 100 of 100 sampled entries |

Sizes from `00.covers.npy`:

| | Covers |
|---|---|
| 500 x 500 | 10,173 |
| long side under 500 px (smaller originals) | 1,229 |
| short side under 256 px | 294 |
| aspect ratio over 2:1 | 36 |

The drive folder `00/` holds 15,436 files, 61 more than the catalog lists. Those are covers
deleted upstream (D11), which stay on disk and stay out of the gallery.

## Full run

```
nohup caffeinate -i pipeline/.venv/bin/python -u -m pipeline.gallery.shard_and_hash --all \
  > ~/Projects/record-scan-data/gallery/run-all.log 2>&1 &
```

Started 2026-10-27 13:01 EDT and ended 2026-10-28 00:18 EDT, 11 h 17 min in all. The rate held near
100 covers/s through the middle prefixes and rose to 110 covers/s for the last ones.

| | Value |
|---|---|
| covers written | 3,977,195, every cover in the catalog, of which 866,557 are extra fronts |
| skipped | 0; no errors logs, no prefix left pending |
| prefixes done | 256 of 256, all with recipe `original-phash8-pillow11.3.0-scipy1.18.1-numpy2.2.6` |
| shards | 281.9 GB |
| `covers.npy` files | 1.1 GB |
| distinct pHashes | 3,683,556, so about 294,000 covers share an exact hash with another |
| internal disk free after the run | 77 GB |

Sizes from all 256 `covers.npy` files:

| | Covers |
|---|---|
| 500 x 500 | 2,635,850 |
| long side under 500 px (smaller originals) | 327,645 |
| short side under 256 px | 77,668 |
| aspect ratio over 2:1 | 6,400 |

The duplicate clusters (#20) come next and need only `GALLERY_DIR`, not the drive.

## Duplicate clusters

Issue #20, code at `a67368a`, run 2026-10-28 on the 256 `covers.npy` files above, all with one
hash recipe. No drive.

```
PYTHONPATH=. pipeline/.venv/bin/python -m pipeline.gallery.cluster_duplicates   # --max-distance 8
```

| | Value |
|---|---|
| covers | 3,977,195 |
| releases (N) | 3,962,307 |
| clusters at Hamming distance <= 8 (deduplicated N) | 2,817,037 |
| time | 339 s, 12.3 GB peak memory |
| `clusters.npy` | 602 MB |

| Cluster size, in releases | Clusters | Releases |
|---|---|---|
| 1 | 2,472,329 | 2,472,329 |
| 2 | 247,739 | 495,478 |
| 3 | 54,268 | 162,804 |
| 4 | 19,152 | 76,608 |
| 5 to 9 | 19,287 | 117,082 |
| 10 to 99 | 4,253 | 67,275 |
| 100 to 999 | 8 | 1,734 |
| 1,000 or more | 1 | 568,997 |

The largest clusters hold 568,997, 340, 301, 301 and 229 releases.

### Distance 8 is too loose, and joins chain

The one cluster of 568,997 releases, 14% of the catalog, is a bug in the rule, not a finding about
the catalog. A cluster is every cover reachable through pairs within the distance, so one loose
pair links two groups. `cluster_thresholds.py` and `cluster_sheets.py` beside this file measured
why: they find every pair of distinct hashes within 8 bits, count components at each threshold,
and sample pairs at each distance onto contact sheets.

Pairs of distinct hashes by distance (pHash sets about half its 64 bits by a median split, so odd
distances are rare):

| Distance | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| pairs | 138 | 144,908 | 714 | 277,768 | 783 | 689,087 | 898 | 2,255,336 |
| same picture, of 16 sampled | | 14 | | 12 | | 10 | | 2 |

The pairs that aren't the same picture are, at distance 2, plain covers with a small motif; at 4,
also series covers on one template (one label's releases with the logo on another color, volume 1
and volume 17 of one compilation) and dark covers; at 8, mostly unrelated covers.

Largest component at each threshold, in covers when clustering by hash alone and in releases when
a release's extra fronts are joined too:

| Distance <= | 0 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| components, by hash | 3,683,556 | 3,561,581 | 3,449,885 | 3,283,949 | 2,824,790 |
| largest, by hash (covers) | 1,549 | 2,394 | 28,802 | 123,970 | 568,639 |
| largest, with extra fronts (releases) | 1,548 | 2,393 | 28,824 | 124,153 | 568,997 |

Joining extra fronts barely matters. The chaining runs through low-detail covers: 32 random members
of the largest component at <= 4 share nothing but a plain background, and at <= 8 nothing at all.
At distance 0 the largest groups are placeholders: hash `0x8000000000000000` (1,549 covers, flat
images) and `0x0` (542). The exclusion list (#56) takes those out.

So these clusters must not reach `identify` or `eval`: a hit on any of 568,997 releases would count
as right. Choosing a rule that doesn't chain is a follow-up for Gautham.

## Earlier attempts

The first version resized every cover to a 256 px short side at JPEG quality 90 (`712c296`,
then `3b6a4a2`). Prefix 00 then took 157 s and gave a 355 MB shard, which put the full copy at
about 92 GB. That run was stopped after six prefixes, with 0 covers skipped in 93,342. The
review fixes changed its recipe; then Gautham chose to keep the originals, since the models'
input sizes run from 224 to 448 px and the sweep hasn't picked one. Its 2.2 GB were removed.
