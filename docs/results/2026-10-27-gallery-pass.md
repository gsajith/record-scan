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

Started 2026-10-27 13:01 EDT. After prefix 00 the run put the rest at about 8.7 h. At 1,084 MB
per prefix the shards come to about 278 GB, against 346 GB free when it started. Totals land
here when it ends, followed by the duplicate clusters (#20).

## Earlier attempts

The first version resized every cover to a 256 px short side at JPEG quality 90 (`712c296`,
then `3b6a4a2`). Prefix 00 then took 157 s and gave a 355 MB shard, which put the full copy at
about 92 GB. That run was stopped after six prefixes, with 0 covers skipped in 93,342. The
review fixes changed its recipe; then Gautham chose to keep the originals, since the models'
input sizes run from 224 to 448 px and the sweep hasn't picked one. Its 2.2 GB were removed.
