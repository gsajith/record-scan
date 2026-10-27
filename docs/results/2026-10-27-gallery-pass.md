# Gallery pass: 256 px shards and perceptual hashes

Plan Task 5, issue #19, decision D10. The prefix 00 numbers below come from `712c296`; the full run runs at `3b6a4a2`, whose recipe adds Pillow's version (`short256-q90-lanczos-phash8-pillow11.3.0`), so it redid prefix 00: same 15,375 covers, 0 skipped, 172 s. Commands ran from the repo root with
`pipeline/.venv`, the Seagate drive mounted over USB, and the catalog built from
`covers-2026-10-24.jsonl` (3,977,195 covers).

## Prefix 00

`python -m pipeline.gallery.shard_and_hash --prefixes 00` took 2 min 47 s in all: 10 s to list
every prefix's covers from the catalog, then 157 s for the prefix.

| | Value |
|---|---|
| covers in the catalog for prefix 00 | 15,375, of which 3,360 are extra fronts |
| written | 15,375 |
| skipped as unreadable | 0 |
| rate | 98 covers/s at 46% CPU, so the drive's read speed sets the pace |
| `00.tar` | 355 MB, 23 KB per cover |
| `00.phash.npy` | 4.1 MB |
| distinct hashes | 15,344 of 15,375 |

Covers on the drive are mostly 500 x 500 and about 62 KB each. A 1-in-50 sample of the shard had
304 of 307 at a 256 px short side; the other 3 were smaller covers, which the pass doesn't
upscale.

The drive folder `00/` holds 15,436 files, 61 more than the catalog lists. Those are covers
deleted upstream (D11), which stay on disk and stay out of the gallery.

## Estimate for the rest

The run printed 3,961,820 covers left across 255 prefixes, about 11.3 h at prefix 00's rate.
At 23 KB per cover the shards come to about 92 GB; the internal disk had 232 GB free.

## Full run

A first attempt started at 12:18 and was stopped after prefix 04 (0 skipped in 77,567 covers) to
take the PR review's fixes, which change the recipe. The run restarted 2026-10-27 12:34 EDT, detached:

```
nohup caffeinate -i pipeline/.venv/bin/python -u -m pipeline.gallery.shard_and_hash --all \
  > ~/Projects/record-scan-data/gallery256/run-all.log 2>&1 &
```

In the first attempt, prefixes 01 to 04 took 175 to 188 s each (82 to 89 covers/s), about 12.5 h
for the rest. Totals land here when it
ends, followed by the duplicate clusters (#20).
