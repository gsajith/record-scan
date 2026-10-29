# Gallery sanitation: blanks, strips and shared pictures

Issue #56, after plan Task 5 (#19, #68) and before the frozen sweep (#22). Run 2026-10-28 and
2026-10-29 on the 256 prefixes from `2026-10-27-gallery-pass.md` (3,977,195 covers of 3,962,307
releases) and the catalog built from `covers-2026-10-24.jsonl`. No drive. Commands ran from the
repo root with `pipeline/.venv`.

```
PYTHONPATH=. pipeline/.venv/bin/python -m pipeline.gallery.exclusions measure --workers 9   # 2c3d68c, 557 s
PYTHONPATH=. pipeline/.venv/bin/python docs/results/2026-10-28-gallery-sanitation/sheets.py <out dir>   # 7c01470
PYTHONPATH=. pipeline/.venv/bin/python -m pipeline.gallery.exclusions write   # c7e5f98, 30 s
PYTHONPATH=. pipeline/.venv/bin/python docs/results/2026-10-28-gallery-sanitation/counts.py   # c7e5f98
```

The list's recipe reads
`blank-strip-shared-v1 blank1.0 strip2.5 shared5release-groups-10first-artists rgb-std64-pillow11.3.0-numpy2.2.6 catalog-c506a85dfac1bcee gallery-adc597dcc38e303f`.
`load_exclusions` refuses any other, so a new cutoff, rule, catalog or gallery means rewriting
the list. Nothing on the drive or in the shards changed: the list is a filter.

## The rule

A cover is excluded for the first reason that applies:

- blank: spread under 1. Spread is the RGB standard deviation of a 64x64 copy, 0 to 255.
- strip: long side over short side above 2.5. These are strips, spines, obi strips, inserts and
  longboxes.
- shared: one exact pHash over 5 or more release groups whose releases have 10 or more distinct
  first credited artists, Various Artists skipped. These are label stock sleeves and placeholders.
  The picture can't name a release.

Gautham signed off on these cutoffs from the contact sheets on 2026-10-29. Small covers stay:
they still embed, and dropping one leaves its release unfindable. Covers around 2:1 stay too
(see below).

## Counts at the chosen cutoffs

| Reason | Covers | Releases |
|---|---|---|
| blank | 2,301 | 2,300 |
| strip | 938 | 931 |
| shared | 7,435 (323 pictures) | 7,435 |
| total | 10,674 | 10,666 |

10,472 releases have every cover excluded. They can't be found by picture. #70 makes them their
own eval slice.

Labeled positives with every cover of their album excluded: 0 of 8 in gold, 0 of 0 in wild. The
eval reports this count per run as `positives_all_covers_excluded`.

## Buckets on the contact sheets

The sheets are made of covers, so they stay out of git, in
`~/Projects/record-scan-data/gallery/sanitation-56/`: `blank.html`, `strip.html`, `sleeve.html`,
`templates-multi.html`, `templates-single.html` and `shared.html`. Each shows 100 random covers
per band, or the biggest 300 groups plus 100 random from the rest.

Spread, over the full gallery:

| Spread | Covers | Releases | On the sheet |
|---|---|---|---|
| 0 to 1 | 2,301 | 2,300 | solid color |
| 1 to 2 | 545 | 545 | real minimalist covers: text or a small motif on a plain ground |
| 2 to 4 | 2,026 | 2,024 | |
| 4 to 6 | 3,650 | 3,647 | |
| 6 to 8 | 5,404 | 5,403 | |
| 8 to 10 | 7,438 | 7,434 | |
| 10 to 12 | 10,047 | 10,044 | |
| 12 to 16 | 29,066 | 29,036 | |

Aspect ratio, long side over short:

| Aspect | Covers | Releases | On the sheet |
|---|---|---|---|
| 1.8 to 1.95 | 3,762 | 3,759 | kept |
| 1.95 to 2.05 | 5,861 | 5,789 | kept: landscape ones are back and front side by side, portrait ones mostly Japanese 8 cm CD singles |
| 2.05 to 2.5 | 3,184 | 3,171 | kept |
| 2.5 to 3 | 406 | 400 | strips, inserts, a few longboxes |
| 3 to 4 | 403 | 402 | |
| 4 to 6 | 112 | 112 | |
| 6 and over | 40 | 40 | |

Bands include their lower bound, so the 19 covers at exactly 2.5 sit in the 2.5 to 3 band but
stay in the gallery. Of the 942 above 2.5, 4 are also blank and count under blank.

#68's templates, split by artist count as the issue's comments asked:

| Artists | Templates | Releases |
|---|---|---|
| 0 or 1 | 2,080 | 20,237 |
| 2 to 4 | 890 | 7,844 |
| 5 to 19 | 1,494 | 14,427 |
| 20 and over | 355 | 22,124 |

Templates don't work as an exclusion bucket, at any artist count:

- The biggest multi-artist templates are distinct real covers. The first, 2,946 releases over 2,429
  artists, holds dark covers with a small motif, chained by noisy hashes at 2 bits. Others of 300
  to 700 releases are the same. Their median spread is about 20, so spread can't catch them.
- Templates with 2 to 4 artists are mostly one album plus its singles and featured artists, such
  as Charlie Puth's "Whatever's Clever!" or Ariana Grande's "Sweetener".
- Single-artist templates mix numbered series (Nutrition Facts episodes, show downloads) with
  album-and-singles families.

So the list uses exact pHash matches instead. Pictures with one exact pHash over 5 or more release
groups, counting all non-VA artists:

| Artists | Pictures | Releases |
|---|---|---|
| 5 to 9 | 593 | 4,290 |
| 10 to 19 | 297 | 3,619 |
| 20 and over | 197 | 7,932 |

The sheet showed label sleeves (Anjunabeats, Monstercat, NCS, V-Disc, wbur's Circle Round),
placeholders (Doctor Who's "image coming soon") and a few dark near-blanks at 10 or more artists.
The 5 to 9 band mixes in singles credited to their featured artists. Counting all credits also
caught one deadmau5 cover over 28 releases and 10 artists, a single with its remixers. So the
rule counts first credited artists, which gives the 323 pictures and 7,435 releases above.

## Where the list applies

- `build_index` leaves excluded covers out of the index and records the list's hash in
  `exclusions.txt`. Embeddings stay, so changing the list needs a rebuild but no re-embedding.
- The eval's gallery draw leaves excluded covers out of targets and distractors, drawing further
  along the same seeded permutation. The list's hash is part of the gallery hash.
