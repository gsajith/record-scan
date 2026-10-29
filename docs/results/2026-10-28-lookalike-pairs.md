# Templates, lookalike pairs and dedup groups from the pHash

Issue #68, replacing the chained clusters from #20. Code at `3610ea3`, run 2026-10-28 on the 256
`covers.npy` files from `2026-10-27-gallery-pass.md` and the catalog built from
`covers-2026-10-24.jsonl`. No drive.

```
PYTHONPATH=. pipeline/.venv/bin/python -m pipeline.gallery.cluster_duplicates   # --lookalike-distance 4
```

## The rule

A pHash records only coarse brightness layout, so a close hash marks a candidate and the catalog
decides. Each cover carries the artists of every release using its hash, Various Artists left out,
and two covers share an artist when those sets meet.

- A design joins hashes up to 2 bits apart with no artist check, and same-artist hashes up to N
  bits apart. A design whose releases span 5 or more release groups is a template: its picture
  names the design, not the release. That covers label stock sleeves and one artist's series
  (show downloads, numbered volumes) with one rule.
- A lookalike pair is two releases with covers up to N bits apart that share an artist, neither
  cover in a template. Pairs only, with no closure.
- A dedup group is a connected component of hashes up to 2 bits apart that share an artist, plus
  each release's own covers, template covers left out. A release with 3 or more covers stands
  alone (see Box sets).

N is 4, Gautham's pick (see Choosing N). The recipe sidecar beside each output reads
`series-designs-v2 template2-5release-groups dedup2 boxset3 lookalike4 catalog-c506a85dfac1bcee gallery-adc597dcc38e303f`.
`series-designs-v2` is `GROUP_RULE`, which names the logic and changes whenever `group()` would
write different outputs for the same numbers. The catalog fingerprint hashes every release's
mbid, artist ids and release group, and the gallery part hashes every prefix's done marker. The
loaders refuse any other recipe. They read the whole catalog to check it, 11 to 14 s.

## Full-gallery counts at N = 4

| | Value |
|---|---|
| covers | 3,977,195 |
| releases | 3,962,307 |
| time | 69 s, 6.8 GB peak memory |
| templates | 4,819 designs, 64,678 covers of 64,630 releases |
| lookalike pairs | 676,148 |
| dedup groups | 3,624,854 |
| `templates.npy`, `lookalikes.npy`, `clusters.npy` | 18 MB, 195 MB, 602 MB |

Lookalike pairs by distance in bits:

| 0 | 2 | 3 | 4 |
|---|---|---|---|
| 356,086 | 160,175 | 1 | 159,886 |

Dedup group sizes, in releases:

| Size | Groups | Releases |
|---|---|---|
| 1 | 3,381,544 | 3,381,544 |
| 2 | 191,974 | 383,948 |
| 3 | 32,422 | 97,266 |
| 4 | 10,379 | 41,516 |
| 5 to 9 | 7,583 | 45,037 |
| 10 to 99 | 952 | 12,996 |

The largest dedup groups hold 52, 46, 42, 41 and 40 releases, against 568,997 for the chained
clusters at 8 bits. No dedup group spans more than 4 release groups.

## Box sets

The coordinator's contact sheets of the largest dedup groups at N = 6 showed box sets bridging
albums. A release with many fronts joined all its covers' groups: a Glenn Gould group of 85
releases mixed Bach's WTC, the Goldberg Variations, Schoenberg, Beethoven and Mozart through the
Complete Original Jacket Collection, and a Doors group of 60 mixed The Doors, Waiting for the Sun
and Morrison Hotel through A Collection. 6 dedup groups spanned 5 or more release groups, and 24
groups spanning 3 or more held a release with 3 or more covers. The coordinator's fix: a release
with 3 or more covers (`BOX_SET_COVERS`) stands alone in `clusters.npy`. Its covers' hashes still
link other releases, and it still forms lookalike pairs. Without the rule, the run at N = 4 had
3,624,445 dedup groups, the largest 85, 60, 52, 46 and 43 releases.

## Choosing the template rule

The issue first counted distinct artist ids. Template counts on the full gallery under each rule
tried, at a lookalike distance of 8 unless noted:

| Template rule | Templates | Releases in them |
|---|---|---|
| 5+ distinct artist ids (as first written) | 29,761 | 67,041 |
| the same, but 2+ releases | 4,958 | 42,402 |
| 5+ distinct first credited non-VA artists | 1,429 | 30,897 |
| 5+ release groups, designs within 2 bits | 4,302 | 57,871 |
| 5+ release groups, plus same-artist links at N = 4 (shipped) | 4,819 | 64,630 |
| 5+ release groups, plus same-artist links at N = 6 | 5,240 | 71,572 |
| 5+ release groups, plus same-artist links at N = 8 | 5,701 | 78,739 |

Counting artist ids made 24,803 single releases templates: one album credited to a composer,
conductor, orchestra and soloists spans 5 ids alone, and its reissues then stopped pairing. The
coordinator chose first credited artists, which left out every single-release group. The
hand-check below then showed that one artist's series, not other artists' designs, made most of
the wrong pairs, so the coordinator replaced the artist count with release groups. A series is a
template for the same reason a stock sleeve is.

Each rule keeps small designs unflagged: a design over 2 to 4 release groups, such as Queen's
"Greatest Hits II" and "IV", stays out of the templates. #56's pixel-spread check still catches the
blank ones.

## Choosing N

Sheets of 50 random lookalike pairs at exactly each distance, each pair showing the two releases'
closest covers, made by `2026-10-28-lookalike-pairs/lookalike_sheets.py`. A pair is the same
picture if it is the same artwork, even rescanned, recolored for one album's edition, cropped or
with different sticker text. A different volume, show, episode or single in a shared design is
not, nor is a recolored remix single. A remix or acoustic version that keeps the single's artwork
unchanged is marked borderline and counted as the same picture. Every verdict, with the two mbids,
is in `2026-10-28-lookalike-pairs/verdicts/`. The sheets are covers, so they stay out of git, in
`~/Projects/record-scan-data/gallery/probe-68/`.

Same picture, of 50, with borderline pairs in brackets:

| Sample | Rule | 0 bits | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| pair-weighted | first credited artists | 43 | 34 (1) | 36 (2) | 36 | 32 |
| release-weighted | release groups, 2-bit designs | 46 | 47 (2) | 43 (2) | 40 | 36 (1) |
| release-weighted | release groups, same-artist links at N (N = distance) | | | 46 (2) | 45 | 44 (4) |

Pair-weighted sampling draws pairs uniformly, so a series of k releases weighs about k^2/2 and a
few big series fill the sample: LivePhish show downloads alone were 6 of 50 at 4 bits, with
Nippon Columbia's "Eternal 20xx" volumes, Rumonoise catalog codes and podcast episodes close
behind. Precision was flat from 2 to 8 bits, so the distance wasn't what failed. Release-weighted
sampling draws a release with a partner at that distance, then one of its partners, which is how
an eval query meets pairs. The coordinator chose it for picking N.

No distance reached 48 of 50, the bar in the issue. The coordinator's fallback, the largest N
with 45 or more, gave 6, at exactly 45. Gautham picked 4 instead (2026-10-28): 46 and 45 of 50
are the same within sampling noise, and 4 errs strict. The run at 6 gave 838,710 pairs against
676,148 at 4, and 5,240 templates over 71,572 releases.

Joining same-artist links at N into the designs took 8 bits from 36 to 44, 6 bits from 40 to 45
and 4 bits from 43 to 46. The misses left at 6 and 8 bits are different singles, episodes and
volumes on one design spanning fewer than 5 release groups, or recolored remix singles.

## Scripts

- `lookalike_sheets.py <out dir> pair|release <distance>...` draws a sample set and its sheets
  from the current `lookalikes.npy`. Run `cluster_duplicates` at the rule and N under test first.
  The pair-weighted sets came from the first-artist rule at `6cbb98f`, the 2-bit-design sets from
  `3d6fef9` at N = 8, and the same-artist-link sets from `52e2c50` at N = 4, 6 and 8. Each set's
  verdict file lists its pairs, so it doesn't depend on rerunning the draw.
