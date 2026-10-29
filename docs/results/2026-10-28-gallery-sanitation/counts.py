"""Counts for the sanitation report, from the written exclusion list.

    PYTHONPATH=. pipeline/.venv/bin/python docs/results/2026-10-28-gallery-sanitation/counts.py

Run `exclusions write` first. Prints covers and releases per reason, shared pictures, releases left
with no cover, and the labeled positives (gold and wild, every split) whose every cover is excluded.
"""
import numpy as np

from pipeline import config
from pipeline.gallery.exclusions import load_exclusions, load_measures
from pipeline.gold.labels import load_labels
from pipeline.metadata.catalog import Catalog

excluded = load_exclusions(config.GALLERY_DIR)
measures = load_measures(config.GALLERY_DIR)
keys = measures["key"]
mbids = keys.astype("U36")
out = np.isin(keys, list(excluded))
print(f"gallery: {len(keys):,} covers of {len(np.unique(mbids)):,} releases")
for reason in ("blank", "strip", "shared"):
    sel = np.isin(keys, [k for k, r in excluded.items() if r == reason])
    print(f"  {reason}: {sel.sum():,} covers of {len(np.unique(mbids[sel])):,} releases")
shared = np.isin(keys, [k for k, r in excluded.items() if r == "shared"])
print(f"  shared pictures: {len(np.unique(measures['phash'][shared])):,}")
kept = set(mbids[~out].tolist())
lost = set(mbids[out].tolist()) - kept
print(f"  total: {out.sum():,} covers; {len(lost):,} releases left with no cover")
catalog = Catalog()
for set_dir in (config.GOLD_DIR, config.WILD_DIR):
    positives = [lab for lab in load_labels(set_dir) if lab.kind == "positive"] if set_dir.exists() else []
    gone = []
    for lab in positives:
        covers = [c for m in catalog.same_album(lab.mbid) for c in catalog.covers(m)] if catalog.release(lab.mbid) else []
        if covers and all(c.file.split("/")[-1].removesuffix(".jpg") in excluded for c in covers):
            gone.append(lab)
    print(f"{set_dir.name}: {len(positives)} labeled positives, {len(gone)} with every cover of their album excluded"
          + "".join(f"\n  {lab.file} {lab.mbid}" for lab in gone))
