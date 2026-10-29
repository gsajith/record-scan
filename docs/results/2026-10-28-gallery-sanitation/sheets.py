"""Contact sheets of the suspect buckets for #56, so Gautham picks the cutoffs by eye.

    PYTHONPATH=. pipeline/.venv/bin/python docs/results/2026-10-28-gallery-sanitation/sheets.py <out dir> [page...]

Run cluster_duplicates and `exclusions measure` first. Writes one self-contained HTML page per
bucket to <out dir> (thumbnails inlined) and prints the counts per band over the full gallery.
The pages are made of covers, so they stay out of git.

- templates-multi.html, templates-single.html: #68's templates with 2+ artists and with 0 or 1,
  biggest first (the top 300 by releases), then 100 random from the rest. Each card shows up to
  4 covers, the counts and up to 8 release titles.
- blank.html: covers by spread band, 100 random per band.
- strip.html: covers by aspect ratio (long side over short) beyond 2.5, 100 random per band.
- sleeve.html: aspect ratio 1.8 to 2.5, often a front and back side by side, 100 random per band.
- shared.html: pictures with one exact pHash over 5+ release groups and 5+ artists, most artists
  first (the top 300), then 100 random from the rest, with counts per artist band.

Name pages (blank strip sleeve templates shared) to write only those.
"""
import base64
import html
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline import config
from pipeline.gallery.cluster_duplicates import VARIOUS_ARTISTS, load_templates, read_catalog
from pipeline.gallery.exclusions import load_measures
from pipeline.metadata.catalog import Catalog

out = Path(sys.argv[1])
pages = set(sys.argv[2:]) or {"blank", "strip", "sleeve", "templates", "shared"}
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(56)
catalog = Catalog()
tars: dict[str, tarfile.TarFile] = {}
measures = load_measures(config.GALLERY_DIR)
by_key = {k: i for i, k in enumerate(measures["key"].tolist())}
templates = load_templates(config.GALLERY_DIR / "templates.npy")
long_side = np.maximum(measures["width"], measures["height"]).astype(np.float64)
short_side = np.minimum(measures["width"], measures["height"]).astype(np.float64)
aspect = long_side / np.maximum(short_side, 1)

SPREAD_BANDS = [(0, 1), (1, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 16)]
STRIP_BANDS = [(2.5, 3), (3, 4), (4, 6), (6, np.inf)]
SLEEVE_BANDS = [(1.8, 1.95), (1.95, 2.05), (2.05, 2.5)]


def thumb(key: str, size: int = 150) -> str:
    prefix = key[:2]
    tar = tars.get(prefix) or tars.setdefault(prefix, tarfile.open(config.GALLERY_DIR / f"{prefix}.tar"))
    with Image.open(io.BytesIO(tar.extractfile(f"{key}.jpg").read())) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def describe(key: str) -> str:
    release = catalog.release(key[:36])
    return f"{', '.join(release.artists)} - {release.title}" if release else "(not in catalog)"


def page(title: str, intro: str, sections: list[str]) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light dark; --bg: #fff; --fg: #111; --muted: #666; --line: #ddd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #111; --fg: #eee; --muted: #999; --line: #333; }} }}
body {{ background: var(--bg); color: var(--fg); font: 13px/1.4 system-ui, sans-serif; margin: 16px; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
figure {{ margin: 0; width: 150px; }}
img {{ max-width: 150px; max-height: 150px; display: block; background: var(--line); }}
figcaption, .meta {{ color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }}
.card {{ display: flex; gap: 12px; padding: 10px 0; border-bottom: 1px solid var(--line); flex-wrap: wrap; }}
.card .covers {{ display: flex; gap: 6px; }}
.card .text {{ min-width: 260px; flex: 1; }}
</style></head><body><h1>{html.escape(title)}</h1><p>{intro}</p>{''.join(sections)}</body></html>"""


def cover_figure(i: int) -> str:
    key = str(measures["key"][i])
    caption = (f"{measures['width'][i]}x{measures['height'][i]}, spread {measures['spread'][i]:.1f}<br>"
               f"{html.escape(describe(key))}<br>{key}")
    return f'<figure><img src="{thumb(key)}" alt=""><figcaption>{caption}</figcaption></figure>'


def band_sections(values: np.ndarray, bands: list[tuple[float, float]], name: str) -> tuple[list[str], list[str]]:
    sections, counts = [], []
    for lo, hi in bands:
        idx = np.flatnonzero((values >= lo) & (values < hi))
        releases = len(np.unique(measures["key"][idx].astype("U36")))
        counts.append(f"  {name} {lo} to {hi}: {len(idx):,} covers of {releases:,} releases")
        sample = np.sort(rng.choice(idx, min(100, len(idx)), replace=False)) if len(idx) else idx
        figures = "".join(cover_figure(int(i)) for i in sample)
        sections.append(f"<h2>{name} {lo} to {hi}: {len(idx):,} covers, {releases:,} releases "
                        f"({len(sample)} shown)</h2><div class='grid'>{figures}</div>")
    return sections, counts


def write_bands(file: str, title: str, intro: str, values, bands, name):
    sections, counts = band_sections(values, bands, name)
    (out / file).write_text(page(title, intro, sections))
    print(f"{title}:\n" + "\n".join(counts))


if "blank" in pages:
    write_bands("blank.html", "Near-uniform covers", "Spread is the RGB standard deviation of a 64x64 copy "
                "(0 to 255). 100 random covers per band.", measures["spread"], SPREAD_BANDS, "spread")
if "strip" in pages:
    write_bands("strip.html", "Strips and spines", "Aspect ratio is the long side over the short side. "
                "100 random covers per band.", aspect, STRIP_BANDS, "aspect")
if "sleeve" in pages:
    write_bands("sleeve.html", "About 2:1", "Often a front and back side by side. 100 random covers per band.",
                aspect, SLEEVE_BANDS, "aspect")

ids, first = np.unique(templates["template_id"], return_index=True)
releases, artists = templates["releases"][first], templates["artists"][first]
groups_of = templates["release_groups"][first]
order_of = np.argsort(templates["template_id"], kind="stable")
bounds = np.searchsorted(templates["template_id"][order_of], ids)
bounds = np.append(bounds, len(templates))


def template_card(t: int) -> str:
    rows = templates[order_of[bounds[t]:bounds[t + 1]]]
    keys = rows["key"].tolist()
    shown = rng.choice(len(keys), min(4, len(keys)), replace=False)
    spreads = measures["spread"][[by_key[k] for k in keys]]
    covers = "".join(f'<img src="{thumb(keys[i])}" alt="">' for i in shown)
    titles = sorted({describe(k) for k in rng.choice(keys, min(30, len(keys)), replace=False).tolist()})[:8]
    text = (f"<b>template {ids[t]}</b>: {releases[t]:,} releases, {groups_of[t]:,} release groups, "
            f"{artists[t]:,} artists, median spread {np.median(spreads):.1f}"
            f"<div class='meta'>{'<br>'.join(html.escape(s) for s in titles)}</div>")
    return f"<div class='card'><div class='covers'>{covers}</div><div class='text'>{text}</div></div>"


for file, title, mask in [] if "templates" not in pages else (("templates-multi.html", "Templates with 2+ artists", artists >= 2),
                          ("templates-single.html", "Templates with 0 or 1 artist", artists <= 1)):
    chosen = np.flatnonzero(mask)
    biggest = chosen[np.argsort(-releases[chosen], kind="stable")]
    top, rest = biggest[:300], biggest[300:]
    tail = np.sort(rng.choice(rest, min(100, len(rest)), replace=False)) if len(rest) else rest
    tail = tail[np.argsort(-releases[tail], kind="stable")]
    intro = (f"{len(chosen):,} templates over {int(releases[chosen].sum()):,} releases. "
             f"The {len(top)} biggest by releases, then {len(tail)} random from the other {len(rest):,}.")
    sections = [f"<h2>Biggest {len(top)}</h2>" + "".join(template_card(int(t)) for t in top),
                f"<h2>Random {len(tail)} from the rest</h2>" + "".join(template_card(int(t)) for t in tail)]
    (out / file).write_text(page(title, intro, sections))
    print(f"{title}: {intro}")
    for lo, hi in ((0, 1), (2, 4), (5, 19), (20, 10**9)):
        if (sel := mask & (artists >= lo) & (artists <= hi)).any():
            print(f"  {lo} to {hi} artists: {int(sel.sum()):,} templates, {int(releases[sel].sum()):,} releases")


if "shared" in pages:
    artists_of, release_group_of, _ = read_catalog(config.CATALOG_DB)
    rows = np.concatenate([np.load(f) for f in sorted(config.GALLERY_DIR.glob("*.covers.npy"))])
    hashes, of_row, counts = np.unique(rows["phash"], return_inverse=True, return_counts=True)
    order = np.argsort(of_row, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)])
    shared = []  # (artists, release groups, releases, keys)
    for h in np.flatnonzero(counts >= 5):
        keys = rows["key"][order[starts[h]:starts[h + 1]]].tolist()
        mbids = {k[:36] for k in keys}
        artist_ids = {a for m in mbids for a in artists_of.get(m, ()) if a != VARIOUS_ARTISTS}
        groups = {release_group_of.get(m) or m for m in mbids}
        if len(groups) >= 5 and len(artist_ids) >= 5:
            shared.append((len(artist_ids), len(groups), len(mbids), keys))
    shared.sort(key=lambda r: -r[0])
    print("Shared pictures (one exact pHash over 5+ release groups and 5+ artists):")
    for lo, hi in ((5, 9), (10, 19), (20, 10**9)):
        band = [r for r in shared if lo <= r[0] <= hi]
        print(f"  {lo} to {hi} artists: {len(band):,} pictures, {sum(r[2] for r in band):,} releases")

    def shared_card(n: int, r) -> str:
        artist_count, group_count, release_count, keys = r
        covers = "".join(f'<img src="{thumb(k)}" alt="">' for k in rng.choice(keys, min(3, len(keys)), replace=False))
        titles = sorted({describe(k) for k in rng.choice(keys, min(30, len(keys)), replace=False).tolist()})[:8]
        spreads = measures["spread"][[by_key[k] for k in keys]]
        text = (f"<b>{n}</b>: {release_count:,} releases, {group_count:,} release groups, {artist_count:,} artists, "
                f"median spread {np.median(spreads):.1f}<div class='meta'>{'<br>'.join(html.escape(t) for t in titles)}</div>")
        return f"<div class='card'><div class='covers'>{covers}</div><div class='text'>{text}</div></div>"

    top, rest = shared[:300], shared[300:]
    tail = sorted((rest[i] for i in rng.choice(len(rest), min(100, len(rest)), replace=False)), key=lambda r: -r[0])
    intro = (f"{len(shared):,} pictures over {sum(r[2] for r in shared):,} releases. The {len(top)} with the most "
             f"artists, then {len(tail)} random from the other {len(rest):,}.")
    sections = [f"<h2>Most artists</h2>" + "".join(shared_card(n + 1, r) for n, r in enumerate(top)),
                f"<h2>Random {len(tail)} from the rest</h2>" + "".join(shared_card(301 + n, r) for n, r in enumerate(tail))]
    (out / "shared.html").write_text(page("Shared pictures", intro, sections))
