"""Contact sheets of 50 random lookalike pairs at pHash distance 4, 6 and 8, then 2 and 0, for hand-checking N.

    PYTHONPATH=. pipeline/.venv/bin/python docs/results/2026-10-28-lookalike-pairs/lookalike_sheets.py <out dir>

Run cluster_duplicates --lookalike-distance 8 first. Each sheet holds 25 numbered pairs, and each
pair shows the two releases' closest covers. <out dir>/pairs-d<distance>.tsv lists the pairs in
sheet order. The sheets are made of covers, so they stay out of git.
"""
import io
import sys
import tarfile

import numpy as np
from PIL import Image, ImageDraw

from pipeline import config

out = sys.argv[1]
rows = np.concatenate([np.load(f) for f in sorted(config.GALLERY_DIR.glob("*.covers.npy"))])
mbid_of_row = rows["key"].astype("U36")
order = np.argsort(mbid_of_row, kind="stable")
sorted_mbids = mbid_of_row[order]
pairs = np.load(config.GALLERY_DIR / "lookalikes.npy")
rng = np.random.default_rng(68)
tars = {}


def covers_of(mbid):
    lo, hi = np.searchsorted(sorted_mbids, mbid), np.searchsorted(sorted_mbids, mbid, side="right")
    return rows[order[lo:hi]]


def closest(a, b):
    """The two releases' covers whose hashes are closest."""
    ca, cb = covers_of(a), covers_of(b)
    d = np.bitwise_count(ca["phash"][:, None] ^ cb["phash"][None, :])
    i, j = np.unravel_index(d.argmin(), d.shape)
    return str(ca["key"][i]), str(cb["key"][j])


def image(key, size=180):
    prefix = key[:2]
    tar = tars.get(prefix) or tars.setdefault(prefix, tarfile.open(config.GALLERY_DIR / f"{prefix}.tar"))
    with Image.open(io.BytesIO(tar.extractfile(f"{key}.jpg").read())) as im:
        return im.convert("RGB").resize((size, size))


W = 180
for d in (4, 6, 8, 2, 0):  # 2 and 0 added after 4, 6 and 8 all fell short, keeping their picks
    pick = rng.choice(np.flatnonzero(pairs["distance"] == d), 50, replace=False)
    lines = []
    for sheet in range(2):
        canvas = Image.new("RGB", (5 * (2 * W + 16), 5 * (W + 22)), "white")
        draw = ImageDraw.Draw(canvas)
        for n, k in enumerate(pick[sheet * 25:(sheet + 1) * 25]):
            a, b = closest(str(pairs["mbid_a"][k]), str(pairs["mbid_b"][k]))
            number = sheet * 25 + n + 1
            lines.append(f"{number}\t{pairs['mbid_a'][k]}\t{pairs['mbid_b'][k]}\t{a}\t{b}")
            x, y = (n % 5) * (2 * W + 16), (n // 5) * (W + 22)
            draw.text((x + 2, y + 4), f"{number}", fill="black")
            canvas.paste(image(a), (x, y + 20))
            canvas.paste(image(b), (x + W + 2, y + 20))
        canvas.save(f"{out}/pairs-d{d}-{sheet + 1}.jpg", quality=85)
    with open(f"{out}/pairs-d{d}.tsv", "w") as f:
        f.write("pair\tmbid_a\tmbid_b\tcover_a\tcover_b\n" + "\n".join(lines) + "\n")
    print(f"distance {d}: {int((pairs['distance'] == d).sum()):,} pairs, sheets written")
