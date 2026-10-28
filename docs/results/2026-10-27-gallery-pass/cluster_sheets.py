"""Contact sheets: 16 random pairs at pHash distance 2, 4, 6 and 8, and 32 members of the largest
component at <= 4 and <= 8. Run cluster_thresholds.py first with the same <out dir>.

    PYTHONPATH=. pipeline/.venv/bin/python docs/results/2026-10-27-gallery-pass/cluster_sheets.py <out dir>

The sheets are made of covers, so they stay out of git.
"""
import io, tarfile, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pipeline import config
S = sys.argv[1]
files = sorted(config.GALLERY_DIR.glob("*.covers.npy"))
rows = np.concatenate([np.load(f) for f in files])
hashes, first_row = np.unique(rows["phash"], return_index=True)
p = np.load(S + "/pairs.npz"); a, b, dist = p["a"], p["b"], p["dist"]
rng = np.random.default_rng(1)
tars = {}
def img(h_idx, size=160):
    key = str(rows["key"][first_row[h_idx]])
    prefix = key[:2]
    t = tars.get(prefix) or tars.setdefault(prefix, tarfile.open(config.GALLERY_DIR / f"{prefix}.tar"))
    with Image.open(io.BytesIO(t.extractfile(f"{key}.jpg").read())) as im:
        return im.convert("RGB").resize((size, size))
def sheet(pairs, name, title):
    W = 160
    out = Image.new("RGB", (W * 2 * 4 + 30, (W + 4) * ((len(pairs) + 3) // 4) + 20), "white")
    d = ImageDraw.Draw(out); d.text((4, 4), title, fill="black")
    for n, (i, j) in enumerate(pairs):
        x, y = (n % 4) * (2 * W + 10), 20 + (n // 4) * (W + 4)
        out.paste(img(i), (x, y)); out.paste(img(j), (x + W, y))
    out.save(f"{S}/{name}.jpg", quality=80)
normal = np.bitwise_count(hashes) > 20  # skip the blank 0x0 and 0x8000... hashes
for d in (2, 4, 6, 8):
    idx = np.flatnonzero((dist == d) & normal[a] & normal[b])
    pick = rng.choice(idx, 16, replace=False)
    sheet([(a[k], b[k]) for k in pick], f"pairs-d{d}", f"16 random pairs at distance {d}")
# members of the giant component at d<=8, and at d<=4
H = len(hashes)
for d in (4, 8):
    k = dist <= d
    g = coo_matrix((np.ones(k.sum(), np.int8), (a[k], b[k])), shape=(H, H))
    _, lab = connected_components(g, directed=False)
    big = np.bincount(lab).argmax()
    members = np.flatnonzero(lab == big)
    pick = rng.choice(members, 32, replace=False)
    sheet([(pick[2*n], pick[2*n+1]) for n in range(16)], f"giant-d{d}", f"32 random members of the largest component at d<={d}")
    vals, cnt = np.unique(hashes[members], return_counts=True)
    print(d, len(members), "mean popcount", np.bitwise_count(hashes[members]).mean())
