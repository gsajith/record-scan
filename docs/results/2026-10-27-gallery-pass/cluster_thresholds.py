"""Near pairs of the gallery's distinct pHashes by distance, and component sizes at each threshold.

    PYTHONPATH=. pipeline/.venv/bin/python docs/results/2026-10-27-gallery-pass/cluster_thresholds.py <out dir>

Writes <out dir>/pairs.npz for cluster_sheets.py. Prints the lines quoted in ../2026-10-27-gallery-pass.md.
"""
import sys
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pipeline import config
from pipeline.gallery import cluster_duplicates as cd
S = sys.argv[1] + "/"
files = sorted(config.GALLERY_DIR.glob("*.covers.npy"))
rows = np.concatenate([np.load(f) for f in files])
hashes, hash_of_row, counts = np.unique(rows["phash"], return_inverse=True, return_counts=True)
mbids, mbid_of_row = np.unique(rows["key"].astype("U36"), return_inverse=True)
t = time.monotonic()
a, b = cd.near_pairs(hashes, 8)
dist = np.bitwise_count(hashes[a] ^ hashes[b]).astype(np.int8)
print("pairs", len(a), f"{time.monotonic()-t:.0f}s", "by distance", np.bincount(dist))
np.savez(S + "pairs.npz", a=a, b=b, dist=dist)
H, M = len(hashes), len(mbids)
def comps(d, with_mbid):
    k = dist <= d
    src, dst = a[k], b[k]
    if with_mbid:
        src = np.concatenate([src, hash_of_row]); dst = np.concatenate([dst, H + mbid_of_row]); n = H + M
    else:
        n = H
    g = coo_matrix((np.ones(len(src), np.int8), (src, dst)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return lab
for d in (0, 2, 4, 5, 6, 7, 8):
    for wm in (False, True):
        lab = comps(d, wm)
        if wm:
            sizes = np.bincount(lab[H:])  # releases
        else:
            sizes = np.bincount(lab, weights=counts).astype(int)  # covers
        print(f"d<={d} mbid_bridges={wm}: comps {np.count_nonzero(sizes)} largest {np.sort(sizes)[::-1][:4]}")
# exact-hash top counts
top = np.argsort(counts)[::-1][:10]
print("most repeated hashes:", [(hex(int(hashes[i])), int(counts[i])) for i in top])
