"""
build_index.py
---------------
Aggregates augmented .npz embedding shards into a single HNSW index.
Each augmented embedding shares the same UUID label.
"""

import glob
import numpy as np
import hnswlib
import json
import os

from embed_batch import OUT_DIR

DIM = 512

INDEX_PATH = os.path.join(OUT_DIR, "cover_hnsw.index")
META_PATH  = os.path.join(OUT_DIR, "index_metadata.json")


def main():
    shard_files = sorted(glob.glob(os.path.join(OUT_DIR, "*.npz")))
    total = sum(np.load(f, allow_pickle=True)["embs"].shape[0] for f in shard_files)
    print(f"Building index from {len(shard_files)} shards ({total:,} vectors total)")

    p = hnswlib.Index(space="cosine", dim=DIM)
    p.init_index(max_elements=total, ef_construction=200, M=48)

    meta = {}
    idx = 0
    for shard in shard_files:
        data = np.load(shard, allow_pickle=True)
        uuids = data["uuids"]
        embs = data["embs"].astype(np.float32)
        n = len(uuids)
        p.add_items(embs, np.arange(idx, idx + n))
        for i, u in enumerate(uuids):
            meta[str(idx + i)] = u.tolist() if isinstance(u, np.ndarray) else u
        idx += n
        print(f"Added {n} embeddings from {shard}")

    p.save_index(INDEX_PATH)
    with open(META_PATH, "w") as f:
        json.dump(meta, f)
    print(f"Saved index to {INDEX_PATH}, metadata to {META_PATH}")

if __name__ == "__main__":
    main()
