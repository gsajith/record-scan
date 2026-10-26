"""Build an HNSW index over a model's embedding shards.

    python -m pipeline.retrieval.build_index --model NAME [--M 32] [--ef-construction 200]

Reads config.EMB_DIR/<model>/<prefix>.npz and writes index.hnsw and labels.npy (row id ->
mbid, U36) beside them. NAME is the shard folder, so the 2025-11 clip-vit-b-32-aug13 shards
build too.
"""

import argparse
import os
from pathlib import Path

import hnswlib
import numpy as np

from pipeline import config

SHARD_GLOB = "[0-9a-f][0-9a-f].npz"  # skips the .tmp.npz an interrupted embed run leaves


def build_index(shards: list[Path], dim: int, out_dir: Path, M: int, ef_construction: int) -> int:
    """Index every row of every shard under cosine distance. Returns the row count."""
    if not shards:
        raise SystemExit(f"No embedding shards to index in {out_dir}")
    parts = []
    for shard in shards:
        # The 2025-11 shards store uuids as an object array, which needs pickle to load.
        # Safe here: shards are local files our own embed runs wrote under RS_DATA_DIR.
        with np.load(shard, allow_pickle=True) as data:
            uuids, embs = data["uuids"].astype("U36"), data["embs"]
        if embs.ndim != 2 or embs.shape[1] != dim or len(uuids) != len(embs):
            raise SystemExit(f"{shard} holds {embs.shape} vectors for {len(uuids)} ids, expected dim {dim}")
        parts.append((uuids, embs))
    labels = np.concatenate([uuids for uuids, _ in parts])

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(labels), ef_construction=ef_construction, M=M)
    start = 0
    for uuids, embs in parts:
        index.add_items(embs.astype(np.float32), np.arange(start, start + len(uuids)))
        start += len(uuids)

    out_dir.mkdir(parents=True, exist_ok=True)
    index.save_index(str(out_dir / "index.hnsw.tmp"))
    with (out_dir / "labels.npy.tmp").open("wb") as f:  # a file handle stops numpy from appending .npy
        np.save(f, labels)
    os.replace(out_dir / "index.hnsw.tmp", out_dir / "index.hnsw")
    os.replace(out_dir / "labels.npy.tmp", out_dir / "labels.npy")
    return len(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="shard folder name under the embeddings dir")
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    args = parser.parse_args()

    emb_dir = config.EMB_DIR / args.model
    shards = sorted(emb_dir.glob(SHARD_GLOB))
    dim = np.load(shards[0], allow_pickle=True)["embs"].shape[1] if shards else 0
    n = build_index(shards, dim, emb_dir, args.M, args.ef_construction)
    print(f"Indexed {n:,} vectors ({dim}-d) from {len(shards)} shards into {emb_dir / 'index.hnsw'}")


if __name__ == "__main__":
    main()
