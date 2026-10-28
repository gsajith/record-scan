"""Build an HNSW index over a model's embedding shards.

    python -m pipeline.retrieval.build_index --model NAME [--M 32] [--ef-construction 200]

Reads config.EMB_DIR/<model>/<prefix>.npz and writes index.hnsw, labels.npy (row id ->
mbid, U36) and signature.txt beside them. Every shard must carry the same embedder
signature (see embedders.embedder_signature); shards from before signatures need
re-embedding. It refuses to run while any shard is behind its .done marker or the catalog's
current cover list for that prefix: run embed --all first. Prefixes with covers but no shard
yet are left out with a warning, so a few embedded prefixes still make a test index.
"""

import argparse
import os
from collections.abc import Callable
from pathlib import Path

import hnswlib
import numpy as np

from pipeline import config
from pipeline.gallery.shard_and_hash import PREFIXES
from pipeline.metadata.catalog import Catalog
from pipeline.retrieval.embed import is_done

SHARD_GLOB = "[0-9a-f][0-9a-f].npz"  # skips the .tmp.npz an interrupted embed run leaves


def build_index(shards: list[Path], dim: int, out_dir: Path, M: int, ef_construction: int) -> int:
    """Index every row of every shard under cosine distance. Returns the row count."""
    if not shards:
        raise SystemExit(f"No embedding shards to index in {out_dir}")
    parts = []
    signatures = set()
    for shard in shards:
        with np.load(shard) as data:
            if "signature" not in data:
                raise SystemExit(f"{shard} predates embedder signatures. Re-embed it.")
            signatures.add(str(data["signature"]))
            uuids, embs = data["uuids"].astype("U36"), data["embs"]
        if embs.ndim != 2 or embs.shape[1] != dim or len(uuids) != len(embs):
            raise SystemExit(f"{shard} holds {embs.shape} vectors for {len(uuids)} ids, expected dim {dim}")
        if len(uuids):  # a prefix with no covers; hnswlib fails on zero rows when they come first
            parts.append((uuids, embs))
    if not parts:
        raise SystemExit(f"The shards in {out_dir} hold no vectors")
    if len(signatures) > 1:
        raise SystemExit(f"Shards in {out_dir} come from different embedders, by preprocessing, weights or library ({', '.join(sorted(signatures))}). "
                         "Re-embed the old ones.")
    labels = np.concatenate([uuids for uuids, _ in parts])

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(labels), ef_construction=ef_construction, M=M)
    start = 0
    for uuids, embs in parts:
        index.add_items(embs.astype(np.float32), np.arange(start, start + len(uuids)))
        start += len(uuids)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Drop the old signature first: a crash mid-swap then leaves an index identify refuses
    (out_dir / "signature.txt").unlink(missing_ok=True)
    index.save_index(str(out_dir / "index.hnsw.tmp"))
    with (out_dir / "labels.npy.tmp").open("wb") as f:  # a file handle stops numpy from appending .npy
        np.save(f, labels)
    os.replace(out_dir / "index.hnsw.tmp", out_dir / "index.hnsw")
    os.replace(out_dir / "labels.npy.tmp", out_dir / "labels.npy")
    (out_dir / "signature.txt").write_text(signatures.pop())
    return len(labels)


def stale_shards(shards: list[Path], cover_files: Callable[[str], list[str]]) -> list[str]:
    """Prefixes whose shard isn't done, by its own embedder, from the cover list cover_files gives now."""
    stale = []
    for shard in shards:
        with np.load(shard) as data:
            signature = str(data["signature"]) if "signature" in data else ""
        if not is_done(shard.stem, cover_files(shard.stem), shard.parent, signature):
            stale.append(shard.stem)
    return stale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="shard folder name under the embeddings dir")
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    args = parser.parse_args()

    emb_dir = config.EMB_DIR / args.model
    shards = sorted(emb_dir.glob(SHARD_GLOB))
    catalog = Catalog()
    if stale := stale_shards(shards, catalog.cover_files):
        raise SystemExit(f"Shards {' '.join(stale)} in {emb_dir} are behind the catalog or unfinished. "
                         f"Run embed --model {args.model} --all first.")
    if missing := [p for p in PREFIXES if not (emb_dir / f"{p}.npz").exists() and catalog.cover_files(p)]:
        print(f"Warning: {len(missing)} prefixes with covers have no shard yet, so the index is partial. "
              f"Run embed --model {args.model} --all for the full catalog.")
    dim = np.load(shards[0])["embs"].shape[1] if shards else 0
    n = build_index(shards, dim, emb_dir, args.M, args.ef_construction)
    print(f"Indexed {n:,} vectors ({dim}-d) from {len(shards)} shards into {emb_dir / 'index.hnsw'}")


if __name__ == "__main__":
    main()
