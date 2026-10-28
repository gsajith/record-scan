"""Group releases whose covers are the same picture: reissues, regional pressings, deluxe editions.

    python -m pipeline.gallery.cluster_duplicates [--max-distance 8]

Reads every <prefix>.covers.npy in config.GALLERY_DIR (written by shard_and_hash) and joins two
covers when their pHashes match exactly or differ in at most --max-distance bits. Clusters are of
releases: a release's extra fronts are separate covers, and the release joins every cluster its
covers fall in. Joins are transitive, so A near B near C puts all three in one cluster. Writes
config.GALLERY_DIR/clusters.npy, one row per release (mbid, cluster_id, cluster_size in releases),
and prints the release count, the cluster count and a histogram of cluster sizes.

Needs no drive. Refuses a gallery whose prefixes were hashed with different recipes, or whose
prefixes aren't done, since their hashes can't be compared.
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pipeline import config
from pipeline.gallery.shard_and_hash import hash_recipe

CLUSTERS_DTYPE = np.dtype([("mbid", "U36"), ("cluster_id", np.uint32), ("cluster_size", np.uint32)])
# The near-duplicate search needs this many of its blocks to match exactly (see near_pairs)
MATCHING_BLOCKS = 4


def cluster(covers_files: list[Path], max_distance: int) -> dict[str, int]:
    """{mbid: cluster_id} for every release in the covers files. Covers whose pHashes are within
    max_distance bits are joined, as are all covers of one release; clusters are the connected
    components. Ids run from 0 and are the same on every run over the same files."""
    _check_recipes(covers_files)
    rows = np.concatenate([np.load(f) for f in covers_files])
    mbids, mbid_of_row = np.unique(rows["key"].astype("U36"), return_inverse=True)  # mbid is key[:36]
    hashes, hash_of_row = np.unique(rows["phash"], return_inverse=True)  # identical hashes share a node
    near_a, near_b = near_pairs(hashes, max_distance)
    # Nodes are the distinct hashes, then the releases. Each cover links its hash to its release.
    n_hashes = len(hashes)
    src = np.concatenate([near_a, hash_of_row])
    dst = np.concatenate([near_b, n_hashes + mbid_of_row])
    size = n_hashes + len(mbids)
    graph = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(size, size))
    _, labels = connected_components(graph, directed=False)
    # Every hash belongs to some release, so the releases' labels already run 0..count-1
    return dict(zip(mbids.tolist(), labels[n_hashes:].tolist()))


def near_pairs(hashes: np.ndarray, max_distance: int) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (i < j by position in hashes) of distinct hashes at most max_distance bits apart.

    Multi-index hashing by pigeonhole: split the 64 bits into max_distance + 4 blocks. Two hashes
    within max_distance bits differ in at most max_distance blocks, so they match exactly on at
    least 4. One table per choice of 4 blocks finds every pair that matches on those, and a pair is
    kept only in its first table (its lowest 4 matching blocks), so no pair is returned twice.
    """
    empty = np.empty(0, dtype=np.intp)
    if max_distance == 0:
        return empty, empty
    n_blocks = max_distance + MATCHING_BLOCKS
    if n_blocks > 64:
        raise ValueError(f"max_distance {max_distance} is too large for 64-bit hashes")
    block_masks = [np.uint64(sum(1 << b for b in bits)) for bits in np.array_split(np.arange(64), n_blocks)]
    found_a, found_b = [empty], [empty]
    for table in itertools.combinations(range(n_blocks), MATCHING_BLOCKS):
        mask = np.uint64(0)
        for b in table:
            mask |= block_masks[b]
        a, b = _same_key_pairs(hashes & mask)
        diff = hashes[a] ^ hashes[b]
        keep = np.bitwise_count(diff) <= max_distance
        # A pair that also matches on a block before this table's last, outside the table, has an
        # earlier table, which kept it
        for skipped in set(range(table[-1])) - set(table):
            keep &= (diff & block_masks[skipped]) != 0
        found_a.append(a[keep])
        found_b.append(b[keep])
    return np.concatenate(found_a), np.concatenate(found_b)


def _same_key_pairs(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every index pair (i, j), i != j, with keys[i] == keys[j], each pair once."""
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    bounds = np.concatenate([[0], np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1, [len(keys)]])
    run_end = np.repeat(bounds[1:], np.diff(bounds))  # for each sorted position, the end of its run
    later = run_end - np.arange(len(keys)) - 1  # how many positions after it share its key
    first = np.repeat(np.arange(len(keys)), later)
    # The k-th pair of position i is (i, i + k + 1)
    step = np.arange(len(first)) - np.repeat(np.cumsum(later) - later, later) + 1
    return order[first], order[first + step]


def _check_recipes(covers_files: list[Path]) -> None:
    by_recipe: dict[str, list[str]] = {}
    for path in covers_files:
        prefix = path.name.removesuffix(".covers.npy")
        if not (path.parent / f"{prefix}.done").exists():
            raise ValueError(f"Prefix {prefix} isn't done; rerun shard_and_hash before clustering")
        by_recipe.setdefault(hash_recipe(prefix, path.parent), []).append(prefix)
    if len(by_recipe) > 1:
        found = "; ".join(f"{recipe}: {' '.join(prefixes)}" for recipe, prefixes in by_recipe.items())
        raise ValueError(f"Prefixes were hashed with different recipes, so their hashes can't be compared: {found}")
