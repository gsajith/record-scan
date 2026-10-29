"""Find covers that are one picture or one design, from their perceptual hashes.

    python -m pipeline.gallery.cluster_duplicates [--lookalike-distance N]

Reads every <prefix>.covers.npy in config.GALLERY_DIR (written by shard_and_hash) and the releases'
artist ids and release groups from the catalog. A pHash records only coarse brightness layout, so a close hash marks a
candidate, and an artist the two releases share decides. Each cover carries the artists of every
release using its hash, Various Artists left out. Writes three outputs to config.GALLERY_DIR:

- templates.npy: shared designs, such as label stock sleeves, series frames, one artist's show
  downloads or numbered volumes, and near-blank covers. A design joins hashes up to
  TEMPLATE_DISTANCE bits apart, with no artist check, and same-artist hashes up to N bits apart;
  a design whose releases span TEMPLATE_RELEASE_GROUPS or more release groups is a template,
  since its picture names the design, not the release. One row per cover: key, template_id, and
  the design's release, release group and artist counts.
- lookalikes.npy: pairs of releases with covers up to N bits apart that share an artist, neither
  cover in a template. Pairs only: A near B and B near C don't pair A with C. One row per pair:
  mbid_a < mbid_b and the smallest distance between their covers.
- clusters.npy: one row per release (mbid, cluster_id, cluster_size in releases). Connected
  components of hashes up to DEDUP_DISTANCE bits apart that share an artist, plus each release's
  own covers, template covers left out. One vector per cluster in the index.

Each has a .recipe beside it naming the rule, the catalog and the gallery it came from, and the
loaders refuse a stale one. Needs no drive. Refuses a gallery whose prefixes were hashed with
different recipes, or whose prefixes aren't done, since their hashes can't be compared.
"""

import argparse
import hashlib
import itertools
import json
import os
import sqlite3
import sys
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pipeline import config
from pipeline.gallery.shard_and_hash import hash_recipe

CLUSTERS_DTYPE = np.dtype([("mbid", "U36"), ("cluster_id", np.uint32), ("cluster_size", np.uint32)])
LOOKALIKES_DTYPE = np.dtype([("mbid_a", "U36"), ("mbid_b", "U36"), ("distance", np.uint8)])
TEMPLATES_DTYPE = np.dtype([("key", "U64"), ("template_id", np.uint32), ("releases", np.uint32),
                            ("release_groups", np.uint32), ("artists", np.uint32)])
# Hashes this close are one design, as are same-artist hashes within the lookalike distance, and a
# design over this many release groups is a template: the picture names the design, not the
# release. That covers one label's stock sleeve and one artist's series (show downloads, numbered
# volumes) alike.
TEMPLATE_DISTANCE = 2
TEMPLATE_RELEASE_GROUPS = 5
# Same-artist hashes this close are one cover in the index
DEDUP_DISTANCE = 2
# Same-artist hashes this close are lookalikes, a pair at a time. Gautham's pick: 4 and 6 scored 46 and
# 45 of 50 hand-checked pairs, the same within sampling noise, and 4 errs strict
# (docs/results/2026-10-28-lookalike-pairs.md).
LOOKALIKE_DISTANCE = 4
# Names group()'s logic in the recipe. Change it whenever group() would write different outputs
# from the same input and numbers above, so old outputs are refused, as HASH_RECIPE does for hashes.
GROUP_RULE = "series-designs-v1"
# On compilations, so sharing it says nothing about two covers
VARIOUS_ARTISTS = "89ad4ac3-39f7-470e-963a-56509c546377"
# The near-duplicate search needs this many of its blocks to match exactly (see near_pairs)
MATCHING_BLOCKS = 4
SIZE_BUCKETS = [("1 release", 1, 1), ("2 releases", 2, 2), ("3 releases", 3, 3), ("4 releases", 4, 4),
                ("5 to 9 releases", 5, 9), ("10 to 99 releases", 10, 99), ("100 to 999 releases", 100, 999),
                ("1,000 or more releases", 1000, sys.maxsize)]


@dataclass(frozen=True)
class Groups:
    """What group() finds. templates: TEMPLATES_DTYPE rows, one per cover in a template group.
    lookalikes: LOOKALIKES_DTYPE rows, one per release pair, mbid_a < mbid_b. clusters: {mbid:
    cluster_id} for every release, ids from 0."""

    templates: np.ndarray
    lookalikes: np.ndarray
    clusters: dict[str, int]


def group(covers_files: list[Path], artists: Mapping[str, Iterable[str]], release_groups: Mapping[str, str | None],
          lookalike_distance: int) -> Groups:
    """Template groups, lookalike pairs and dedup clusters of the releases in the covers files.

    artists maps a release's mbid to its artist ids (a missing mbid has none), and release_groups
    to its release group (a missing or None one stands alone). Each cover carries the artists of
    every release that uses its pHash, Various Artists left out, and two covers share an artist
    when those sets meet. Results are the same on every run over the same input.
    """
    if lookalike_distance < DEDUP_DISTANCE:
        raise ValueError(f"lookalike_distance must be at least DEDUP_DISTANCE ({DEDUP_DISTANCE})")
    _check_recipes(covers_files)
    rows = np.concatenate([np.load(f) for f in covers_files])
    mbids, mbid_of_row = np.unique(rows["key"].astype("U36"), return_inverse=True)  # mbid is key[:36]
    hashes, hash_of_row = np.unique(rows["phash"], return_inverse=True)  # identical hashes share a node
    n_hashes, n_releases = len(hashes), len(mbids)
    # Each distinct (hash, release) once, sorted by hash: the releases that use each hash
    hash_release = np.unique(hash_of_row.astype(np.int64) * n_releases + mbid_of_row)
    hr_hash, hr_release = hash_release // n_releases, hash_release % n_releases
    release_ptr, release_ids = _release_artists(mbids, artists)
    hash_artists = _hash_artists(hr_hash, hr_release, release_ptr, release_ids, n_hashes)
    # A release without a release group stands alone, under its own mbid
    _, release_group = np.unique([release_groups.get(m) or m for m in mbids.tolist()], return_inverse=True)

    near_a, near_b = near_pairs(hashes, max(lookalike_distance, TEMPLATE_DISTANCE))
    distance = np.bitwise_count(hashes[near_a] ^ hashes[near_b]).astype(np.uint8)
    shared = _share_an_artist(near_a, near_b, hash_artists)
    # A design joins hashes within TEMPLATE_DISTANCE bits, and same-artist hashes within the
    # lookalike distance, so one artist's series joins even where its covers sit further apart
    design = (distance <= TEMPLATE_DISTANCE) | shared
    template_of_hash, template_counts = _templates(near_a[design], near_b[design], hr_hash, hr_release, release_group,
                                                  hash_artists, n_hashes)
    in_template = np.flatnonzero(template_of_hash[hash_of_row] >= 0)
    row_template = template_of_hash[hash_of_row[in_template]]
    templates = np.empty(len(in_template), dtype=TEMPLATES_DTYPE)
    templates["key"], templates["template_id"] = rows["key"][in_template], row_template
    for field, counts in template_counts.items():
        templates[field] = counts[row_template]
    templates = templates[np.argsort(templates["key"], kind="stable")]
    # Template covers drop out of the other two outputs. A hash shares an artist with itself when
    # it has one, so the releases of one hash pair up.
    usable = (template_of_hash < 0) & (np.diff(hash_artists[0]) > 0)
    same = np.flatnonzero(usable)
    keep = usable[near_a] & usable[near_b] & shared & (distance <= lookalike_distance)
    link_a = np.concatenate([near_a[keep], same])
    link_b = np.concatenate([near_b[keep], same])
    link_distance = np.concatenate([distance[keep], np.zeros(len(same), np.uint8)])
    lookalikes = _release_pairs(link_a, link_b, link_distance, hr_hash, hr_release, n_hashes, mbids)

    # Dedup clusters: nodes are the hashes, then the releases. Each usable hash links to its
    # releases, and usable hashes up to DEDUP_DISTANCE apart that share an artist link.
    close = link_distance <= DEDUP_DISTANCE
    by_hash = usable[hr_hash]
    src = np.concatenate([link_a[close], hr_hash[by_hash]])
    dst = np.concatenate([link_b[close], n_hashes + hr_release[by_hash]])
    size = n_hashes + n_releases
    graph = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(size, size))
    _, labels = connected_components(graph, directed=False)
    _, cluster_ids = np.unique(labels[n_hashes:], return_inverse=True)  # renumber from 0
    return Groups(templates, lookalikes, dict(zip(mbids.tolist(), cluster_ids.tolist())))


def _templates(design_a: np.ndarray, design_b: np.ndarray, hr_hash: np.ndarray, hr_release: np.ndarray,
               release_group: np.ndarray, hash_artists: tuple[np.ndarray, np.ndarray],
               n_hashes: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Each hash's template id, or -1, and each template's counts of releases, release groups and
    artists (Various Artists left out), keyed by their TEMPLATES_DTYPE fields.

    A template is a connected component of hashes over the design links (design_a[i], design_b[i])
    whose releases span TEMPLATE_RELEASE_GROUPS or more release groups."""
    graph = coo_matrix((np.ones(len(design_a), dtype=np.int8), (design_a, design_b)), shape=(n_hashes, n_hashes))
    n_groups, group_of_hash = connected_components(graph, directed=False)
    owner = group_of_hash[hr_hash].astype(np.int64)
    ptr, ids = hash_artists

    def distinct(groups: np.ndarray, values: np.ndarray) -> np.ndarray:
        n = int(values.max()) + 1 if len(values) else 1
        return np.bincount(np.unique(groups * n + values) // n, minlength=n_groups)

    counts = {"releases": distinct(owner, hr_release),
              "release_groups": distinct(owner, release_group[hr_release]),
              "artists": distinct(np.repeat(group_of_hash, np.diff(ptr)).astype(np.int64), ids)}
    is_template = counts["release_groups"] >= TEMPLATE_RELEASE_GROUPS
    template_id = np.full(n_groups, -1, dtype=np.int64)
    template_id[is_template] = np.arange(int(is_template.sum()))
    return template_id[group_of_hash], {field: c[is_template] for field, c in counts.items()}


def _release_artists(mbids: np.ndarray, artists: Mapping[str, Iterable[str]]) -> tuple[np.ndarray, np.ndarray]:
    """Each release's artists as integers, Various Artists left out, in CSR form: release r's are
    ids[ptr[r]:ptr[r + 1]]."""
    codes: dict[str, int] = {}
    ptr, ids = [0], []
    for mbid in mbids.tolist():
        ids.extend({codes.setdefault(a, len(codes)) for a in artists.get(mbid, ()) if a != VARIOUS_ARTISTS})
        ptr.append(len(ids))
    return np.array(ptr, dtype=np.int64), np.array(ids, dtype=np.int64)


def _hash_artists(hr_hash: np.ndarray, hr_release: np.ndarray, release_ptr: np.ndarray, release_ids: np.ndarray,
                  n_hashes: int) -> tuple[np.ndarray, np.ndarray]:
    """Each hash's artists, the union over the releases using it, in CSR form: hash h's artists
    are ids[ptr[h]:ptr[h + 1]], sorted."""
    counts = np.diff(release_ptr)[hr_release]
    owner = np.repeat(hr_hash, counts)
    ids = release_ids[_ranges(release_ptr[hr_release], counts)]
    n_artists = int(release_ids.max()) + 1 if len(release_ids) else 1
    pairs = np.unique(owner * n_artists + ids)
    ptr = np.concatenate([[0], np.cumsum(np.bincount(pairs // n_artists, minlength=n_hashes))])
    return ptr, pairs % n_artists


def _ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """starts[0], ..., starts[0] + counts[0] - 1, then the same for each later start."""
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    return np.repeat(starts, counts) + offsets


def _share_an_artist(a: np.ndarray, b: np.ndarray, hash_artists: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Whether hashes a[i] and b[i] have an artist in common, for every i."""
    ptr, ids = hash_artists
    n_artists = int(ids.max()) + 1 if len(ids) else 1
    keys = []
    for side in (a, b):
        counts = ptr[side + 1] - ptr[side]
        link = np.repeat(np.arange(len(side), dtype=np.int64), counts)
        keys.append(link * n_artists + ids[_ranges(ptr[side], counts)])
    # Each side lists an artist once per link, so a key on both sides is a shared artist
    both = np.intersect1d(keys[0], keys[1], assume_unique=True)
    shared = np.zeros(len(a), dtype=bool)
    shared[both // n_artists] = True
    return shared


def _release_pairs(a: np.ndarray, b: np.ndarray, distance: np.ndarray, hr_hash: np.ndarray, hr_release: np.ndarray,
                   n_hashes: int, mbids: np.ndarray) -> np.ndarray:
    """LOOKALIKES_DTYPE rows: every release of hash a[i] with every release of hash b[i], one row per
    release pair at its smallest distance, a release never paired with itself."""
    ptr = np.concatenate([[0], np.cumsum(np.bincount(hr_hash, minlength=n_hashes))])
    count_a, count_b = ptr[a + 1] - ptr[a], ptr[b + 1] - ptr[b]
    per_link = count_a * count_b
    link = np.repeat(np.arange(len(a)), per_link)
    k = _ranges(np.zeros(len(a), dtype=np.int64), per_link)  # position within the link's grid
    ra = hr_release[ptr[a][link] + k // count_b[link]]
    rb = hr_release[ptr[b][link] + k % count_b[link]]
    d = distance[link]
    lo, hi = np.minimum(ra, rb), np.maximum(ra, rb)
    keep = lo != hi
    lo, hi, d = lo[keep], hi[keep], d[keep]
    order = np.lexsort((d, hi, lo))  # by pair, then distance, so a pair's first row is its closest
    lo, hi, d = lo[order], hi[order], d[order]
    first = np.ones(len(lo), dtype=bool)
    first[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
    out = np.empty(int(first.sum()), dtype=LOOKALIKES_DTYPE)
    out["mbid_a"], out["mbid_b"], out["distance"] = mbids[lo[first]], mbids[hi[first]], d[first]
    return out


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


def read_catalog(db_path: Path) -> tuple[dict[str, list[str]], dict[str, str | None], str]:
    """Every catalog release's artist ids and release group, by mbid, and catalog_fingerprint(db_path)."""
    artists, release_groups, digest = {}, {}, hashlib.sha1()
    for mbid, ids_json, release_group in _catalog_rows(db_path):
        artists[mbid] = json.loads(ids_json)
        release_groups[mbid] = release_group
        digest.update(f"{mbid}\t{ids_json}\t{release_group}\n".encode())
    return artists, release_groups, digest.hexdigest()[:16]


def catalog_fingerprint(db_path: Path) -> str:
    """A hash of every release's mbid, artist ids and release group, the part of the catalog the
    groups use."""
    digest = hashlib.sha1()
    for mbid, ids_json, release_group in _catalog_rows(db_path):
        digest.update(f"{mbid}\t{ids_json}\t{release_group}\n".encode())
    return digest.hexdigest()[:16]


def _catalog_rows(db_path: Path) -> Iterator[tuple[str, str, str | None]]:
    if not db_path.is_file():
        raise SystemExit(f"Catalog not found at {db_path}. Build it with: python -m pipeline.metadata.build_catalog_db")
    db = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        yield from db.execute("SELECT mbid, artist_ids_json, release_group FROM releases ORDER BY mbid")
    finally:
        db.close()


def groups_recipe(covers_files: list[Path], lookalike_distance: int, catalog: str) -> str:
    """What the three outputs were made from: the rule's version, distances and cutoff, the catalog's
    fingerprint and every prefix's done marker, which holds its hash recipe and cover list. A
    catalog update or a new rule makes old outputs stale."""
    _check_recipes(covers_files)
    markers = "\n".join(f"{f.name}: {(f.parent / f.name.replace('.covers.npy', '.done')).read_text()}"
                         for f in sorted(covers_files))
    return (f"{GROUP_RULE} template{TEMPLATE_DISTANCE}-{TEMPLATE_RELEASE_GROUPS}release-groups dedup{DEDUP_DISTANCE} "
            f"lookalike{lookalike_distance} catalog-{catalog} gallery-{hashlib.sha1(markers.encode()).hexdigest()[:16]}")


def write_groups(groups: Groups, out_dir: Path, recipe: str) -> None:
    """templates.npy, lookalikes.npy and clusters.npy (one row per release sorted by mbid, with its
    cluster's size in releases), each with a .recipe beside it."""
    mbids = np.array(sorted(groups.clusters), dtype="U36")
    ids = np.array([groups.clusters[m] for m in mbids.tolist()], dtype=np.uint32)
    clusters = np.empty(len(mbids), dtype=CLUSTERS_DTYPE)
    clusters["mbid"], clusters["cluster_id"], clusters["cluster_size"] = mbids, ids, np.bincount(ids)[ids]
    for name, rows in (("templates", groups.templates), ("lookalikes", groups.lookalikes), ("clusters", clusters)):
        path = out_dir / f"{name}.npy"
        recipe_path = path.with_suffix(".recipe")
        recipe_path.unlink(missing_ok=True)  # a write that fails partway must not look current
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as f:  # a file handle stops numpy from appending another .npy
            np.save(f, rows)
        os.replace(tmp, path)
        recipe_path.write_text(recipe)


def load_clusters(path: Path, lookalike_distance: int = LOOKALIKE_DISTANCE, catalog_db: Path | None = None) -> dict[str, int]:
    """{mbid: cluster_id}, as group() returned it, for the index. See _load for what it refuses."""
    rows = _load(path, lookalike_distance, catalog_db)
    return dict(zip(rows["mbid"].tolist(), rows["cluster_id"].tolist()))


def load_lookalikes(path: Path, lookalike_distance: int = LOOKALIKE_DISTANCE, catalog_db: Path | None = None) -> np.ndarray:
    """LOOKALIKES_DTYPE rows, as group() returned them, for the eval's hit rule."""
    return _load(path, lookalike_distance, catalog_db)


def load_templates(path: Path, lookalike_distance: int = LOOKALIKE_DISTANCE, catalog_db: Path | None = None) -> np.ndarray:
    """TEMPLATES_DTYPE rows, as group() returned them, for the exclusion list's review."""
    return _load(path, lookalike_distance, catalog_db)


def _load(path: Path, lookalike_distance: int, catalog_db: Path | None) -> np.ndarray:
    """Refuses an output made with another lookalike distance or rule, from a gallery that has
    changed since, or from another catalog (see groups_recipe). Reads the whole catalog to check."""
    recipe_path = path.with_suffix(".recipe")
    catalog = catalog_fingerprint(Path(catalog_db or config.CATALOG_DB))
    expected = groups_recipe(sorted(path.parent.glob("*.covers.npy")), lookalike_distance, catalog)
    if not recipe_path.exists() or recipe_path.read_text() != expected:
        raise ValueError(f"{path} is stale or not at lookalike distance {lookalike_distance}; rerun cluster_duplicates")
    return np.load(path)


def size_histogram(clusters: dict[str, int]) -> list[str]:
    """Lines counting clusters, and the releases in them, by cluster size in releases."""
    sizes = np.bincount(np.fromiter(clusters.values(), dtype=np.int64, count=len(clusters)))
    lines = []
    for label, low, high in SIZE_BUCKETS:
        in_bucket = sizes[(sizes >= low) & (sizes <= high)]
        if len(in_bucket):
            n = len(in_bucket)
            lines.append(f"  {label}: {n:,} cluster{'s' * (n != 1)}, {int(in_bucket.sum()):,} releases")
    lines.append(f"  largest: {', '.join(f'{s:,}' for s in np.sort(sizes)[::-1][:5])}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lookalike-distance", type=int, default=LOOKALIKE_DISTANCE,
                        help="most pHash bits two same-artist lookalikes differ in")
    args = parser.parse_args()

    out_dir = config.GALLERY_DIR
    covers_files = sorted(out_dir.glob("*.covers.npy"))
    if not covers_files:
        sys.exit(f"No covers.npy files in {out_dir}; run shard_and_hash first")
    start = time.monotonic()
    artists, release_groups, catalog = read_catalog(config.CATALOG_DB)
    # Taken before grouping, so a prefix redone during the run makes the loaders refuse the result
    recipe = groups_recipe(covers_files, args.lookalike_distance, catalog)
    groups = group(covers_files, artists, release_groups, args.lookalike_distance)
    write_groups(groups, out_dir, recipe)
    n_covers = sum(len(np.load(f, mmap_mode="r")) for f in covers_files)
    templates = groups.templates
    n_template_releases = len(np.unique(templates["key"].astype("U36")))  # mbid is key[:36]
    print(f"{n_covers:,} covers of {len(groups.clusters):,} releases in {len(covers_files)} prefixes "
          f"({time.monotonic() - start:.0f} s), written to {out_dir}")
    print(f"Templates: {len(np.unique(templates['template_id'])):,} designs over {TEMPLATE_RELEASE_GROUPS}+ release "
          f"groups, {len(templates):,} covers of {n_template_releases:,} releases")
    by_distance = np.bincount(groups.lookalikes["distance"], minlength=args.lookalike_distance + 1)
    print(f"Lookalikes: {len(groups.lookalikes):,} same-artist pairs within {args.lookalike_distance} bits; "
          f"by distance: {', '.join(f'{d}: {n:,}' for d, n in enumerate(by_distance.tolist()) if n)}")
    print(f"Clusters: {len(set(groups.clusters.values())):,} within {DEDUP_DISTANCE} bits, by size:")
    print("\n".join(size_histogram(groups.clusters)))


if __name__ == "__main__":
    main()
