import itertools
import json
import shutil
import sqlite3
import sys
import tarfile

import numpy as np
import pytest
from PIL import Image

from pipeline import config
from pipeline.gallery import cluster_duplicates, shard_and_hash
from pipeline.gallery.cluster_duplicates import group, load_clusters, load_lookalikes, load_templates
from pipeline.gallery.shard_and_hash import is_done, process_prefix

SQUARE = "00000000-0000-4000-8000-000000000001"
WIDE = "00000000-0000-4000-8000-000000000002"
CORRUPT = "00000000-0000-4000-8000-000000000003"
SMALL = "00000000-0000-4000-8000-000000000004"
EXTRA = f"{SQUARE}-987654321"  # an extra front of SQUARE, added by a catalog update
COPY = "01000000-0000-4000-8000-000000000005"  # SQUARE's picture, bigger and more compressed
OTHER = "02000000-0000-4000-8000-000000000006"


def noise(size, seed):
    """A textured picture, so perceptual hashes of different seeds are far apart."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)  # blocky, like a cover's shapes
    return Image.fromarray(small).resize(size, Image.Resampling.NEAREST)


@pytest.fixture
def drive(tmp_path, reload_config):
    """A tmp image root with three prefix folders and a corrupt file. Returns {prefix: files}."""
    root = tmp_path / "images"
    for prefix in ("00", "01", "02"):
        (root / prefix).mkdir(parents=True)
    noise((500, 500), 1).save(root / "00" / f"{SQUARE}.jpg")
    noise((600, 400), 2).save(root / "00" / f"{WIDE}.jpg")
    (root / "00" / f"{CORRUPT}.jpg").write_bytes(b"not a jpeg")
    noise((200, 200), 3).save(root / "00" / f"{SMALL}.jpg")
    noise((500, 500), 4).save(root / "00" / f"{EXTRA}.jpg")
    noise((700, 700), 1).save(root / "01" / f"{COPY}.jpg", quality=60)
    noise((500, 500), 5).save(root / "02" / f"{OTHER}.jpg")
    reload_config(RS_IMAGE_ROOT=root)
    files = {"00": [f"00/{k}.jpg" for k in (SQUARE, EXTRA, WIDE, CORRUPT, SMALL)],
             "01": [f"01/{COPY}.jpg"], "02": [f"02/{OTHER}.jpg"]}
    return files


def read_shard(path):
    with tarfile.open(path) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers()}


def test_process_prefix_shards_the_original_bytes_and_logs_the_corrupt_one(drive, tmp_path):
    out = tmp_path / "gallery"

    assert process_prefix("00", drive["00"], out, workers=2) == (4, 1)

    shard = read_shard(out / "00.tar")
    assert set(shard) == {f"{k}.jpg" for k in (SQUARE, EXTRA, WIDE, SMALL)}
    for name, data in shard.items():  # lossless: sized copies are made from these later
        assert data == (config.IMAGE_ROOT / "00" / name).read_bytes()
    rows = np.load(out / "00.covers.npy")
    sizes = {k: (int(w), int(h)) for k, w, h in zip(rows["key"], rows["width"], rows["height"])}
    assert sizes == {SQUARE: (500, 500), EXTRA: (500, 500), WIDE: (600, 400), SMALL: (200, 200)}
    assert CORRUPT in (out / "00.errors.log").read_text()


def load_hashes(path):
    rows = np.load(path)
    return dict(zip(rows["key"], rows["phash"]))


def bits_apart(a, b):
    return int(a ^ b).bit_count()


def test_process_prefix_hashes_each_cover_so_lookalikes_land_close(drive, tmp_path):
    out = tmp_path / "gallery"
    for prefix in ("00", "01", "02"):
        process_prefix(prefix, drive[prefix], out, workers=2)

    hashes = load_hashes(out / "00.covers.npy") | load_hashes(out / "01.covers.npy") | load_hashes(out / "02.covers.npy")
    assert np.load(out / "00.covers.npy")["phash"].dtype == np.uint64
    assert set(hashes) == {SQUARE, EXTRA, WIDE, SMALL, COPY, OTHER}
    assert bits_apart(hashes[SQUARE], hashes[COPY]) <= 8
    assert bits_apart(hashes[SQUARE], hashes[OTHER]) > 16
    assert bits_apart(hashes[SQUARE], hashes[EXTRA]) > 16


def test_a_done_prefix_is_skipped_until_its_cover_list_changes(drive, tmp_path):
    out = tmp_path / "gallery"
    assert not is_done("00", drive["00"], out)
    process_prefix("00", drive["00"], out, workers=2)
    assert is_done("00", drive["00"], out)

    shard_before = (out / "00.tar").read_bytes()
    assert process_prefix("00", drive["00"], out, workers=2) == (0, 0)
    assert (out / "00.tar").read_bytes() == shard_before
    # The same covers in another order are the same prefix, and the shard is in name order
    assert is_done("00", drive["00"][::-1], out)
    assert list(read_shard(out / "00.tar")) == sorted(read_shard(out / "00.tar"))

    # A lost shard isn't done, whatever the marker says
    (out / "00.tar").unlink()
    assert not is_done("00", drive["00"], out)
    assert process_prefix("00", drive["00"], out, workers=2) == (4, 1)

    # A catalog update drops a cover: the prefix is redone with the new list
    fewer = [f for f in drive["00"] if SMALL not in f]
    assert not is_done("00", fewer, out)
    assert process_prefix("00", fewer, out, workers=2) == (3, 1)
    assert f"{SMALL}.jpg" not in read_shard(out / "00.tar")


def test_a_drive_that_drops_mid_prefix_leaves_the_prefix_pending(drive, tmp_path, monkeypatch):
    out = tmp_path / "gallery"
    real_read = shard_and_hash.read_and_hash

    def read_then_unmount(path):
        result = real_read(path)
        shutil.rmtree(config.IMAGE_ROOT, ignore_errors=True)  # the drive goes away after the first cover
        return result

    monkeypatch.setattr(shard_and_hash, "read_and_hash", read_then_unmount)
    with pytest.raises(SystemExit):
        process_prefix("00", drive["00"], out, workers=1)
    assert not is_done("00", drive["00"], out)
    assert not (out / "00.tar").exists()
    assert not (out / "00.errors.log").exists()  # stopped at the first unreadable cover, logged nothing


def test_a_read_error_leaves_the_prefix_pending_but_a_missing_file_does_not(drive, tmp_path):
    out = tmp_path / "gallery"
    wide = config.IMAGE_ROOT / "00" / f"{WIDE}.jpg"
    files = drive["00"] + ["00/00000000-0000-4000-8000-0000000000ff.jpg"]  # listed, never downloaded
    wide.chmod(0)  # stands in for an I/O error on the USB drive
    try:
        assert process_prefix("00", files, out, workers=2) == (3, 3)
    finally:
        wide.chmod(0o644)
    assert not is_done("00", files, out)  # a rerun gets another try at WIDE

    assert process_prefix("00", files, out, workers=2) == (4, 2)
    assert is_done("00", files, out)
    log = (out / "00.errors.log").read_text()
    assert CORRUPT in log and "0000000000ff" in log and WIDE not in log


def test_a_redo_with_a_read_error_does_not_inherit_the_old_done_marker(drive, tmp_path):
    out = tmp_path / "gallery"
    process_prefix("00", drive["00"], out, workers=2)
    (out / "00.covers.npy").unlink()
    assert not is_done("00", drive["00"], out)  # lost hashes, like a lost shard, mean not done

    wide = config.IMAGE_ROOT / "00" / f"{WIDE}.jpg"
    wide.chmod(0)
    try:
        process_prefix("00", drive["00"], out, workers=2)
    finally:
        wide.chmod(0o644)
    assert not is_done("00", drive["00"], out)  # the partial redo isn't done under the old marker


def test_a_library_upgrade_keeps_the_copy_done_and_records_the_hash_recipe(drive, tmp_path):
    out = tmp_path / "gallery"
    process_prefix("00", drive["00"], out, workers=2)
    assert shard_and_hash.hash_recipe("00", out) == shard_and_hash.HASH_RECIPE

    # Done markers from another numpy: the copy is still good, and the old hash recipe is visible
    done = out / "00.done"
    done.write_text(done.read_text().replace(shard_and_hash.HASH_RECIPE, "phash8-pillow11.3.0-scipy1.18.1-numpy0.0.0"))
    assert is_done("00", drive["00"], out)
    assert shard_and_hash.hash_recipe("00", out) == "phash8-pillow11.3.0-scipy1.18.1-numpy0.0.0"


TWIN = "02000000-0000-4000-8000-000000000007"  # SQUARE's file, byte for byte, under another release


@pytest.fixture
def gallery(drive, tmp_path):
    """The drive fixture sharded and hashed, plus TWIN in prefix 02. Returns the covers.npy paths."""
    shutil.copy(config.IMAGE_ROOT / "00" / f"{SQUARE}.jpg", config.IMAGE_ROOT / "02" / f"{TWIN}.jpg")
    drive["02"].append(f"02/{TWIN}.jpg")
    out = tmp_path / "gallery"
    for prefix in ("00", "01", "02"):
        process_prefix(prefix, drive[prefix], out, workers=2)
    return [out / f"{p}.covers.npy" for p in ("00", "01", "02")]


def test_group_pairs_identical_and_near_duplicate_covers_but_not_unrelated_ones(gallery):
    artists = {m: ["x"] for m in (SQUARE, WIDE, SMALL, COPY, OTHER, TWIN)}  # one artist, so only pictures differ
    groups = group(gallery, artists, {}, lookalike_distance=8)

    assert set(groups.clusters) == {SQUARE, WIDE, SMALL, COPY, OTHER, TWIN}  # releases, not files: EXTRA is SQUARE's
    assert set(pairs(groups)) == {tuple(sorted(p)) for p in ((SQUARE, TWIN), (SQUARE, COPY), (COPY, TWIN))}
    assert groups.clusters[SQUARE] == groups.clusters[TWIN] == groups.clusters[COPY]
    assert len({groups.clusters[m] for m in (SQUARE, WIDE, SMALL, OTHER)}) == 4


def write_covers(out, prefix, rows, recipe=shard_and_hash.HASH_RECIPE):
    """A covers.npy and a done marker, as process_prefix leaves them, from (key, phash) rows."""
    out.mkdir(parents=True, exist_ok=True)
    covers = np.array([(key, phash, 500, 500) for key, phash in rows], dtype=shard_and_hash.COVERS_DTYPE)
    np.save(out / f"{prefix}.covers.npy", covers)
    (out / f"{prefix}.done").write_text(f"original-{recipe} 0123456789abcdef")
    return out / f"{prefix}.covers.npy"


def mbid(i):
    return f"{i:08x}-0000-4000-8000-000000000000"


def test_lookalikes_reach_the_distance_in_bits_and_no_further(tmp_path):
    eight = BASE ^ 0x8000_0400_0020_0101  # 8 bits flipped, spread over the 64
    rows = [(mbid(0), BASE), (mbid(1), eight), (mbid(2), BASE ^ 0xFF00_0000_0000_0000),  # 8 adjacent bits
            (mbid(3), BASE ^ 0x1FF), (mbid(4), ~BASE & (2**64 - 1))]  # 9 bits, and every bit
    groups = group([write_covers(tmp_path, "00", rows)], {mbid(i): ["x"] for i in range(5)}, {},
                   lookalike_distance=8)

    assert {p for p, d in pairs(groups).items() if mbid(0) in p} == {(mbid(0), mbid(1)), (mbid(0), mbid(2))}


def test_group_refuses_prefixes_hashed_with_different_recipes_or_not_done(tmp_path):
    files = [write_covers(tmp_path, "00", [(mbid(0), 1)]),
             write_covers(tmp_path, "01", [(mbid(1), 1)], recipe="phash8-pillow11.3.0-scipy1.18.1-numpy0.0.0")]
    with pytest.raises(ValueError, match="different recipes.*numpy0.0.0: 01"):
        group(files, {}, {}, lookalike_distance=8)

    (tmp_path / "01.done").unlink()  # a prefix left pending after read errors has a partial covers.npy
    with pytest.raises(ValueError, match="01 isn't done"):
        group(files, {}, {}, lookalike_distance=8)


def write_catalog(artists, release_groups=None):
    """A catalog.sqlite at config.CATALOG_DB holding just the releases' artist ids and release groups."""
    config.CATALOG_DB.parent.mkdir(parents=True, exist_ok=True)
    config.CATALOG_DB.unlink(missing_ok=True)
    with sqlite3.connect(config.CATALOG_DB) as db:
        db.execute("CREATE TABLE releases (mbid TEXT PRIMARY KEY, artist_ids_json TEXT NOT NULL, release_group TEXT)")
        db.executemany("INSERT INTO releases VALUES (?, ?, ?)",
                       [(m, json.dumps(a), (release_groups or {}).get(m)) for m, a in artists.items()])
    db.close()


GALLERY_ARTISTS = {SQUARE: ["x"], TWIN: ["x"], COPY: ["y", "x"], WIDE: ["x"], SMALL: ["z"], OTHER: ["x"]}


@pytest.fixture
def ran(gallery, tmp_path, reload_config, monkeypatch, capsys):
    """The CLI run over the gallery fixture, with a catalog. Returns the gallery dir and what it printed."""
    reload_config(RS_DATA_DIR=tmp_path)  # GALLERY_DIR is where the gallery fixture wrote
    write_catalog(GALLERY_ARTISTS)
    monkeypatch.setattr(sys, "argv", ["cluster_duplicates", "--lookalike-distance", "8"])
    capsys.readouterr()
    cluster_duplicates.main()
    return tmp_path / "gallery", capsys.readouterr().out


def test_the_cli_writes_groups_that_load_back_and_prints_their_counts(gallery, ran):
    ran, out = ran
    expected = group(gallery, GALLERY_ARTISTS, {}, lookalike_distance=8)

    assert load_clusters(ran / "clusters.npy", lookalike_distance=8) == expected.clusters
    assert (load_lookalikes(ran / "lookalikes.npy", lookalike_distance=8) == expected.lookalikes).all()
    assert len(load_templates(ran / "templates.npy", lookalike_distance=8)) == 0
    rows = np.load(ran / "clusters.npy")
    assert dict(zip(rows["mbid"], rows["cluster_size"].tolist())) == {SQUARE: 3, TWIN: 3, COPY: 3, WIDE: 1, SMALL: 1, OTHER: 1}
    assert "7 covers of 6 releases in 3 prefixes" in out
    assert "Templates: 0 groups" in out and "Lookalikes: 3 same-artist pairs within 8 bits; by distance: 0: 3" in out
    assert "Clusters: 4 within 2 bits" in out and "1 release: 3 clusters" in out and "3 releases: 1 cluster" in out


def test_loading_groups_refuses_ones_made_with_another_distance_gallery_or_catalog(ran):
    ran, _ = ran
    with pytest.raises(ValueError, match="stale"):
        load_clusters(ran / "clusters.npy", lookalike_distance=6)
    write_catalog(GALLERY_ARTISTS | {WIDE: ["w"]})  # a catalog update credits WIDE to another artist
    with pytest.raises(ValueError, match="stale"):
        load_lookalikes(ran / "lookalikes.npy", lookalike_distance=8)
    write_catalog(GALLERY_ARTISTS, {COPY: "an album"})  # or files COPY under a release group
    with pytest.raises(ValueError, match="stale"):
        load_lookalikes(ran / "lookalikes.npy", lookalike_distance=8)
    write_catalog(GALLERY_ARTISTS)
    load_templates(ran / "templates.npy", lookalike_distance=8)  # the old catalog again, so current
    done = ran / "01.done"
    done.write_text(done.read_text() + "0")  # a catalog update redid prefix 01 with another cover list
    with pytest.raises(ValueError, match="stale"):
        load_templates(ran / "templates.npy", lookalike_distance=8)

BASE = 0x0123_4567_89AB_CDEF


def flip(h, *bits):
    for bit in bits:
        h ^= 1 << bit
    return h


def pairs(groups):
    return {(str(a), str(b)): int(d) for a, b, d in groups.lookalikes[["mbid_a", "mbid_b", "distance"]]}


def test_lookalikes_keep_a_same_artist_pair_and_drop_a_different_artist_pair_at_the_same_distance(tmp_path):
    rows = [(mbid(0), BASE), (mbid(1), flip(BASE, 3, 40)), (mbid(2), flip(BASE, 9, 50))]
    artists = {mbid(0): ["x"], mbid(1): ["x", "y"], mbid(2): ["z"]}  # 2 is 2 and 4 bits from the others

    groups = group([write_covers(tmp_path, "00", rows)], artists, {}, lookalike_distance=4)

    assert pairs(groups) == {(mbid(0), mbid(1)): 2}


def test_lookalikes_are_pairs_so_a_near_b_near_c_does_not_pair_a_with_c(tmp_path):
    b = flip(BASE, 1, 2, 3, 4)
    rows = [(mbid(0), BASE), (mbid(1), b), (mbid(2), flip(b, 5, 6, 7, 8)), (mbid(3), BASE)]  # 3 is 0's copy
    artists = {mbid(i): ["x"] for i in range(4)}

    groups = group([write_covers(tmp_path, "00", rows)], artists, {}, lookalike_distance=4)

    assert pairs(groups) == {(mbid(0), mbid(1)): 4, (mbid(1), mbid(2)): 4, (mbid(0), mbid(3)): 0, (mbid(1), mbid(3)): 4}


def test_a_two_bit_group_over_five_release_groups_is_a_template_and_leaves_the_lookalikes_and_clusters(tmp_path):
    chain = [BASE, flip(BASE, 1, 2), flip(BASE, 1, 2, 3, 4), flip(BASE, 1, 2, 3, 4, 5, 6), BASE]  # 2 bits a step
    rows = [(mbid(i), h) for i, h in enumerate(chain)]  # releases 0 to 4, one artist and release group each
    lookalike = flip(BASE, 10, 20, 30, 40)  # 4 bits from 0 and 4, more than 2 from every chain hash
    rows += [(mbid(5), lookalike), (mbid(6), lookalike)]
    other = ~BASE & (2**64 - 1)
    rows += [(mbid(7 + i), flip(other, *range(2 * i))) for i in range(4)]  # a chain over only 4 release groups
    artists = {mbid(i): [a] for i, a in enumerate("abcde")}
    artists |= {mbid(5): ["a"], mbid(6): ["a"], mbid(7): ["p"], mbid(8): ["q"], mbid(9): ["r"], mbid(10): ["s", "p"]}

    groups = group([write_covers(tmp_path, "00", rows)], artists, {}, lookalike_distance=6)

    templates = {str(k): (int(t), int(r), int(g), int(a)) for k, t, r, g, a in groups.templates}
    assert templates == {mbid(i): (0, 5, 5, 5) for i in range(5)}
    assert pairs(groups) == {(mbid(5), mbid(6)): 0, (mbid(7), mbid(10)): 6}
    clusters = groups.clusters
    assert len(clusters) == 11 and len({clusters[mbid(i)] for i in range(5)}) == 5  # template releases stand alone
    assert clusters[mbid(5)] == clusters[mbid(6)]
    assert len(set(clusters.values())) == 10



def test_one_artists_series_on_one_design_is_a_template_but_one_albums_reissues_are_not(tmp_path):
    series = [(mbid(i), flip(BASE, i)) for i in range(5)]  # five albums by one artist, 2 bits apart at most
    album = [(mbid(10 + i), flip(~BASE & (2**64 - 1), i)) for i in range(6)]  # one album, six reissues
    artists = {mbid(i): ["dj"] for i in range(5)}
    artists |= {mbid(10 + i): ["composer", "conductor", f"soloist {i}"] for i in range(6)}  # credits vary
    release_groups = {mbid(i): f"volume {i}" for i in range(5)} | {mbid(10 + i): "the album" for i in range(6)}

    groups = group([write_covers(tmp_path, "00", series + album)], artists, release_groups, lookalike_distance=4)

    templates = {str(k): (int(t), int(r), int(g), int(a)) for k, t, r, g, a in groups.templates}
    assert templates == {mbid(i): (0, 5, 5, 1) for i in range(5)}
    assert set(pairs(groups)) == {(mbid(10 + i), mbid(10 + j)) for i, j in itertools.combinations(range(6), 2)}


def test_various_artists_alone_is_not_a_shared_artist(tmp_path):
    va = cluster_duplicates.VARIOUS_ARTISTS
    near = flip(BASE, 7, 8)
    rows = [(mbid(0), BASE), (mbid(1), near), (mbid(2), BASE),  # compilations on one stock sleeve
            (mbid(3), ~BASE & (2**64 - 1)), (mbid(4), ~near & (2**64 - 1))]
    artists = {mbid(0): [va], mbid(1): [va], mbid(2): [va], mbid(3): [va, "x"], mbid(4): ["x"]}

    groups = group([write_covers(tmp_path, "00", rows)], artists, {}, lookalike_distance=4)

    assert pairs(groups) == {(mbid(3), mbid(4)): 2}
    assert len(set(groups.clusters.values())) == 4



def components(n, links):
    """Union-find: each of range(n)'s root once the links are joined."""
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, j in links:
        parent[root(i)] = root(j)
    return [root(i) for i in range(n)]


def test_group_finds_what_comparing_every_pair_of_covers_finds(tmp_path):
    rng = np.random.default_rng(0)
    hashes = []
    for _ in range(120):  # each base hash with a few variants up to 7 bits away, some chaining
        h = int(rng.integers(0, 2**63)) * 2 + int(rng.integers(0, 2))
        hashes.append(h)
        for _ in range(int(rng.integers(0, 5))):
            h = flip(h, *(int(b) for b in rng.choice(64, size=int(rng.integers(0, 8)), replace=False)))
            hashes.append(h)
    va = cluster_duplicates.VARIOUS_ARTISTS
    release = [int(r) for r in rng.integers(0, len(hashes) * 3 // 4, size=len(hashes))]  # some have 2 covers
    artists = {mbid(r): [str(a) for a in rng.choice([*"abcdefg", va], size=int(rng.integers(1, 3)), replace=False)]
               for r in set(release)}
    release_groups = {mbid(r): str(rng.choice(["g1", "g2", "g3"])) for r in set(release) if r % 2}  # the rest stand alone
    design = int(rng.integers(0, 2**63))  # one design over six release groups, one with two releases
    for n, rg in enumerate(["h1", "h2", "h3", "h4", "h5", "h6", "h6"]):
        hashes.append(flip(design, *range(n % 3)))
        release.append(1000 + n)
        artists[mbid(1000 + n)] = ["a"]
        release_groups[mbid(1000 + n)] = rg
    rows = [(f"{mbid(r)}-{i}", h) for i, (r, h) in enumerate(zip(release, hashes))]  # keys are file stems

    groups = group([write_covers(tmp_path, "00", rows)], artists, release_groups, lookalike_distance=8)

    covers = range(len(hashes))
    carried = [set() for _ in covers]  # a cover carries every artist of every release using its hash
    for i, j in itertools.product(covers, covers):
        if hashes[i] == hashes[j]:
            carried[i] |= set(artists[mbid(release[j])]) - {va}

    def bits(i, j):
        return (hashes[i] ^ hashes[j]).bit_count()

    designs = components(len(hashes), [(i, j) for i, j in itertools.combinations(covers, 2) if bits(i, j) <= 2])
    spans, credits = {}, {}  # each design's release groups (a release without one stands alone), and artists
    for i in covers:
        spans.setdefault(designs[i], set()).add(release_groups.get(mbid(release[i]), mbid(release[i])))
        credits.setdefault(designs[i], set()).update(carried[i])
    template = [len(spans[designs[i]]) >= 5 for i in covers]
    linked = [(i, j) for i, j in itertools.combinations(covers, 2)
              if not template[i] and not template[j] and carried[i] & carried[j]]
    expected = {}
    for i, j in linked:
        a, b = sorted((mbid(release[i]), mbid(release[j])))
        if a != b and bits(i, j) <= 8:
            expected[(a, b)] = min(expected.get((a, b), 64), bits(i, j))
    dedup = components(max(release) + 1, [(release[i], release[j]) for i, j in linked if bits(i, j) <= 2])

    design_releases = {}
    for i in covers:
        design_releases.setdefault(designs[i], set()).add(release[i])
    assert {str(k): (int(r), int(g), int(a)) for k, r, g, a in groups.templates[["key", "releases", "release_groups", "artists"]]} == {
        rows[i][0]: (len(design_releases[designs[i]]), len(spans[designs[i]]), len(credits[designs[i]]))
        for i in covers if template[i]}
    assert pairs(groups) == expected
    same = {(a, b): dedup[a] == dedup[b] for a, b in itertools.combinations(sorted(set(release)), 2)}
    chained = [p for p, joined in same.items() if joined and (mbid(p[0]), mbid(p[1])) not in expected]
    assert any(template) and len(expected) > 50 and chained  # real groups to find, some joined only through others
    assert same == {(a, b): groups.clusters[mbid(a)] == groups.clusters[mbid(b)] for a, b in same}
