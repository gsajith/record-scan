import itertools
import shutil
import tarfile

import numpy as np
import pytest
from PIL import Image

from pipeline import config
from pipeline.gallery import shard_and_hash
from pipeline.gallery.cluster_duplicates import cluster
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


def test_cluster_groups_identical_and_near_duplicate_covers_but_not_unrelated_ones(gallery):
    clusters = cluster(gallery, max_distance=8)

    assert set(clusters) == {SQUARE, WIDE, SMALL, COPY, OTHER, TWIN}  # releases, not files: EXTRA is SQUARE's
    assert clusters[SQUARE] == clusters[TWIN] == clusters[COPY]
    assert len({clusters[m] for m in (SQUARE, WIDE, SMALL, OTHER)}) == 4


def write_covers(out, prefix, rows, recipe=shard_and_hash.HASH_RECIPE):
    """A covers.npy and a done marker, as process_prefix leaves them, from (key, phash) rows."""
    out.mkdir(parents=True, exist_ok=True)
    covers = np.array([(key, phash, 500, 500) for key, phash in rows], dtype=shard_and_hash.COVERS_DTYPE)
    np.save(out / f"{prefix}.covers.npy", covers)
    (out / f"{prefix}.done").write_text(f"original-{recipe} 0123456789abcdef")
    return out / f"{prefix}.covers.npy"


def mbid(i):
    return f"{i:08x}-0000-4000-8000-000000000000"


def test_cluster_joins_hashes_up_to_max_distance_bits_apart_and_no_further(tmp_path):
    base = 0x0123_4567_89AB_CDEF
    eight = base ^ 0x8000_0400_0020_0101  # 8 bits flipped, spread over the 64
    rows = [(mbid(0), base), (mbid(1), eight), (mbid(2), base ^ 0xFF00_0000_0000_0000),  # 8 adjacent bits
            (mbid(3), base ^ 0x1FF), (mbid(4), ~base & (2**64 - 1))]  # 9 bits, and every bit
    clusters = cluster([write_covers(tmp_path, "00", rows)], max_distance=8)

    assert clusters[mbid(0)] == clusters[mbid(1)] == clusters[mbid(2)]
    assert len({clusters[mbid(i)] for i in (0, 3, 4)}) == 3
    assert len(set(cluster([tmp_path / "00.covers.npy"], max_distance=0).values())) == 5


def test_cluster_finds_the_same_clusters_as_comparing_every_pair(tmp_path):
    rng = np.random.default_rng(0)
    hashes = []
    for _ in range(150):  # each base hash with a few variants up to 12 bits away, some chaining
        h = int(rng.integers(0, 2**63)) * 2 + int(rng.integers(0, 2))
        hashes.append(h)
        for _ in range(int(rng.integers(0, 4))):
            for bit in rng.choice(64, size=int(rng.integers(1, 13)), replace=False):
                h ^= 1 << int(bit)
            hashes.append(h)
    rows = [(mbid(i), h) for i, h in enumerate(hashes)]
    clusters = cluster([write_covers(tmp_path, "00", rows)], max_distance=8)

    expected = list(range(len(hashes)))  # union-find over every pair

    def root(i):
        while expected[i] != i:
            i = expected[i]
        return i

    for i, j in itertools.combinations(range(len(hashes)), 2):
        if (hashes[i] ^ hashes[j]).bit_count() <= 8:
            expected[root(i)] = root(j)
    same = {(i, j): root(i) == root(j) for i, j in itertools.combinations(range(len(hashes)), 2)}
    chained = [p for p, joined in same.items() if joined and (hashes[p[0]] ^ hashes[p[1]]).bit_count() > 8]
    assert sum(same.values()) > 100 and chained  # real clusters to find, some joined only through others
    assert same == {(i, j): clusters[mbid(i)] == clusters[mbid(j)] for i, j in same}
