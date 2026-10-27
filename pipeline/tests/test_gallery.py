import io
import shutil
import tarfile

import numpy as np
import pytest
from PIL import Image

from pipeline import config
from pipeline.gallery import shard_and_hash
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
        return {m.name: Image.open(io.BytesIO(tar.extractfile(m).read())) for m in tar.getmembers()}


def test_process_prefix_shards_good_covers_at_256_and_logs_the_corrupt_one(drive, tmp_path):
    out = tmp_path / "gallery256"

    assert process_prefix("00", drive["00"], out, workers=2) == (4, 1)

    shard = read_shard(out / "00.tar")
    assert set(shard) == {f"{k}.jpg" for k in (SQUARE, EXTRA, WIDE, SMALL)}
    assert shard[f"{SQUARE}.jpg"].size == (256, 256)
    assert shard[f"{WIDE}.jpg"].size == (384, 256)
    assert shard[f"{SMALL}.jpg"].size == (200, 200)  # never upscaled
    assert all(img.format == "JPEG" for img in shard.values())
    assert CORRUPT in (out / "00.errors.log").read_text()


def load_hashes(path):
    rows = np.load(path)
    return dict(zip(rows["key"], rows["phash"]))


def bits_apart(a, b):
    return int(a ^ b).bit_count()


def test_process_prefix_hashes_each_cover_so_lookalikes_land_close(drive, tmp_path):
    out = tmp_path / "gallery256"
    for prefix in ("00", "01", "02"):
        process_prefix(prefix, drive[prefix], out, workers=2)

    hashes = load_hashes(out / "00.phash.npy") | load_hashes(out / "01.phash.npy") | load_hashes(out / "02.phash.npy")
    assert np.load(out / "00.phash.npy")["phash"].dtype == np.uint64
    assert set(hashes) == {SQUARE, EXTRA, WIDE, SMALL, COPY, OTHER}
    assert bits_apart(hashes[SQUARE], hashes[COPY]) <= 8
    assert bits_apart(hashes[SQUARE], hashes[OTHER]) > 16
    assert bits_apart(hashes[SQUARE], hashes[EXTRA]) > 16


def test_a_done_prefix_is_skipped_until_its_cover_list_changes(drive, tmp_path):
    out = tmp_path / "gallery256"
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
    out = tmp_path / "gallery256"
    real_shrink = shard_and_hash.shrink_and_hash

    def shrink_then_unmount(path):
        result = real_shrink(path)
        shutil.rmtree(config.IMAGE_ROOT, ignore_errors=True)  # the drive goes away after the first cover
        return result

    monkeypatch.setattr(shard_and_hash, "shrink_and_hash", shrink_then_unmount)
    with pytest.raises(SystemExit):
        process_prefix("00", drive["00"], out, workers=1)
    assert not is_done("00", drive["00"], out)
    assert not (out / "00.tar").exists()
    assert not (out / "00.errors.log").exists()  # stopped at the first unreadable cover, logged nothing


def test_a_read_error_leaves_the_prefix_pending_but_a_missing_file_does_not(drive, tmp_path):
    out = tmp_path / "gallery256"
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
