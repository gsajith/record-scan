"""Copy every cover off the drive into WebDataset shards, unchanged, with a perceptual hash and size each.

    python -m pipeline.gallery.shard_and_hash (--prefixes 00 01 | --all) [--workers 6]

The copy frees later steps from the slow USB drive. It keeps the original bytes because the
input size is unknown until the frozen sweep picks a model; sized copies are made from it later.
Covers come from the catalog's covers table, not from globbing the drive, which also holds covers
the catalog dropped. Each prefix folder ("00" to "ff") is read from the drive in file name order
and written to config.GALLERY_DIR on the internal disk:

    <prefix>.tar         WebDataset shard, one <file stem>.jpg per cover: the drive file's bytes
    <prefix>.covers.npy  structured array: key (the file stem, <mbid> or <mbid>-<cover_id>),
                         phash (uint64, imagehash.phash of the original), width, height
    <prefix>.errors.log  covers skipped: corrupt, missing from the drive, or a read error
    <prefix>.done        the recipe and a hash of the prefix's cover list

A rerun skips prefixes done with the same recipe and cover list, so a catalog update or a change
to the hash redoes them. A read error leaves its prefix pending for the next run to retry; if the
drive drops out mid-prefix, the run stops. Nothing is written to the drive.
"""

import argparse
import hashlib
import io
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import PIL
import scipy.fft
from PIL import Image

from pipeline import config
from pipeline.metadata.catalog import Catalog

PREFIXES = [f"{i:02x}" for i in range(256)]
# Names everything that shapes the output, so done prefixes are redone when it changes: the bytes are
# copied as is, and the hash depends on the code in read_and_hash() and the libraries it calls.
RECIPE = f"original-phash8-pillow{PIL.__version__}-scipy{scipy.__version__}-numpy{np.__version__}"
COVERS_DTYPE = np.dtype([("key", "U64"), ("phash", np.uint64), ("width", np.uint32), ("height", np.uint32)])


class CorruptCover(Exception):
    """The file was read but is not a usable image. Unlike a read error, a rerun won't fix it."""


def phash(img: Image.Image) -> int:
    """64-bit DCT perceptual hash, computed as imagehash.phash does: the 8x8 lowest frequencies of
    a 32x32 gray image, one bit each for above or below their median. Lookalikes differ in few bits."""
    gray = np.asarray(img.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = np.asarray(scipy.fft.dctn(gray))[:8, :8]
    bits = (low > np.median(low)).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def read_and_hash(path: Path) -> tuple[bytes, int, int, int] | OSError | CorruptCover:
    """The cover's bytes, its phash, width and height. Returns the OSError if the file can't be read."""
    try:
        data = path.read_bytes()  # read apart from decoding, so drive trouble is never taken for a bad file
    except OSError as e:
        return e
    try:
        with Image.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")  # convert forces the full decode, so truncated files fail here
        return data, phash(img), img.width, img.height
    except Exception as e:  # PIL raises OSError, ValueError, SyntaxError and others for bad files
        return CorruptCover(f"{type(e).__name__}: {e}")


def signature(files: list[str]) -> str:
    """The recipe plus a hash of the cover list, so a catalog update that changes the list redoes the prefix."""
    digest = hashlib.sha1("\n".join(sorted(files)).encode()).hexdigest()[:16]
    return f"{RECIPE} {digest}"


def is_done(prefix: str, files: list[str], out_dir: Path) -> bool:
    done = out_dir / f"{prefix}.done"
    outputs = (out_dir / f"{prefix}.tar", out_dir / f"{prefix}.covers.npy")
    return done.exists() and all(p.exists() for p in outputs) and done.read_text() == signature(files)


def process_prefix(prefix: str, files: list[str], out_dir: Path, workers: int) -> tuple[int, int]:
    """Shard one prefix folder's covers (paths relative to IMAGE_ROOT), in name order. Returns
    (written, skipped), or (0, 0) when the prefix is already done with this cover list. A read error
    other than a missing file leaves the prefix pending, so a rerun tries those covers again."""
    if is_done(prefix, files, out_dir):
        return 0, 0
    root = config.require_image_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}.done").unlink(missing_ok=True)  # a redo that fails partway must not look done
    errors_log = out_dir / f"{prefix}.errors.log"
    errors_log.unlink(missing_ok=True)
    files = sorted(files)
    rows: list[tuple[str, int, int, int]] = []
    retry = False
    shard_tmp = out_dir / f"{prefix}.tar.tmp"
    with ThreadPoolExecutor(workers) as pool, tarfile.open(shard_tmp, "w") as tar:
        for file, result in zip(files, pool.map(lambda f: read_and_hash(root / f), files)):
            if isinstance(result, Exception):
                # An unmounted drive makes every remaining file unreadable: stop, leaving the prefix pending
                config.require_image_root()
                _log_error(errors_log, file, result)
                retry |= isinstance(result, OSError) and not isinstance(result, FileNotFoundError)
                continue
            data, cover_hash, width, height = result
            key = Path(file).stem
            info = tarfile.TarInfo(f"{key}.jpg")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            rows.append((key, cover_hash, width, height))
    rows_tmp = out_dir / f"{prefix}.covers.npy.tmp"
    with rows_tmp.open("wb") as f:  # a file handle stops numpy from appending another .npy
        np.save(f, np.array(rows, dtype=COVERS_DTYPE))
    os.replace(shard_tmp, out_dir / f"{prefix}.tar")
    os.replace(rows_tmp, out_dir / f"{prefix}.covers.npy")
    if not retry:
        (out_dir / f"{prefix}.done").write_text(signature(files))
    return len(rows), len(files) - len(rows)


def _log_error(errors_log: Path, file: str, err: Exception) -> None:
    print(f"Skipped {file}: {err}")
    with errors_log.open("a") as f:
        f.write(f"{file}\t{err}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--prefixes", nargs="+", choices=PREFIXES, metavar="PREFIX")
    which.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    config.require_image_root()
    out_dir = config.GALLERY_DIR
    catalog = Catalog()
    # Cover counts of every pending prefix, for an ETA over the whole gallery whichever prefixes run now
    pending = {p: len(files) for p in PREFIXES if not is_done(p, files := catalog.cover_files(p), out_dir)}
    todo = PREFIXES if args.all else list(dict.fromkeys(args.prefixes))
    if skipped := [p for p in todo if p not in pending]:
        print(f"Skipping {' '.join(skipped)}: already done")
    covers_left = sum(pending.values())
    print(f"{len(pending)} prefixes and {covers_left} covers pending in {out_dir}", flush=True)
    start = time.monotonic()
    covers_run = 0
    for prefix in [p for p in todo if p in pending]:
        files = catalog.cover_files(prefix)
        prefix_start = time.monotonic()
        written, errors = process_prefix(prefix, files, out_dir, args.workers)
        secs = time.monotonic() - prefix_start
        covers_run += len(files)
        if is_done(prefix, files, out_dir):
            covers_left -= pending.pop(prefix)
            status = ""
        else:
            status = " LEFT PENDING after read errors, rerun to retry;"
        eta_hours = covers_left * (time.monotonic() - start) / max(covers_run, 1) / 3600
        shard_mb = (out_dir / f"{prefix}.tar").stat().st_size / 1e6
        print(f"{prefix}: {written} covers, {errors} skipped, {secs:.0f} s ({len(files) / max(secs, 1e-9):.0f} img/s), "
              f"shard {shard_mb:.0f} MB;{status} gallery has {len(pending)} prefixes, {covers_left} covers left, "
              f"about {eta_hours:.1f} h", flush=True)


if __name__ == "__main__":
    main()
