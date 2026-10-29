"""Measure the gallery's covers to find the ones that aren't usable covers.

    python -m pipeline.gallery.exclusions measure [--workers 9]

Some downloaded fronts are placeholders, blanks, or strips and spines. This module records what
it takes to find them, from the shards on the internal disk, with no drive. Nothing on the drive
or in the shards is changed or deleted.

measure writes, per prefix, to config.GALLERY_DIR:

    <prefix>.spread.npy     key and spread (see spread()) of every cover in <prefix>.tar
    <prefix>.spread.recipe  SPREAD_RECIPE and the prefix's done marker, so a redone copy or a
                            changed measure makes the prefix measure again

load_measures() joins them with each cover's width and height from <prefix>.covers.npy.
"""

import argparse
import io
import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

from pipeline import config

# Change it whenever spread() would give different numbers, so old measures are refused
SPREAD_RECIPE = f"rgb-std64-pillow{PIL.__version__}-numpy{np.__version__}"
SPREAD_DTYPE = np.dtype([("key", "U64"), ("spread", np.float32)])
MEASURES_DTYPE = np.dtype([("key", "U64"), ("width", np.uint32), ("height", np.uint32), ("spread", np.float32)])
SIDE = 64


def spread(img: Image.Image) -> float:
    """How far the pixels stray from the image's mean color: the RGB standard deviation of a 64x64
    copy, the root of the mean of the three channels' variances, on a 0 to 255 scale. A blank is
    near 0. The small copy ignores JPEG noise and fine grain."""
    img.draft("RGB", (SIDE, SIDE))  # a JPEG decodes at 1/2 to 1/8 scale, still at least 64 px
    pixels = np.asarray(img.convert("RGB").resize((SIDE, SIDE), Image.Resampling.BOX), dtype=np.float64)
    return float(np.sqrt(pixels.reshape(-1, 3).var(axis=0).mean()))


def _recipe(prefix: str, gallery_dir: Path) -> str:
    return f"{SPREAD_RECIPE} {(gallery_dir / f'{prefix}.done').read_text()}"


def _is_measured(prefix: str, gallery_dir: Path) -> bool:
    recipe = gallery_dir / f"{prefix}.spread.recipe"
    return (gallery_dir / f"{prefix}.spread.npy").exists() and recipe.exists() \
        and recipe.read_text() == _recipe(prefix, gallery_dir)


def measure_prefix(prefix: str, gallery_dir: Path) -> int:
    """Measure every cover in <prefix>.tar. Returns how many, 0 when the prefix is already measured."""
    if _is_measured(prefix, gallery_dir):
        return 0
    recipe = _recipe(prefix, gallery_dir)  # read before the shard, so a copy redone meanwhile is refused later
    rows = []
    with tarfile.open(gallery_dir / f"{prefix}.tar") as tar:
        for member in tar:
            with Image.open(io.BytesIO(tar.extractfile(member).read())) as img:
                rows.append((member.name.removesuffix(".jpg"), spread(img)))
    out = gallery_dir / f"{prefix}.spread.npy"
    (gallery_dir / f"{prefix}.spread.recipe").unlink(missing_ok=True)  # a failed write must not look current
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("wb") as f:  # a file handle stops numpy from appending another .npy
        np.save(f, np.array(rows, dtype=SPREAD_DTYPE))
    os.replace(tmp, out)
    (gallery_dir / f"{prefix}.spread.recipe").write_text(recipe)
    return len(rows)


def load_measures(gallery_dir: Path) -> np.ndarray:
    """MEASURES_DTYPE rows for every cover in the gallery, in covers.npy order, prefix by prefix.
    Refuses a prefix that isn't measured, or was measured from an older copy or another recipe."""
    parts = []
    for covers_file in sorted(gallery_dir.glob("*.covers.npy")):
        prefix = covers_file.name.removesuffix(".covers.npy")
        if not _is_measured(prefix, gallery_dir):
            raise ValueError(f"Prefix {prefix} isn't measured with this gallery and recipe; rerun exclusions measure")
        covers = np.load(covers_file)
        spreads = np.load(gallery_dir / f"{prefix}.spread.npy")
        by_key = dict(zip(spreads["key"].tolist(), spreads["spread"].tolist()))
        rows = np.empty(len(covers), dtype=MEASURES_DTYPE)
        for field in ("key", "width", "height"):
            rows[field] = covers[field]
        rows["spread"] = [by_key[k] for k in covers["key"].tolist()]  # the shard holds every cover in covers.npy
        parts.append(rows)
    return np.concatenate(parts) if parts else np.empty(0, dtype=MEASURES_DTYPE)


def _measure(prefix: str) -> tuple[str, int, float]:
    start = time.monotonic()
    return prefix, measure_prefix(prefix, config.GALLERY_DIR), time.monotonic() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("measure", help="record every cover's spread")
    measure.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    gallery_dir = config.GALLERY_DIR
    prefixes = sorted(f.name.removesuffix(".covers.npy") for f in gallery_dir.glob("*.covers.npy"))
    if not prefixes:
        raise SystemExit(f"No covers.npy files in {gallery_dir}; run shard_and_hash first")
    pending = [p for p in prefixes if not _is_measured(p, gallery_dir)]
    print(f"{len(pending)} of {len(prefixes)} prefixes to measure in {gallery_dir}", flush=True)
    start = time.monotonic()
    with ProcessPoolExecutor(args.workers) as pool:
        for prefix, n, secs in pool.map(_measure, pending):
            print(f"{prefix}: {n:,} covers in {secs:.0f} s", flush=True)
    print(f"Measured {len(pending)} prefixes in {time.monotonic() - start:.0f} s")


if __name__ == "__main__":
    main()
