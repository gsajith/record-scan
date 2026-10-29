"""The exclusion list: gallery covers that aren't usable covers, kept out of the index and the eval.

    python -m pipeline.gallery.exclusions measure [--workers 9]
    python -m pipeline.gallery.exclusions write

Some downloaded fronts are placeholders, blanks, or strips and spines. Both steps read the shards
on the internal disk and need no drive. Nothing on the drive or in the shards is changed or
deleted: the list is a filter, so every call can be undone by changing a cutoff and rewriting it.

measure writes, per prefix, to config.GALLERY_DIR:

    <prefix>.spread.npy     key and spread (see spread()) of every cover in <prefix>.tar
    <prefix>.spread.recipe  SPREAD_RECIPE and the prefix's done marker, so a redone copy or a
                            changed measure makes the prefix measure again

write reads those, each cover's size from <prefix>.covers.npy and cluster_duplicates'
templates.npy, and writes excluded.npy (one row per excluded cover: key and reason, see
excluded()) with a .recipe naming the rule, its cutoffs and what it was made from.
load_exclusions() refuses a stale list.
"""

import argparse
import hashlib
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
from pipeline.gallery.cluster_duplicates import LOOKALIKE_DISTANCE, catalog_fingerprint, groups_recipe, load_templates

# Change it whenever spread() would give different numbers, so old measures are refused
SPREAD_RECIPE = f"rgb-std64-pillow{PIL.__version__}-numpy{np.__version__}"
SPREAD_DTYPE = np.dtype([("key", "U64"), ("spread", np.float32)])
MEASURES_DTYPE = np.dtype([("key", "U64"), ("width", np.uint32), ("height", np.uint32), ("spread", np.float32)])
EXCLUDED_DTYPE = np.dtype([("key", "U64"), ("reason", "U8")])
SIDE = 64
# Names excluded()'s logic in the list's recipe. Change it whenever excluded() would pick other
# covers from the same input and cutoffs, so an old list is refused.
EXCLUDE_RULE = "blank-strip-template-v1"
# Cutoffs, Gautham's pick from the contact sheets (docs/results/2026-10-28-gallery-sanitation.md)
BLANK_SPREAD = 2.0
STRIP_ASPECT = 2.5
TEMPLATE_ARTISTS = 2


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


def excluded(measures: np.ndarray, templates: np.ndarray, blank_spread: float, strip_aspect: float,
             template_artists: int) -> np.ndarray:
    """EXCLUDED_DTYPE rows, sorted by key: the covers that aren't usable covers, each with one
    reason, the first that applies of "blank" (spread under blank_spread), "strip" (long side over
    short side above strip_aspect) and "template" (in a templates.npy design over template_artists
    or more artists). Small covers stay: they still embed, and dropping one leaves its release
    unfindable."""
    long_side = np.maximum(measures["width"], measures["height"]).astype(np.float64)
    short_side = np.maximum(np.minimum(measures["width"], measures["height"]), 1).astype(np.float64)
    shared = set(templates["key"][templates["artists"] >= template_artists].tolist())
    reasons = np.full(len(measures), "", dtype=EXCLUDED_DTYPE["reason"])
    reasons[np.isin(measures["key"], list(shared))] = "template"
    reasons[long_side / short_side > strip_aspect] = "strip"
    reasons[measures["spread"] < blank_spread] = "blank"
    out = np.empty(int((reasons != "").sum()), dtype=EXCLUDED_DTYPE)
    out["key"], out["reason"] = measures["key"][reasons != ""], reasons[reasons != ""]
    return out[np.argsort(out["key"], kind="stable")]


def exclusions_recipe(gallery_dir: Path, catalog: str) -> str:
    """What excluded.npy was made from: the rule and its cutoffs, the spread recipe, and the
    templates' recipe (their rule, the catalog fingerprint and every prefix's done marker)."""
    covers_files = sorted(gallery_dir.glob("*.covers.npy"))
    return (f"{EXCLUDE_RULE} blank{BLANK_SPREAD} strip{STRIP_ASPECT} template-artists{TEMPLATE_ARTISTS} "
            f"{SPREAD_RECIPE} {groups_recipe(covers_files, LOOKALIKE_DISTANCE, catalog)}")


def write_exclusions(gallery_dir: Path, catalog_db: Path | None = None) -> np.ndarray:
    """Write gallery_dir/excluded.npy (EXCLUDED_DTYPE rows at this module's cutoffs) and its .recipe.
    Returns the rows."""
    catalog = catalog_fingerprint(Path(catalog_db or config.CATALOG_DB))
    recipe = exclusions_recipe(gallery_dir, catalog)  # taken first, so a prefix redone meanwhile makes it stale
    templates = load_templates(gallery_dir / "templates.npy", catalog_db=catalog_db)
    rows = excluded(load_measures(gallery_dir), templates, BLANK_SPREAD, STRIP_ASPECT, TEMPLATE_ARTISTS)
    path = gallery_dir / "excluded.npy"
    path.with_suffix(".recipe").unlink(missing_ok=True)  # a failed write must not look current
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, rows)
    os.replace(tmp, path)
    path.with_suffix(".recipe").write_text(recipe)
    return rows


def load_exclusions(gallery_dir: Path, catalog_db: Path | None = None) -> dict[str, str]:
    """{cover key: reason} from excluded.npy. Refuses a list made at other cutoffs, by another rule,
    or from a gallery, catalog or templates that have changed since. Reads the whole catalog to check."""
    path = gallery_dir / "excluded.npy"
    recipe = path.with_suffix(".recipe")
    expected = exclusions_recipe(gallery_dir, catalog_fingerprint(Path(catalog_db or config.CATALOG_DB)))
    if not recipe.exists() or recipe.read_text() != expected:
        raise ValueError(f"{path} is missing or stale; rerun python -m pipeline.gallery.exclusions write")
    rows = np.load(path)
    return dict(zip(rows["key"].tolist(), rows["reason"].tolist()))


def exclusions_hash(exclusions: dict[str, str]) -> str:
    """A short hash of the excluded keys and reasons, for the eval's gallery hash."""
    lines = "\n".join(f"{key}\t{reason}" for key, reason in sorted(exclusions.items()))
    return hashlib.sha256(lines.encode()).hexdigest()[:12]


def _measure(prefix: str) -> tuple[str, int, float]:
    start = time.monotonic()
    return prefix, measure_prefix(prefix, config.GALLERY_DIR), time.monotonic() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("measure", help="record every cover's spread")
    measure.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    sub.add_parser("write", help="write excluded.npy at this module's cutoffs")
    args = parser.parse_args()

    gallery_dir = config.GALLERY_DIR
    if args.command == "write":
        rows = write_exclusions(gallery_dir)
        reasons, counts = np.unique(rows["reason"], return_counts=True)
        releases = len(np.unique(rows["key"].astype("U36")))  # mbid is key[:36]
        print(f"Excluded {len(rows):,} covers of {releases:,} releases into {gallery_dir / 'excluded.npy'}: "
              + ", ".join(f"{r} {n:,}" for r, n in zip(reasons.tolist(), counts.tolist())))
        return
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
