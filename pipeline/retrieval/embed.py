"""Embed gallery covers from the drive, one shard per prefix folder.

    python -m pipeline.retrieval.embed --model NAME (--prefixes 00 01 | --all) [--batch 64] [--workers 6]

Covers come from the catalog's covers table, as in the gallery pass, not from globbing the drive.
Writes config.EMB_DIR/<model>/<prefix>.npz (uuids U36, embs float16, and the embedder's
signature) and a <prefix>.done marker holding that signature and a hash of the prefix's cover
list. A rerun skips a prefix only when both match, so a prefix done by another embedder (changed
preprocessing) or before a catalog update changed its covers is embedded again, and an
interrupted prefix starts over. Unreadable files, and covers whose file name is not a release id,
go to <prefix>.errors.log in the same folder and are skipped. If the drive goes away mid-prefix,
the run exits without marking the prefix done.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from pipeline import config
from pipeline.gallery.shard_and_hash import PREFIXES, cover_list_digest
from pipeline.metadata.catalog import Catalog
from pipeline.retrieval.embedders import EMBEDDER_NAMES, Embedder, load_embedder


def _decode_rgb(path: Path) -> Image.Image | Exception:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")  # convert forces the full decode, so truncated files fail here
    except Exception as e:  # PIL raises OSError, ValueError, SyntaxError and others for bad files
        return e


def embed_images(
    paths: list[Path], embedder: Embedder, batch: int, workers: int, errors_log: Path
) -> tuple[list[str], np.ndarray]:
    """Embed paths in order, keyed by file stem. Workers decode the next batch while this one embeds."""
    stems: list[str] = []
    chunks: list[np.ndarray] = []
    batches = [paths[i : i + batch] for i in range(0, len(paths), batch)]
    with ThreadPoolExecutor(workers) as pool, tqdm(total=len(paths), unit="img", leave=False) as bar:
        decoding = pool.map(_decode_rgb, batches[0]) if batches else iter(())
        for i, batch_paths in enumerate(batches):
            decoded = list(decoding)
            if i + 1 < len(batches):
                decoding = pool.map(_decode_rgb, batches[i + 1])  # map submits now, so decoding overlaps embed
            images = []
            for path, img in zip(batch_paths, decoded):
                if isinstance(img, Exception):
                    _log_error(errors_log, path, img)
                else:
                    stems.append(path.stem)
                    images.append(img)
            if images:
                chunks.append(embedder.embed(images))
            bar.update(len(batch_paths))
    embs = np.concatenate(chunks) if chunks else np.zeros((0, embedder.dim), np.float32)
    return stems, embs


def _log_error(errors_log: Path, path: Path, err: Exception | str) -> None:
    tqdm.write(f"Skipped {path}: {err}")
    with errors_log.open("a") as f:
        f.write(f"{path}\t{err}\n")


def embed_prefix(prefix: str, files: list[str], embedder: Embedder, out_dir: Path, batch: int, workers: int) -> int:
    """Embed one prefix folder's covers (paths relative to IMAGE_ROOT). Returns the number of covers
    written, 0 if the prefix is already done with this embedder and cover list."""
    if is_done(prefix, files, out_dir, embedder.signature):
        return 0
    root = config.require_image_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / f"{prefix}.done"
    done.unlink(missing_ok=True)  # a redo that fails partway must not look done
    errors_log = out_dir / f"{prefix}.errors.log"
    errors_log.unlink(missing_ok=True)  # a restarted prefix logs its errors afresh
    paths = []
    for path in sorted(root / f for f in files):
        if config.is_uuid(path.stem):
            paths.append(path)
        else:
            _log_error(errors_log, path, "file name is not a release id")
    uuids, embs = embed_images(paths, embedder, batch, workers, errors_log)
    # An unmounted drive makes every remaining file "unreadable"; don't save that as done
    config.require_image_root()
    tmp = out_dir / f"{prefix}.tmp.npz"
    with tmp.open("wb") as f:  # a file handle stops numpy from appending another .npz
        np.savez(f, uuids=np.array(uuids, dtype="U36"), embs=embs.astype(np.float16), signature=embedder.signature)
    os.replace(tmp, out_dir / f"{prefix}.npz")
    done.write_text(_marker(files, embedder.signature))
    return len(uuids)


def _marker(files: list[str], signature: str) -> str:
    return f"{signature} {cover_list_digest(files)}"


def is_done(prefix: str, files: list[str], out_dir: Path, signature: str) -> bool:
    """Embedded by the embedder with this signature from this cover list."""
    # Older markers hold the signature alone or nothing, so their prefixes count as pending
    done = out_dir / f"{prefix}.done"
    return done.exists() and done.read_text() == _marker(files, signature)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=EMBEDDER_NAMES)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--prefixes", nargs="+", choices=PREFIXES, metavar="PREFIX")
    which.add_argument("--all", action="store_true")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    config.require_image_root()
    embedder = load_embedder(args.model)
    out_dir = config.EMB_DIR / embedder.name
    catalog = Catalog()
    covers = {p: catalog.cover_files(p) for p in PREFIXES}
    pending = [p for p in PREFIXES if not is_done(p, covers[p], out_dir, embedder.signature)]
    prefixes = pending if args.all else [p for p in dict.fromkeys(args.prefixes) if p in pending]
    if skipped := [p for p in (args.prefixes or []) if p not in pending]:
        print(f"Skipping {' '.join(skipped)}: already done")
    for prefix in prefixes:
        start = time.monotonic()
        n = embed_prefix(prefix, covers[prefix], embedder, out_dir, args.batch, args.workers)
        secs = time.monotonic() - start
        print(f"{prefix}: {n} covers in {secs:.0f} s ({n / max(secs, 1e-9):.0f} img/s)")
    left = [p for p in PREFIXES if not is_done(p, covers[p], out_dir, embedder.signature)]
    print(f"{len(left)} prefixes left in {out_dir}")


if __name__ == "__main__":
    main()
