"""Embed gallery covers from the drive, one shard per prefix folder.

    python -m pipeline.retrieval.embed --model NAME (--prefixes 00 01 | --all) [--batch 64] [--workers 6]

Writes config.EMB_DIR/<model>/<prefix>.npz (uuids U36, embs float16, and the embedder's
signature) and a <prefix>.done marker holding that signature. A rerun skips prefixes done by
the same embedder; a prefix done by another one (changed preprocessing) is embedded again,
and an interrupted prefix starts over. Unreadable
files go to <prefix>.errors.log in the same folder and are skipped. If the drive goes away
mid-prefix, the run exits without marking the prefix done.
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


def embed_prefix(prefix: str, embedder: Embedder, out_dir: Path, batch: int, workers: int) -> int:
    """Embed one prefix folder of the drive. Returns the number of covers written, 0 if already done."""
    done = out_dir / f"{prefix}.done"
    if _done_by(done, embedder.signature):
        return 0
    root = config.require_image_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    errors_log = out_dir / f"{prefix}.errors.log"
    errors_log.unlink(missing_ok=True)  # a restarted prefix logs its errors afresh
    paths = []
    for path in sorted((root / prefix).glob("*.jpg")):
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
    done.write_text(embedder.signature)
    return len(uuids)


def _done_by(done: Path, signature: str) -> bool:
    # Markers from before signatures are empty, so their prefixes count as pending
    return done.exists() and done.read_text() == signature


def pending_prefixes(out_dir: Path, signature: str) -> list[str]:
    """Prefix folders on the drive not yet embedded in out_dir by the embedder with this signature."""
    root = config.require_image_root()
    prefixes = sorted(p.name for p in root.iterdir() if p.is_dir() and len(p.name) == 2)
    return [p for p in prefixes if not _done_by(out_dir / f"{p}.done", signature)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=EMBEDDER_NAMES)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--prefixes", nargs="+")
    which.add_argument("--all", action="store_true")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    embedder = load_embedder(args.model)
    out_dir = config.EMB_DIR / embedder.name
    pending = pending_prefixes(out_dir, embedder.signature)
    prefixes = pending if args.all else [p for p in args.prefixes if p in pending]
    if skipped := [p for p in (args.prefixes or []) if p not in pending]:
        print(f"Skipping {' '.join(skipped)}: already done or not a prefix folder on the drive")
    for prefix in prefixes:
        start = time.monotonic()
        n = embed_prefix(prefix, embedder, out_dir, args.batch, args.workers)
        secs = time.monotonic() - start
        print(f"{prefix}: {n} covers in {secs:.0f} s ({n / max(secs, 1e-9):.0f} img/s)")
    print(f"{len(pending_prefixes(out_dir, embedder.signature))} prefixes left in {out_dir}")


if __name__ == "__main__":
    main()
