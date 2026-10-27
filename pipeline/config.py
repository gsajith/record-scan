"""Paths, environment and shared checks for the pipeline.

Values are computed from the environment at import time. Other modules read them at
call time (`from pipeline import config`, then `config.IMAGE_ROOT`) so tests can set
RS_DATA_DIR or RS_IMAGE_ROOT and `importlib.reload(config)`.
"""

import os
import re
from pathlib import Path

DATA_DIR = Path(os.environ.get("RS_DATA_DIR", "~/Projects/record-scan-data")).expanduser()
IMAGE_ROOT = Path(
    os.environ.get(
        "RS_IMAGE_ROOT",
        "/Volumes/Seagate Portabl/record scan Pictures/record_scan_cover_images_split",
    )
).expanduser()

META_DIR = DATA_DIR / "metadata"
EMB_DIR = DATA_DIR / "embeddings"
MODELS_DIR = DATA_DIR / "models"
GOLD_DIR = DATA_DIR / "gold"
DETECTOR_DIR = DATA_DIR / "detector"
WILD_DIR = DATA_DIR / "wild"
GALLERY256_DIR = DATA_DIR / "gallery256"
EXPORTS_DIR = DATA_DIR / "exports"
LIGHTBANK_DIR = DATA_DIR / "lightbank"
CATALOG_DB = META_DIR / "catalog.sqlite"

REPO_ROOT = Path(__file__).resolve().parent.parent
# Measured numbers live in the repo, one file per run
RESULTS_DIR = REPO_ROOT / "docs" / "results"

# MusicBrainz release ids, lowercase. Checked before an id goes into a URL or a path.
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def is_uuid(s: str) -> bool:
    # fullmatch, because `$` alone also matches before a trailing newline
    return UUID_RE.fullmatch(s) is not None


def require_image_root() -> Path:
    """Exit unless the cover image drive is mounted. Prefix folder "00" proves it."""
    if not (IMAGE_ROOT / "00").is_dir():
        raise SystemExit(
            f"Cover images not found at {IMAGE_ROOT}. Mount the drive or set RS_IMAGE_ROOT."
        )
    return IMAGE_ROOT


def image_path(uuid: str) -> Path:
    """Where the 2025 download put a release's cover. Later covers have other names and some 2025
    ones were dropped: use Catalog.covers() for a release's current covers."""
    return IMAGE_ROOT / uuid[:2] / f"{uuid}.jpg"


def device() -> str:
    import torch  # deferred so scripts that never touch a model skip the torch import

    return "mps" if torch.backends.mps.is_available() else "cpu"
