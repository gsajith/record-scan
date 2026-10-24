"""
export_yolo_dataset.py
----------------------
Exports annotated album cover data from Supabase into YOLOv8 dataset format.
Run from the repo root: python -m pipeline.detector.export_yolo_dataset

Reads from:
  - Supabase `annotations` table (image_url, x_center, y_center, width, height)
  - Supabase storage bucket containing images
  - Supabase credentials from the repo root .env

Writes to config.DETECTOR_DIR:
  album_dataset/
    ├── images/train/
    ├── images/val/
    ├── labels/train/
    └── labels/val/
  album_cover.yaml (YOLO dataset config with the absolute dataset path)

Performs an 85/15 train/val split automatically.
Skips negative samples (class_id = -1 or null bbox), but keeps images so YOLO sees them as "no album" examples.
Only downloads https URLs on the Supabase project's own host.
"""

import json
import os
import random
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm
from supabase import create_client, Client
from dotenv import load_dotenv

from pipeline import config

# ------------
# CONFIG
# ------------
TRAIN_RATIO = 0.85


# ------------
# UTILS
# ------------
def download_image(url: str, out_path: str):
    """Download image from Supabase public URL."""
    r = requests.get(url, stream=True, timeout=30, allow_redirects=False)
    if r.status_code == 200:
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    return False


def is_supabase_url(url: str, supabase_netloc: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.netloc == supabase_netloc


def write_yolo_label(rec, out_path):
    """Write a single YOLO label file."""
    x, y, w, h = rec["x_center"], rec["y_center"], rec["width"], rec["height"]
    cls = rec.get("class_id", 0)
    with open(out_path, "w") as f:
        f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def export_split(dataset_dir, split_name, records, supabase_netloc):
    img_dir = f"{dataset_dir}/images/{split_name}"
    lbl_dir = f"{dataset_dir}/labels/{split_name}"
    print(f"Exporting {split_name} split ({len(records)} images)...")

    for rec in tqdm(records):
        url = rec["image_url"]

        # Skip entries with no URL, or a URL off the Supabase host
        if not url:
            continue
        if not is_supabase_url(url, supabase_netloc):
            print(f"⚠️ Skipping non-Supabase URL: {url}")
            continue

        img_id = os.path.splitext(os.path.basename(url))[0]
        img_path = f"{img_dir}/{img_id}.jpg"
        lbl_path = f"{lbl_dir}/{img_id}.txt"

        # Download image if missing
        if not os.path.exists(img_path):
            ok = download_image(url, img_path)
            if not ok:
                print(f"⚠️ Skipping failed download: {url}")
                continue

        # Skip negative samples for label writing
        if (
            rec.get("class_id") == -1
            or any(rec.get(k) is None for k in ("x_center", "y_center", "width", "height"))
        ):
            # keep image, skip label file
            continue

        write_yolo_label(rec, lbl_path)


def write_dataset_yaml(dataset_dir: Path):
    """Write album_cover.yaml next to the dataset. JSON-quoting keeps odd paths valid YAML."""
    yaml_path = dataset_dir.parent / "album_cover.yaml"
    yaml_path.write_text(
        f"path: {json.dumps(str(dataset_dir.resolve()))}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: album_cover\n"
    )
    print(f"Wrote {yaml_path}")


def main():
    load_dotenv(Path(__file__).parents[2] / ".env")
    supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    supabase_key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    dataset_dir = config.DETECTOR_DIR / "album_dataset"

    # ------------
    # SETUP
    # ------------
    supabase: Client = create_client(supabase_url, supabase_key)
    supabase_netloc = urlparse(supabase_url).netloc
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(dataset_dir / sub, exist_ok=True)

    # ------------
    # FETCH ANNOTATIONS
    # ------------
    print("Fetching annotations from Supabase...")
    res = supabase.table("annotations").select("*").execute()
    records = res.data or []
    print(f"Fetched {len(records)} total annotations")

    random.shuffle(records)
    split_idx = int(len(records) * TRAIN_RATIO)
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    # ------------
    # EXPORT
    # ------------
    export_split(dataset_dir, "train", train_records, supabase_netloc)
    export_split(dataset_dir, "val", val_records, supabase_netloc)
    write_dataset_yaml(dataset_dir)

    print("✅ Dataset export complete!")
    print(f"Train: {len(train_records)}  |  Val: {len(val_records)}")


if __name__ == "__main__":
    main()
