"""
export_yolo_dataset.py
----------------------
Exports annotated album cover data from Supabase into YOLOv8 dataset format.

Reads from:
  - Supabase `annotations` table (image_url, x_center, y_center, width, height)
  - Supabase storage bucket containing images

Writes to:
  album_dataset/
    ├── images/train/
    ├── images/val/
    ├── labels/train/
    └── labels/val/

Performs an 85/15 train/val split automatically.
Skips negative samples (class_id = -1 or null bbox), but keeps images so YOLO sees them as "no album" examples.
"""

import os
import random
import requests
from tqdm import tqdm
from supabase import create_client, Client
from dotenv import load_dotenv

# ------------
# CONFIG
# ------------
load_dotenv(dotenv_path="album-annotator/.env.local")
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
DATASET_DIR = "album_dataset"
TRAIN_RATIO = 0.85

# ------------
# SETUP
# ------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
os.makedirs(f"{DATASET_DIR}/images/train", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/images/val", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/labels/val", exist_ok=True)

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
# UTILS
# ------------
def download_image(url: str, out_path: str):
    """Download image from Supabase public URL."""
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    return False


def write_yolo_label(rec, out_path):
    """Write a single YOLO label file."""
    x, y, w, h = rec["x_center"], rec["y_center"], rec["width"], rec["height"]
    cls = rec.get("class_id", 0)
    with open(out_path, "w") as f:
        f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def export_split(split_name, records):
    img_dir = f"{DATASET_DIR}/images/{split_name}"
    lbl_dir = f"{DATASET_DIR}/labels/{split_name}"
    print(f"Exporting {split_name} split ({len(records)} images)...")

    for rec in tqdm(records):
        url = rec["image_url"]
        img_id = os.path.splitext(os.path.basename(url))[0]
        img_path = f"{img_dir}/{img_id}.jpg"
        lbl_path = f"{lbl_dir}/{img_id}.txt"

        # Skip entries with no URL
        if not url:
            continue

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


# ------------
# EXPORT
# ------------
export_split("train", train_records)
export_split("val", val_records)

print("✅ Dataset export complete!")
print(f"Train: {len(train_records)}  |  Val: {len(val_records)}")
