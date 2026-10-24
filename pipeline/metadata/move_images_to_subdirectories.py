import json
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
json_file = "./existing_files.json"
base_dir = Path("/Volumes/Seagate Portabl/Pictures/record_scan_cover_images")
output_dir = Path("/Volumes/Seagate Portabl/Pictures/record_scan_cover_images_split")
output_dir.mkdir(exist_ok=True)

# Load list of album IDs
with open(json_file, "r") as f:
    album_ids = json.load(f)

total = len(album_ids)
print(f"Processing {total} album covers...")


def move_one(album_id):
    src = base_dir / f"{album_id}.jpg"
    subfolder = output_dir / album_id[:2]
    subfolder.mkdir(exist_ok=True)
    dst = subfolder / f"{album_id}.jpg"
    try:
        shutil.move(str(src), str(dst))
        return True
    except FileNotFoundError:
        return False

# Use ThreadPoolExecutor to move files in parallel
batch_size = 1000  # chunk progress printing
processed = 0

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(move_one, album_id): album_id for album_id in album_ids}
    for idx, future in enumerate(as_completed(futures), start=1):
        processed += 1
        if processed % batch_size == 0:
            print(f"Processed {processed}/{total} files ({processed/total:.2%})")

print("Done!")
