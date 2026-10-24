# Historical: moved the flat image folder into IMAGE_ROOT/<uuid[:2]>/ once. The split
# layout is now canonical. Run from the repo root:
#   python -m pipeline.metadata.move_images_to_subdirectories <flat source folder>
import argparse
import json
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline import config


def main():
    parser = argparse.ArgumentParser(description="Move flat <uuid>.jpg files into prefix folders.")
    parser.add_argument("base_dir", type=Path, help="flat folder holding <uuid>.jpg files")
    base_dir = parser.parse_args().base_dir

    # Paths
    output_dir = config.require_image_root()
    json_file = config.META_DIR / "existing_files.json"

    # Load list of album IDs
    with open(json_file, "r") as f:
        album_ids = json.load(f)

    valid_ids = [album_id for album_id in album_ids if config.is_uuid(album_id)]
    print(f"Skipped {len(album_ids) - len(valid_ids)} malformed album IDs")
    album_ids = valid_ids

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


if __name__ == "__main__":
    main()
