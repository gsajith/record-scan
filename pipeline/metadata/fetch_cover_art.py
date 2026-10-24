import asyncio
import aiohttp
import aiofiles
import os
import json
import time
from pathlib import Path

# ---------------- CONFIG ---------------- #
INPUT_JSON = "slim_releases_with_covers.json"

OUTPUT_DIR = "/Volumes/Seagate Portabl/Pictures/record_scan_cover_images"
FAILED_FILE = Path("failed_cover_fetches.json")
EXISTING_FILE_LIST = Path("existing_files.json")

NUM_TO_FETCH = 5555555               # How many to fetch this run
CONCURRENCY = 20                     # Parallelism
IMAGE_SIZE_SUFFIX = "-500"           # "-1200" for higher resolution

TOTAL_ENTRIES_ESTIMATE = 3_129_812
SAVE_EXISTING_INTERVAL = 100         # Save existing_files JSON every N downloads
SAVE_FAILED_INTERVAL = 100           # Save failed map every N failures
# ---------------------------------------- #

os.makedirs(OUTPUT_DIR, exist_ok=True)

async def download_image(session, release, sem, idx, total):
    """Download one cover image."""
    release_id = release["id"]
    cover_id = release.get("cover_id")

    if cover_id is None:
        return f"skip2:{release_id}:{idx}:{total}:no cover ID"

    url = f"https://coverartarchive.org/release/{release_id}/{cover_id}{IMAGE_SIZE_SUFFIX}.jpg"
    out_path = os.path.join(OUTPUT_DIR, f"{release_id}.jpg")

    async with sem:
        try:
            t0 = time.time()
            async with session.get(url, timeout=30) as resp:
                resp.raise_for_status()
                content = await resp.read()

            async with aiofiles.open(out_path, "wb") as f:
                await f.write(content)

            # Debug log
            print(f"[{idx}/{total}] Saved {release_id} in {time.time()-t0:.2f}s", flush=True)
            return f"ok:{release_id}:{idx}:{total}"
        except Exception as e:
            return f"fail:{release_id}:{idx}:{total}:{type(e).__name__}:{e}"


async def main():
    # Load JSON
    print("Loading JSON into memory...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data):,} records")

    # Load or scan existing files
    if EXISTING_FILE_LIST.exists():
        with open(EXISTING_FILE_LIST, "r", encoding="utf-8") as f:
            existing_files = set(json.load(f))
        print(f"Loaded {len(existing_files):,} existing files from JSON")
    else:
        print("Scanning output directory for existing images...")
        existing_files = set()
        count = 0
        for entry in os.scandir(OUTPUT_DIR):
            if entry.is_file() and entry.name.endswith(".jpg"):
                existing_files.add(entry.name.split(".")[0])
            count += 1
            if count % 100_000 == 0:
                print(f"Scanned {count:,} files so far...", flush=True)
        print(f"Found {len(existing_files):,} existing images")
        # Save initial scan
        with open(EXISTING_FILE_LIST, "w", encoding="utf-8") as f:
            json.dump(list(existing_files), f)

    # Filter out already-downloaded entries
    data = [r for r in data if r["id"] not in existing_files]
    print(f"{len(data):,} entries remain to fetch after filtering existing images")
    targets = data[:NUM_TO_FETCH]
    print(f"Prepared {len(targets):,} targets to fetch")

    # Load failed map if exists
    if FAILED_FILE.exists():
        with open(FAILED_FILE, "r", encoding="utf-8") as f:
            failed_map = json.load(f)
        print(f"Loaded {len(failed_map):,} failed entries")
    else:
        failed_map = {}

    sem = asyncio.Semaphore(CONCURRENCY)
    start = time.time()
    downloaded_count = 0
    completed_count = 0

    async with aiohttp.ClientSession() as session:
        tasks = [
            download_image(session, r, sem, idx, TOTAL_ENTRIES_ESTIMATE)
            for idx, r in enumerate(targets, start=1)
        ]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed_count += 1
            parts = result.split(":", 5)
            status, release_id, idx_str, total_str, *rest = parts
            idx = int(idx_str)
            total = int(total_str)

            if status == "ok":
                downloaded_count += 1
                existing_files.add(release_id)

                if downloaded_count % SAVE_EXISTING_INTERVAL == 0:
                    async with aiofiles.open(EXISTING_FILE_LIST, "w", encoding="utf-8") as f:
                        await f.write(json.dumps(list(existing_files), ensure_ascii=False))
                    print(f"Saved {len(existing_files):,} existing files to JSON")

                print(f"Saved {release_id} ({completed_count:,} / {total:,}) - ({idx:,})", flush=True)

            elif status == "skip2":
                reason = rest[0] if rest else "unknown"
                print(f"No cover ID {release_id} ({completed_count:,} / {total:,}, {reason}) - ({idx:,})", flush=True)

            elif status == "fail":
                err_type = rest[0] if rest else "Error"
                err_msg = rest[1] if len(rest) > 1 else ""
                print(f"Failed {release_id} ({completed_count:,} / {total:,}) - ({idx:,}): {err_type} {err_msg}", flush=True)
                failed_map[release_id] = {"idx": idx, "error": f"{err_type} {err_msg}"}

                if len(failed_map) % SAVE_FAILED_INTERVAL == 0:
                    async with aiofiles.open(FAILED_FILE, "w", encoding="utf-8") as f:
                        await f.write(json.dumps(failed_map, ensure_ascii=False, indent=2))
                    print(f"Saved {len(failed_map):,} failed entries to JSON")

    elapsed = time.time() - start
    if downloaded_count > 0:
        avg_per_item = elapsed / downloaded_count
        items_per_hour = downloaded_count / elapsed * 3600
        projected_hours = TOTAL_ENTRIES_ESTIMATE / items_per_hour
    else:
        avg_per_item = 0
        items_per_hour = 0
        projected_hours = 0

    print("\n--- Timing Summary ---")
    print(f"Downloaded {downloaded_count} images in {elapsed:.2f} seconds")
    print(f"Average per item: {avg_per_item:.2f} sec")
    print(f"Throughput: {items_per_hour:,.0f} items/hour")
    print(f"Projected time for {TOTAL_ENTRIES_ESTIMATE:,}: {projected_hours:.1f} hours")

    # Final save of failed map and existing_files
    async with aiofiles.open(FAILED_FILE, "w", encoding="utf-8") as f:
        await f.write(json.dumps(failed_map, ensure_ascii=False, indent=2))
    async with aiofiles.open(EXISTING_FILE_LIST, "w", encoding="utf-8") as f:
        await f.write(json.dumps(list(existing_files), ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
