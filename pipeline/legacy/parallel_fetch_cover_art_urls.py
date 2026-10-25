# Run from the repo root: python -m pipeline.metadata.parallel_fetch_cover_art_urls
import asyncio
import aiohttp
import json

from pipeline import config

# File names, all in config.META_DIR
INPUT_FILE = "slim_releases.json"
OUTPUT_FILE = "slim_releases_with_covers.json"
FAILED_FILE = "failed_cover_id_fetches.json"

# Config
API_BASE = "https://coverartarchive.org/release/"
CONCURRENCY = 64       # number of parallel workers
SAVE_EVERY = 1000      # save progress every N releases
REQUEST_DELAY = 0.05  # delay between requests per worker (seconds)


async def fetch_cover_id(session, release_id):
    """
    Query Cover Art Archive for a release ID.
    Return the cover image ID if available
    Priority:
      1. Approved front covers
      2. Unapproved front covers
      3. Any approved cover
    """
    url = API_BASE + release_id
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status == 404:
                return None
            resp.raise_for_status()
            data = await resp.json()

            best_cover = None
            best_priority = 0  # higher number = higher priority

            for image in data.get("images", []):
                thumbs = image.get("thumbnails", {})
                if not ("500" in thumbs or "large" in thumbs):
                    continue

                is_front = image.get("front") or "Front" in image.get("types", [])
                is_approved = image.get("approved", False)

                # Assign priority
                if is_front and is_approved:
                    priority = 3
                elif is_front:
                    priority = 2
                elif is_approved:
                    priority = 1
                else:
                    priority = 0

                if priority > best_priority:
                    best_priority = priority
                    best_cover = image["id"]

                # Stop early if highest priority
                if best_priority == 3:
                    break

            return best_cover

    except Exception as e:
        print(f"Request failed for {release_id}: {type(e).__name__}: {e}")
        return None


async def save_progress(releases, cover_map, failed_map):
    """Save intermediate progress in a background thread."""
    await asyncio.to_thread(write_files, releases, cover_map, failed_map)


def write_files(releases, cover_map, failed_map):
    # Save the full releases file
    with open(config.META_DIR / OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(releases, f, ensure_ascii=False, indent=2)
    # Save failed fetches only
    with open(config.META_DIR / FAILED_FILE, "w", encoding="utf-8") as f:
        json.dump(failed_map, f, ensure_ascii=False, indent=2)


async def worker(name, session, queue, releases, cover_map, failed_map, new_ids, save_lock):
    while True:
        item = await queue.get()
        if item is None:
            break
        i, release = item
        rid = release["id"]

        # skip if already has cover_id
        if rid in cover_map and cover_map[rid] is not None:
            release["cover_id"] = cover_map[rid]
            queue.task_done()
            continue

        # fetch cover ID
        cover_id = await fetch_cover_id(session, rid)
        release["cover_id"] = cover_id
        cover_map[rid] = cover_id

        # log failed fetches only
        if cover_id is None:
            failed_map[rid] = {"title": release["title"]}
        else:
            new_ids = new_ids + 1

        print(f"[{i+1}/{len(releases)}] {release['title']} → {cover_id}")

        # safe periodic save
        if (i + 1) % SAVE_EVERY == 0:
            async with save_lock:
                await save_progress(releases, cover_map, failed_map)

        await asyncio.sleep(REQUEST_DELAY)
        queue.task_done()


async def main():
    output_file = config.META_DIR / OUTPUT_FILE
    failed_file = config.META_DIR / FAILED_FILE

    # Load releases
    with open(config.META_DIR / INPUT_FILE, "r", encoding="utf-8") as f:
        releases = json.load(f)

    # Resume existing output if present
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            processed = json.load(f)
        cover_map = {r["id"]: r.get("cover_id") for r in processed}
    else:
        cover_map = {}

    # Resume failed fetches
    if failed_file.exists():
        with open(failed_file, "r", encoding="utf-8") as f:
            failed_map = json.load(f)
    else:
        failed_map = {}

    # Prepare queue
    queue = asyncio.Queue()
    for i, release in enumerate(releases):
        queue.put_nowait((i, release))

    save_lock = asyncio.Lock()

    new_ids = 0

    async with aiohttp.ClientSession() as session:
        workers = [
            asyncio.create_task(worker(f"w{n}", session, queue, releases, cover_map, failed_map, new_ids, save_lock))
            for n in range(CONCURRENCY)
        ]

        await queue.join()  # wait for all tasks to finish

        # cancel worker tasks
        for w in workers:
            w.cancel()

    # final save
    await save_progress(releases, cover_map, failed_map)
    print(f"✅ Finished processing {len(releases)} releases, ${new_ids} new.")


if __name__ == "__main__":
    asyncio.run(main())
