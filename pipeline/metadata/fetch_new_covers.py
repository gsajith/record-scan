"""Bring the cover set up to date with a newer MusicBrainz dump.

    python -m pipeline.metadata.fetch_new_covers --date 2026-10-24 --since 2025-08-13 [--limit N] [--dry-run]

Run pipeline.metadata.slim_file on the dump's release.tar.xz first. Reads:
- META_DIR/releases-<date>.jsonl: releases with cover art in the new dump
- DATA_DIR/musicbrainz-dump-<date>/mbdump/cover_art_archive.{cover_art,cover_art_type}: every
  image on the Cover Art Archive, from the full export's mbdump-cover-art-archive.tar.bz2
- META_DIR/covers-<since>.jsonl: the covers the previous update left (for the 2025 dump,
  written once by pipeline.legacy.seed_covers_2025). A fixed snapshot, so rebuilding the
  catalog mid-run never changes the plan.
Keeps old covers that are still upstream, downloads fronts uploaded since --since for releases
we already have, and asks coverartarchive.org for the best front of releases with no cover left
(new releases, and ones whose cover was deleted). New images land on the drive beside the 2025
covers as <mbid>-<cover_id>.jpg, and a file that already exists is never replaced. Writes the
full list of covers to META_DIR/covers-<date>.jsonl for build_catalog_db. Rerunning resumes
from META_DIR/fetched-<date>.jsonl, which logs every image landed and every lookup that found
no usable cover; the drive itself is never listed, since 3M files take too long to scan.
"""

import argparse
import json
import os
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from itertools import chain, zip_longest
from pathlib import Path

import requests

from pipeline import config
from pipeline.metadata.catalog import Cover

USER_AGENT = "RecordScan/0.1 (gsajith@umich.edu)"
RETRY_STATUS = {429, 500, 502, 503, 504}
# Connections that fail before or while the body arrives; all worth a retry
DROPPED = (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
           requests.exceptions.ContentDecodingError)
CAA = "https://coverartarchive.org/release"
FRONT_TYPE = "1"  # cover_art_archive.art_type id for "Front"


@dataclass
class Plan:
    keep: list[Cover] = field(default_factory=list)      # covers that stay, files already on the drive
    download: list[Cover] = field(default_factory=list)  # new fronts on releases we already have
    lookup: list[str] = field(default_factory=list)      # releases with no cover left: ask the CAA for one
    dropped: list[Cover] = field(default_factory=list)   # deleted upstream, or the release left MusicBrainz


def new_file(mbid: str, cover_id: str) -> str:
    """Files this job downloads carry the cover id, so they never overwrite a 2025 <mbid>.jpg."""
    return f"{mbid[:2]}/{mbid}-{cover_id}.jpg"


def plan_covers(
    previous: dict[str, list[Cover]],
    releases: Iterable[str],
    cover_release: dict[int, int],
    new_fronts: dict[int, list[int]],
    fronts: set[int],
) -> Plan:
    """Decide what happens to each cover, without touching the network.

    previous: the previous update's covers by mbid. releases: mbids with cover art in the new dump.
    cover_release: CAA image id -> MusicBrainz internal release id, for images still upstream.
    new_fronts: internal release id -> front image ids uploaded since the previous dump.
    fronts: which of the previous image ids are typed Front. The primary should be a front:
    2025 fell back to any approved image when a release had no front, so a new front replaces it.
    The cover dump keys images by internal release id, so an old cover id still upstream is
    the only link from an mbid to its new fronts. Releases with none left get a CAA lookup.
    """
    plan = Plan()
    releases = set(releases)
    for mbid, covers in previous.items():
        if mbid not in releases:
            plan.dropped += covers
            continue
        alive = [c for c in covers if int(c.cover_id) in cover_release]
        plan.dropped += [c for c in covers if c not in alive]
        if not alive:
            plan.lookup.append(mbid)
            continue
        known = {int(c.cover_id) for c in covers}
        downloads = [Cover(mbid, str(f), new_file(mbid, str(f)), primary=False)
                     for f in new_fronts.get(cover_release[int(alive[0].cover_id)], []) if f not in known]
        primary = next((c for c in alive if c.primary), None)
        if primary is None or int(primary.cover_id) not in fronts:
            best = next((c for c in alive if int(c.cover_id) in fronts), None)
            best = best or (downloads[0] if downloads else None) or primary or alive[0]
            alive = [replace(c, primary=c is best) for c in alive]
            downloads = [replace(c, primary=c is best) for c in downloads]
        plan.keep += alive
        plan.download += downloads
    plan.lookup += [m for m in releases if m not in previous]
    plan.lookup.sort()  # a fixed order, so --limit always takes the same releases
    return plan


def get_with_backoff(session, url: str, sleep: Callable[[float], None] = time.sleep, tries: int = 7) -> requests.Response | None:
    """GET url, or None on 404. Backs off 5 s, doubling to 60 s, on 429, 5xx and dropped connections.

    Raises once tries run out; the caller logs the release and a rerun retries it.
    """
    delay = 5.0
    for attempt in range(1, tries + 1):
        try:
            resp = session.get(url, timeout=30)
        except DROPPED:
            if attempt == tries:
                raise
        else:
            if resp.status_code == 404:
                return None
            if resp.status_code not in RETRY_STATUS or attempt == tries:
                resp.raise_for_status()
                return resp
        sleep(delay)
        delay = min(delay * 2, 60.0)
    raise AssertionError("unreachable")


def load_covers(path: Path) -> dict[str, list[Cover]]:
    """A covers-<date>.jsonl list, by mbid."""
    if not path.exists():
        raise SystemExit(f"{path} not found. For the 2025 dump, run: python -m pipeline.legacy.seed_covers_2025")
    covers: dict[str, list[Cover]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = Cover(**json.loads(line))
                covers.setdefault(c.mbid, []).append(c)
    return covers


def load_cover_dump(
    mbdump: Path, previous_ids: set[int], since: str
) -> tuple[dict[int, int], dict[int, list[int]], set[int]]:
    """Read the cover dump into plan_covers' cover_release, new_fronts and fronts."""
    cover_release: dict[int, int] = {}
    recent: dict[int, int] = {}
    with (mbdump / "cover_art_archive.cover_art").open(encoding="utf-8") as f:
        for line in f:
            # id, release, comment, edit, ordering, date_uploaded, edits_pending, mime_type, ...
            cols = line.split("\t", 8)
            image, release = int(cols[0]), int(cols[1])
            if image in previous_ids:
                cover_release[image] = release
            elif cols[5] >= since and cols[7] != "application/pdf":
                recent[image] = release
    new_fronts: dict[int, list[int]] = {}
    fronts: set[int] = set()
    with (mbdump / "cover_art_archive.cover_art_type").open(encoding="utf-8") as f:
        for line in f:
            image_id, art_type = line.rstrip("\n").split("\t")
            if art_type != FRONT_TYPE:
                continue
            image = int(image_id)
            if image in recent:
                new_fronts.setdefault(recent[image], []).append(image)
            elif image in cover_release:
                fronts.add(image)
    for ids in new_fronts.values():
        ids.sort()
    return cover_release, new_fronts, fronts


def best_front(caa: dict) -> str | None:
    """The image id to use as a release's cover, from coverartarchive.org/release/<mbid>.

    Same priority as the 2025 fetch: approved front, then pending front, then any approved
    image. Only images with a 500 px thumbnail ("500", or "large" on older records) count.
    """
    best, best_rank = None, 0
    for img in caa.get("images", []):
        if not {"500", "large"} & set(img.get("thumbnails", {})):
            continue
        front = img.get("front") or "Front" in img.get("types", [])
        approved = img.get("approved", False)
        rank = 3 if front and approved else 2 if front else 1 if approved else 0
        if rank > best_rank:
            best, best_rank = str(img["id"]), rank
    return best


def download(session, cover: Cover, image_root: Path, sleep: Callable[[float], None]) -> None:
    """Fetch the 500 px image to image_root/cover.file, through a .part file so a crash leaves no stub."""
    if not (config.is_uuid(cover.mbid) and cover.cover_id.isdigit()):
        raise ValueError(f"unsafe ids {cover.mbid!r} {cover.cover_id!r}")
    path = image_root / cover.file
    if path.exists():  # already here, from a run whose log line was lost
        return
    if not path.parent.is_dir():  # the 256 prefix folders exist; never create folders on the drive
        raise FileNotFoundError(path.parent)
    resp = get_with_backoff(session, f"{CAA}/{cover.mbid}/{cover.cover_id}-500.jpg", sleep)
    if resp is None:
        raise LookupError(f"no 500 px image for {cover.mbid}/{cover.cover_id}")
    if not resp.content.startswith(b"\xff\xd8"):
        raise ValueError(f"not a JPEG: {cover.mbid}/{cover.cover_id}")
    part = path.with_name(path.name + ".part")
    part.write_bytes(resp.content)
    if path.exists():  # another run got there in the meantime; never replace a cover
        part.unlink()
        return
    os.replace(part, path)


@dataclass
class Counts:
    keep: int = 0
    download: int = 0
    lookup: int = 0
    dropped: int = 0
    pending: int = 0
    fetched: int = 0
    no_cover: int = 0
    failed: int = 0


def covers_after(plan: Plan, landed: dict[str, Cover], releases: set[str]) -> list[Cover]:
    """The full cover list once this update's landed images are in, one primary per release.

    Images an earlier version of the plan fetched (fronts uploaded since the last dump, so
    still valid) stay in as extras.
    """
    to_look_up = set(plan.lookup)
    covers = plan.keep + [c for c in plan.download if c.file in landed]
    covers += [c for c in landed.values() if c.primary and c.mbid in to_look_up]
    files = {c.file for c in covers}
    covers += [replace(c, primary=False) for c in landed.values() if c.mbid in releases and c.file not in files]
    has_primary = {c.mbid for c in covers if c.primary}
    for i, c in enumerate(covers):  # a planned primary that failed to download leaves its release without one
        if c.mbid not in has_primary:
            covers[i] = replace(c, primary=True)
            has_primary.add(c.mbid)
    return covers


def run(
    date: str,
    since: str,
    session,
    limit: int | None = None,
    dry_run: bool = False,
    workers: int = 16,
    sleep: Callable[[float], None] = time.sleep,
) -> Counts:
    image_root = config.require_image_root()
    previous = load_covers(config.META_DIR / f"covers-{since}.jsonl")
    with (config.META_DIR / f"releases-{date}.jsonl").open(encoding="utf-8") as f:
        releases = {json.loads(line)["id"] for line in f if line.strip()}
    previous_ids = {int(c.cover_id) for covers in previous.values() for c in covers}
    cover_release, new_fronts, fronts = load_cover_dump(
        config.DATA_DIR / f"musicbrainz-dump-{date}" / "mbdump", previous_ids, since
    )
    plan = plan_covers(previous, releases, cover_release, new_fronts, fronts)

    log_path = config.META_DIR / f"fetched-{date}.jsonl"
    landed: dict[str, Cover] = {}  # file -> cover, for images this update put on the drive
    no_cover: set[str] = set()     # releases whose lookup found no usable image
    if log_path.exists():
        with log_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # blank, or cut short by a crash: that item reruns
                    continue
                if r["cover_id"] is None:
                    no_cover.add(r["mbid"])
                else:
                    landed[r["file"]] = Cover(**r)
    looked_up = no_cover | {c.mbid for c in landed.values() if c.primary}  # only lookups land primaries

    downloads = [c for c in plan.download if c.file not in landed]
    lookups = [m for m in plan.lookup if m not in looked_up]
    # Alternate the two kinds of work so a --limit run exercises both
    pending: list[Cover | str] = [x for x in chain(*zip_longest(downloads, lookups)) if x is not None]
    counts = Counts(keep=len(plan.keep), download=len(plan.download), lookup=len(plan.lookup),
                    dropped=len(plan.dropped), pending=len(pending))
    print(f"{counts.keep:,} covers kept, {counts.dropped:,} dropped, {counts.download:,} new fronts on known "
          f"releases, {counts.lookup:,} releases to look up; {counts.pending:,} to fetch", flush=True)
    if dry_run:
        return counts
    pending = pending[:limit]

    lock = threading.Lock()
    log = log_path.open("a+", encoding="utf-8")
    if log.tell() > 0:
        log.seek(log.tell() - 1)
        if log.read(1) != "\n":
            log.write("\n")  # a crash cut the last line short; start the next one clean

    def record(entry: dict) -> None:
        with lock:
            log.write(json.dumps(entry) + "\n")
            log.flush()

    def work(item: Cover | str) -> bool:
        """True when an image landed on the drive."""
        if isinstance(item, Cover):
            cover = item
        else:
            resp = get_with_backoff(session, f"{CAA}/{item}", sleep)
            cover_id = best_front(resp.json()) if resp is not None else None
            if cover_id is None:
                record({"mbid": item, "cover_id": None})
                return False
            cover = Cover(item, cover_id, new_file(item, cover_id), True)
        download(session, cover, image_root, sleep)
        landed[cover.file] = cover
        record(asdict(cover))
        return True

    start, done = time.monotonic(), 0
    todo = iter(pending)
    in_flight: dict = {}
    try:
        with ThreadPoolExecutor(workers) as pool:
            # Keep a few items queued per worker and top up as each one finishes, so a slow item
            # (one that keeps failing retries for minutes) never leaves the other workers idle,
            # and 800k futures never exist at once.
            while True:
                for item in todo:
                    in_flight[pool.submit(work, item)] = item
                    if len(in_flight) >= workers * 4:
                        break
                if not in_flight:
                    break
                finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in finished:
                    item = in_flight.pop(future)
                    done += 1
                    try:
                        if future.result():
                            counts.fetched += 1
                        else:
                            counts.no_cover += 1
                    except Exception as e:  # logged and left pending, so a rerun retries it
                        counts.failed += 1
                        print(f"failed {item}: {type(e).__name__}: {e}", flush=True)
                        if not (image_root / "00").is_dir():
                            pool.shutdown(cancel_futures=True)
                            raise SystemExit("The cover drive went away. Remount it and rerun to resume.") from e
                    if done % 1000 == 0:
                        rate = done / (time.monotonic() - start)
                        print(f"{done:,}/{len(pending):,} at {rate:.1f}/s, "
                              f"{(len(pending) - done) / rate / 3600:.1f} h left, {counts.failed:,} failed", flush=True)
    finally:
        log.close()

    covers = covers_after(plan, landed, releases)
    out = config.META_DIR / f"covers-{date}.jsonl"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text("".join(json.dumps(asdict(c)) + "\n" for c in covers))
    os.replace(tmp, out)
    print(f"{counts.fetched:,} fetched, {counts.no_cover:,} without a usable cover, {counts.failed:,} failed; "
          f"{len(covers):,} covers -> {out}", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", required=True, help="the new dump's date, YYYY-MM-DD")
    parser.add_argument("--since", required=True, help="the previous dump's date; fronts uploaded since count as new")
    parser.add_argument("--limit", type=int, help="fetch at most N items this run")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="print the backlog and exit")
    args = parser.parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=args.workers)
    session.mount("https://", adapter)
    session.mount("http://", adapter)  # coverartarchive.org redirects to archive.org mirrors
    run(args.date, args.since, session, args.limit, args.dry_run, args.workers)


if __name__ == "__main__":
    main()
