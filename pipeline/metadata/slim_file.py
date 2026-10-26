"""Slim a MusicBrainz JSON release dump down to the fields the catalog uses.

    python -m pipeline.metadata.slim_file DUMP/release.tar.xz --date 2026-10-24

Streams mbdump/release out of the archive through `tar`, so the 250 GB of JSON inside never
lands on disk. Keeps releases that have cover art. Writes META_DIR/releases-<date>.jsonl and
META_DIR/artists-<date>.json; fetch_new_covers and build_catalog_db read them.
Dumps: https://data.metabrainz.org/pub/musicbrainz/data/json-dumps/
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from pipeline import config


def _drop_empty(d: dict) -> dict:
    return {k: v for k, v in d.items() if v not in (None, "", [])}


def slim_record(obj: dict) -> dict | None:
    """{"release": slim row, "artists": {id: {name, aliases?, disambiguation?}}}, or None without art."""
    art = obj.get("cover-art-archive") or {}
    if not art.get("front") and not art.get("artwork"):
        return None
    media = obj.get("media") or []
    credits = [c["artist"] for c in obj.get("artist-credit") or [] if c.get("artist", {}).get("id")]
    release = {
        "title": obj.get("title"),
        "id": obj.get("id"),
        "artist_ids": [a["id"] for a in credits] or None,
        "asin": obj.get("asin"),
        "disambiguation": obj.get("disambiguation"),
        "format": media[0].get("format") if media else None,
        "aliases": [a["name"] for a in obj.get("aliases") or [] if a.get("name")],
        "release_group": (obj.get("release-group") or {}).get("id"),
    }
    artists = {
        a["id"]: {"name": a.get("name", ""), **_drop_empty({  # the catalog build needs a name, even ""
            "aliases": [x["name"] for x in a.get("aliases") or [] if x.get("name")],
            "disambiguation": a.get("disambiguation"),
        })}
        for a in credits
    }
    return {"release": _drop_empty(release), "artists": artists}


def slim_dump(dump: Path, releases_out: Path, artists_out: Path) -> int:
    """Stream the dump into releases_out (JSONL) and artists_out (JSON). Returns releases kept."""
    tar = subprocess.Popen(["tar", "-xOf", str(dump), "mbdump/release"], stdout=subprocess.PIPE)
    assert tar.stdout is not None
    artists: dict[str, dict] = {}
    kept = seen = bad = 0
    start = time.monotonic()
    tmp = releases_out.with_name(releases_out.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as out:
            for line in tar.stdout:
                seen += 1
                try:
                    slim = slim_record(json.loads(line))
                except (ValueError, TypeError, AttributeError, KeyError):  # one bad line should not cost an hour
                    bad += 1
                    print(f"skipped a malformed line {seen:,}: {line[:80]!r}", flush=True)
                    continue
                if slim is not None:
                    kept += 1
                    out.write(json.dumps(slim["release"], ensure_ascii=False) + "\n")
                    artists.update(slim["artists"])  # later credits win, as in 2025
                if seen % 250_000 == 0:
                    print(f"{seen:,} read, {kept:,} kept, {seen / (time.monotonic() - start):,.0f} lines/s", flush=True)
        if tar.wait() != 0:
            raise SystemExit(f"tar exited with {tar.returncode}; {releases_out} not written")
    except BaseException:
        tar.kill()
        tmp.unlink(missing_ok=True)
        raise
    # Artists first: build_catalog_db picks the newest releases file and expects its artists beside it
    artists_tmp = artists_out.with_name(artists_out.name + ".tmp")
    artists_tmp.write_text(json.dumps(artists, ensure_ascii=False))
    os.replace(artists_tmp, artists_out)
    os.replace(tmp, releases_out)
    print(f"{seen:,} releases read, {kept:,} with cover art, {len(artists):,} artists, {bad:,} malformed lines")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", type=Path, help="release.tar.xz from the JSON dumps")
    parser.add_argument("--date", required=True, help="the dump's date, YYYY-MM-DD; names the output files")
    args = parser.parse_args()
    slim_dump(
        args.dump,
        config.META_DIR / f"releases-{args.date}.jsonl",
        config.META_DIR / f"artists-{args.date}.json",
    )


if __name__ == "__main__":
    main()
